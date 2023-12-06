from collections import defaultdict

import torch
import numpy as np

from ss_baselines.savi.models.rollout_storage import ExternalMemory

np.random.seed(0)


class IPRLPretrainingRolloutStorage:

    def __init__(
        self,
        num_steps,
        num_envs,
        observation_space,
        action_space,
        use_external_memory,
        external_memory_size,
        external_memory_capacity,
        external_memory_dim,
    ):
        self.observations = {}

        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        if action_space.__class__.__name__ == "ActionSpace":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.prev_actions = torch.zeros(num_steps + 1, num_envs, action_shape)
        if action_space.__class__.__name__ == "ActionSpace":
            self.actions = self.actions.long()
            self.prev_actions = self.prev_actions.long()

        self.masks = torch.zeros(num_steps + 1, num_envs, 1)

        self.use_external_memory = use_external_memory
        self.em_size = external_memory_size
        self.em_capacity = external_memory_capacity
        self.em_dim = external_memory_dim
        # This is kept outside for for backward compatibility with _collect_rollout_step
        self.em_masks = torch.zeros(num_steps + 1, num_envs, self.em_size)
        if use_external_memory:
            self.em = ExternalMemory(
                num_envs, self.em_size, self.em_capacity,
                self.em_dim, num_copies=num_steps + 1
            )
        else:
            self.em = None

        self.num_steps = num_steps
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)
        self.em_masks = self.em_masks.to(device)
        if self.use_external_memory:
            self.em.to(device)

    def insert(
        self,
        observations,
        actions,
        not_done_masks,
        em_features,
        dones,
    ):
        for sensor in observations:
            if sensor == "depth":
                self.observations[sensor][self.step + 1].copy_(
                    np.squeeze(observations[sensor], axis=4)
                )
            elif sensor == "generated_instruction":
                self.observations[sensor][self.step + 1].copy_(
                    np.squeeze(observations[sensor], axis=1)
                )
            elif sensor == "oracle_action_sensor":
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor].reshape(-1, 1)
                )
            else:
                self.observations[sensor][self.step + 1].copy_(
                    observations[sensor]
                )

        self.actions[self.step].copy_(actions)
        self.prev_actions[self.step + 1].copy_(actions)
        self.masks[self.step + 1].copy_(not_done_masks)
        if self.use_external_memory:
            self.em.insert(em_features, not_done_masks)
            self.em_masks[self.step + 1].copy_(self.em.masks)

        self.step = self.step + 1

    def after_update(self):
        for sensor in self.observations:
            self.observations[sensor][0].copy_(
                self.observations[sensor][self.step]
            )

        self.masks[0].copy_(self.masks[self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        if self.use_external_memory:
            self.em_masks[0].copy_(self.em_masks[self.step])
        self.step = 0

    def recurrent_generator(self, num_mini_batch):
        num_processes = self.masks.size(1)
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            actions_batch = []
            prev_actions_batch = []
            masks_batch = []
            if self.use_external_memory:
                em_store_batch = []
                em_masks_batch = []
            else:
                em_store_batch = None
                em_masks_batch = None

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][: self.step, ind]
                    )

                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                if self.use_external_memory:
                    em_store_batch.append(self.em.memory[:, : self.step, ind])
                    em_masks_batch.append(self.em_masks[: self.step, ind])

            T, N = self.step, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            prev_actions_batch = torch.stack(prev_actions_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)

            if self.use_external_memory:
                # This is a (em_size, num_steps, bs, em_dim) tensor
                em_store_batch = torch.stack(em_store_batch, 2)
                # This is a (num_steps, bs, em_size) tensor
                em_masks_batch = torch.stack(em_masks_batch, 1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            
            for sensor in observations_batch:
                observations_batch[sensor] = self._flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = self._flatten_helper(T, N, actions_batch)
            prev_actions_batch = self._flatten_helper(T, N, prev_actions_batch)
            masks_batch = self._flatten_helper(T, N, masks_batch)
            if self.use_external_memory:
                em_store_batch = em_store_batch.view(-1, T * N, self.em_dim)
                em_masks_batch = self._flatten_helper(T, N, em_masks_batch)

            yield (
                observations_batch,
                actions_batch,
                prev_actions_batch,
                masks_batch,
                em_store_batch,
                em_masks_batch,
            )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        r"""Given a tensor of size (t, n, ..), flatten it to size (t*n, ...).

        Args:
            t: first dimension of tensor.
            n: second dimension of tensor.
            tensor: target tensor to be flattened.

        Returns:
            flattened tensor of size (t*n, ...)
        """
        return tensor.view(t * n, *tensor.size()[2:])

    @property
    def external_memory(self):
        return self.em.memory

    @property
    def external_memory_masks(self):
        return self.em_masks

    @property
    def external_memory_idx(self):
        return self.em.idx