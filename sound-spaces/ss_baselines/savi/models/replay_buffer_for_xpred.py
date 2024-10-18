import torch
import numpy as np


class ReplayBufferForXPred:
    def __init__(
        self,
        buffer_size: int,
        max_steps: int,
        num_envs: int,
        encoded_observation_dim: int,
        category_belief_dim: int,
        location_belief_dim: int,
        max_instr_len: int,
    ):
        """
        Args:
            buffer_size (int): The number of episode data.
            max_steps (int): The maximum number of steps in a episode.
            num_envs (int): The number of environments.
            encoded_observation_dim (int): The dimention of encoded observation.
            category_belief_dim (int): The dimention of category belief.
            location_belief_dim (int): The dimention of location belief.
            max_instr_len (int): The maximum length of instruction.
        """

        self.buffer_size = buffer_size
        self.max_steps = max_steps
        self.num_envs = num_envs
        self.max_instr_len = max_instr_len
        self.encoded_observation_dim = encoded_observation_dim
        self.category_belief_dim = category_belief_dim
        self.location_belief_dim = location_belief_dim

        self.encoded_observations = torch.zeros(
            self.buffer_size,
            self.num_envs,
            self.max_steps,
            self.encoded_observation_dim,
        )
        self.episode_lengths = torch.zeros(
            self.buffer_size,
            self.num_envs,
            1,
            dtype=torch.int,
        )
        if self.category_belief_dim is not None:
            self.category_beliefs = torch.zeros(
                self.buffer_size,
                self.num_envs,
                self.max_steps,
                self.category_belief_dim,
            )
        if location_belief_dim is not None:
            self.location_beliefs = torch.zeros(
                self.buffer_size,
                self.num_envs,
                self.max_steps,
                self.location_belief_dim,
            )
        self.instructions = torch.zeros(
            self.buffer_size,
            self.num_envs,
            self.max_steps,
            self.max_instr_len,
        )
        
        self.data_count = torch.zeros(self.num_envs, dtype=torch.int)
        self.steps = torch.zeros(self.num_envs, dtype=torch.int)
    
    def to(self, device: str) -> None:
        self.encoded_observations = self.encoded_observations.to(device)
        self.episode_lengths = self.episode_lengths.to(device)

        if self.category_belief_dim is not None:
            self.category_beliefs = self.category_beliefs.to(device)
        
        if self.location_belief_dim is not None:
            self.location_beliefs = self.location_beliefs.to(device)
    
        self.instructions = self.instructions.to(device)
    
    def insert(
        self,
        encoded_observations: torch.Tensor, # (num_envs, encoded_observation_dim)
        dones: torch.Tensor, # (num_envs,)
        instructions: torch.Tensor, # (num_envs, max_instr_len)
        category_beliefs: torch.Tensor, # (num_envs, category_num)
        location_beliefs: torch.Tensor, # (num_envs, location_dim)
    ) -> None:
        if torch.any(self.steps >= self.max_steps):
            for i in range(self.num_envs):
                if self.steps[i] < self.max_steps:
                    self.encoded_observations[self.data_count[i], i, self.steps[i], :] = encoded_observations[i]
                    self.instructions[self.data_count[i], i, self.steps[i], :] = instructions[i]

                    if category_beliefs is not None:
                        self.category_beliefs[self.data_count[i], i, self.steps[i], :] = category_beliefs[i]

                    if location_beliefs is not None:
                        self.location_beliefs[self.data_count[i], i, self.steps[i], :] = location_beliefs[i]
                    
                    self.steps[i] += 1 
        else:
            self.encoded_observations[self.data_count, torch.arange(0, self.num_envs), self.steps, :] = encoded_observations
            self.instructions[self.data_count, torch.arange(0, self.num_envs), self.steps, :] = instructions

            if category_beliefs is not None:
                self.category_beliefs[self.data_count, torch.arange(0, self.num_envs), self.steps, :] = category_beliefs

            if location_beliefs is not None:
                self.location_beliefs[self.data_count, torch.arange(0, self.num_envs), self.steps, :] = location_beliefs
            
            self.steps += 1

        for i in range(self.num_envs):
            if dones[i]:
                self.episode_lengths[self.data_count[i], i] = self.steps[i]
                self.data_count[i] = self.data_count[i] + 1
                self.steps[i] = 0

                if self.data_count[i] >= self.buffer_size:
                    self.encoded_observations[:, i:i+1, :, :] = torch.cat(
                        [
                            self.encoded_observations[1:, i:i+1, :, :],
                            torch.zeros(1, 1, self.max_steps, self.encoded_observation_dim).to(self.encoded_observations.device),
                        ],
                        dim=0,
                    )
                    self.episode_lengths[:, i:i+1, :] = torch.cat(
                        [
                            self.episode_lengths[1:, i:i+1, :],
                            torch.zeros(1, 1, 1).to(self.episode_lengths.device),
                        ],
                        dim=0,
                    )
                    self.instructions[:, i:i+1, :, :] = torch.cat(
                        [
                            self.instructions[1:, i:i+1, :, :],
                            torch.zeros(1, 1, self.max_steps, self.max_instr_len).to(self.instructions.device),
                        ],
                        dim=0,
                    )

                    if category_beliefs is not None:
                        self.category_beliefs[:, i:i+1, :, :] = torch.cat(
                            [
                                self.category_beliefs[1:, i:i+1, :, :],
                                torch.zeros(1, 1, self.max_steps, self.category_belief_dim).to(self.category_beliefs.device),
                            ],
                            dim=0,
                        )

                    if location_beliefs is not None:
                        self.location_beliefs[:, i:i+1, :, :] = torch.cat(
                            [
                                self.location_beliefs[1:, i:i+1, :, :],
                                torch.zeros(1, 1, self.max_steps, self.location_belief_dim).to(self.location_beliefs.device),
                            ],
                            dim=0,
                        )
                    self.data_count[i] = self.data_count[i] - 1

    def random_sampling(self, num: int) -> None:
        indices = torch.arange(self.buffer_size).repeat(self.num_envs).view(
            self.num_envs, self.buffer_size,
        ).permute(1,0) < self.data_count.unsqueeze(0)
        indices = indices.to(self.encoded_observations.device)

        encoded_observations = self.encoded_observations[indices]
        instructions = self.instructions[indices]
        episode_lengths = self.episode_lengths[indices]

        # 各エピソードのどのStep位置を取ってくるかをサンプリング
        episode_lengths = episode_lengths.view(-1,)
        seq_lengths = torch.floor(
            episode_lengths * (torch.rand(episode_lengths.size()).to(episode_lengths.device)**0.5)
        ).int()
        # 平方根をとることで、p(x) = 2xからのサンプリングを行なっている

        # sampleされたseq_lengthsをもとにmaskを作成
        masks = (
            torch.arange(self.max_steps).expand(len(episode_lengths), self.max_steps).to(seq_lengths.device)
        ) > (
            seq_lengths.unsqueeze(1).expand(len(episode_lengths), self.max_steps)
        )

        sampled_indices = torch.randint(0, len(encoded_observations), (num,)).to(self.encoded_observations.device)
        encoded_observations = encoded_observations[sampled_indices]
        masks = masks[sampled_indices]
        seq_lengths = seq_lengths[sampled_indices]
        instructions = instructions[sampled_indices]

        encoded_observations[masks] = 0
        instructions = instructions[torch.arange(0, instructions.size(0)), seq_lengths, :]

        if self.category_belief_dim is not None:
            category_beliefs = self.category_beliefs[indices]
            category_beliefs = category_beliefs[sampled_indices]
            category_beliefs = category_beliefs[torch.arange(0, category_beliefs.size(0)), seq_lengths, :]
        else:
            category_beliefs = None
        
        if self.location_belief_dim is not None:
            location_beliefs = self.location_beliefs[indices]
            location_beliefs = location_beliefs[sampled_indices]
            location_beliefs = location_beliefs[torch.arange(0, location_beliefs.size(0)), seq_lengths, :]
        else:
            location_beliefs = None

        return encoded_observations, masks, seq_lengths, category_beliefs, location_beliefs, instructions
