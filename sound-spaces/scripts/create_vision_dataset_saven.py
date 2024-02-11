
import os
import pickle

import numpy as np


def find_label_id_name_proportion(semantic_ids_, id_name_, semantic_type, objects_of_interest_):

    region = False
    if semantic_type == 'region' and objects_of_interest_:
        semantic_type = 'object'
        region = True

    label_name_proportion_ = []
    for label_id in sorted(id_name_[semantic_type]):
        label_name = id_name_[semantic_type][label_id]

        if label_id not in semantic_ids_[semantic_type]:
            continue

        if objects_of_interest_:
            if semantic_type == 'object' and (label_name not in objects_of_interest_):
                semantic_ids_['object'] = np.where(semantic_ids_['object'] == label_id, 0, semantic_ids_['object'])
                continue

        count_ = (semantic_ids_[semantic_type] == label_id).sum()  # Count occurrence of label_id
        proportion_ = round(count_ / semantic_ids_[semantic_type].size, 2)
        label_name_proportion_.append((label_name, proportion_))

    if region:
        semantic_ids_['region'] = np.where(semantic_ids_['object'] != 0, semantic_ids_['region'], 0)
        semantic_type = 'region'

        label_name_proportion_ = []
        for label_id in sorted(id_name_[semantic_type]):
            if label_id != 0 and label_id in semantic_ids_[semantic_type]:
                label_name = id_name_[semantic_type][label_id]

                count_ = (semantic_ids_[semantic_type] == label_id).sum()  # Count occurrence of label_id
                proportion_ = round(count_ / semantic_ids_[semantic_type].size, 2)
                label_name_proportion_.append((label_name, proportion_))

    return label_name_proportion_


if __name__ == '__main__':

    print("Current working directory: {0}".format(os.getcwd()))
    os.chdir('../sound-spaces')
    print("Current working directory: {0}".format(os.getcwd()))

    mp3d_metadata_filepath = r"data/metadata/mp3d_scenes_semantic_data.bin"
    bin_file = open(mp3d_metadata_filepath, "rb")
    scenes = pickle.load(bin_file)
    objects_id_name = pickle.load(bin_file)
    regions_id_name = pickle.load(bin_file)
    bin_file.close()
    id_name = {'object': objects_id_name, 'region': regions_id_name}

    mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
    bin_file = open(mp3d_objects_of_interest_filepath, "rb")
    ooi_objects_id_name = pickle.load(bin_file)
    ooi_regions_id_name = pickle.load(bin_file)
    bin_file.close()
    objects_of_interest = list(ooi_objects_id_name.values())

    scene_obs_dir = 'data/scene_observations_saven/mp3d'

    scene_valid_semantic_nodes = {}
    min_rgb_value = min_depth_value = np.inf
    max_rgb_value = max_depth_value = -np.inf
    for scene in os.listdir(scene_obs_dir):
        scene, fileext = os.path.splitext(scene)
        print("scene: ", scene)

        if fileext != '.pkl':
            continue

        scene_valid_semantic_nodes.setdefault(scene, [])
        with open(scene_obs_dir + os.sep + scene + '.pkl', 'rb') as fo:
            scene_data = pickle.load(fo)

        for node in scene_data:
            point, rotation = node

            rgb_img = scene_data[node]['rgb'][:, :, :3]
            depth_img = scene_data[node]['depth']

            min_rgb_value_, max_rgb_value_ = np.min(rgb_img), np.max(rgb_img)
            min_depth_value_, max_depth_value_ = np.min(depth_img), np.max(depth_img)

            if min_rgb_value_ < min_rgb_value:
                min_rgb_value = min_rgb_value_
            if max_rgb_value_ > max_rgb_value:
                max_rgb_value = max_rgb_value_

            if min_depth_value_ < min_depth_value:
                min_depth_value = min_depth_value_
            if max_depth_value_ > max_depth_value:
                max_depth_value = max_depth_value_

            count = (rgb_img == 0).sum()  # Count occurrence of 0
            proportion = round(count / rgb_img.size, 2)

            if proportion >= 0.75:
                # print("BAD image: 75% of pixels are black")
                continue

            semantic_ids = {'object': scene_data[node]['object_semantic'],
                            'region': scene_data[node]['region_semantic']}
            filename = str(node) + '_' + str(rotation)

            # Objects:
            label_name_proportion = find_label_id_name_proportion(semantic_ids, id_name, 'object', objects_of_interest)

            if not label_name_proportion:
                # print("BAD image: There is no objects_of_interest in the image")
                continue

            objects_name = [t[0] for t in label_name_proportion]

            objects_id = [list(ooi_objects_id_name.keys())[list(ooi_objects_id_name.values()).index(obj)] for obj in
                          objects_name]

            objects_proportion = [t[1] for t in label_name_proportion]

            objects_name_proportion = {}
            for i in range(len(objects_name)):
                objects_name_proportion[objects_name[i]] = objects_proportion[i]

            objects_proportion_sorted = []
            for k, v in sorted(objects_name_proportion.items(), key=lambda x: x[1], reverse=True):
                objects_proportion_sorted.append((k, v))

            objects_id = objects_name = []
            if objects_proportion_sorted[0][1] >= 0.03:
                # Computing frequency compared to most frequent object
                objects_name_proportion = {k: round(v / objects_proportion_sorted[0][1], 2)
                                           for k, v in objects_name_proportion.items()}

                objects_proportion_sorted = []
                for k, v in sorted(objects_name_proportion.items(), key=lambda x: x[1], reverse=True):
                    objects_proportion_sorted.append((k, v))

                objects_name = []
                magic_num = 0.18
                for obj, v in objects_proportion_sorted:
                    if v >= magic_num:
                        objects_name.append(obj)

                objects_id = [list(ooi_objects_id_name.keys())[list(ooi_objects_id_name.values()).index(obj)] for obj in
                              objects_name]
            else:
                # print("BAD image: Most frequent object is taking less than 3% of the image")
                continue

            # regions:
            label_name_proportion = find_label_id_name_proportion(semantic_ids, id_name, 'region', objects_of_interest)

            regions_name = [t[0] for t in label_name_proportion]

            regions_rare = list(set(regions_name) - set(ooi_regions_id_name.values()))

            regions_proportion = [t[1] for t in label_name_proportion]

            for reg in regions_rare:
                # print("Removing rare region: ", reg)
                regions_proportion.pop(regions_name.index(reg))
                regions_name.remove(reg)

            regions_id = [list(ooi_regions_id_name.keys())[list(ooi_regions_id_name.values()).index(reg)] for reg in
                          regions_name]

            regions_name_proportion = {}
            for i in range(len(regions_name)):
                regions_name_proportion[regions_name[i]] = regions_proportion[i]

            regions_proportion_sorted = []
            for k, v in sorted(regions_name_proportion.items(), key=lambda x: x[1], reverse=True):
                regions_proportion_sorted.append((k, v))

            # Computing frequency compared to most frequent region
            regions_name_proportion = {k: round(v / regions_proportion_sorted[0][1], 2)
                                       for k, v in regions_name_proportion.items()}

            regions_proportion_sorted = []
            for k, v in sorted(regions_name_proportion.items(), key=lambda x: x[1], reverse=True):
                regions_proportion_sorted.append((k, v))

            regions_name = []
            magic_num = 0.20
            for reg, v in regions_proportion_sorted:
                if v > magic_num:
                    regions_name.append(reg)

            regions_id = [list(ooi_regions_id_name.keys())[list(ooi_regions_id_name.values()).index(reg)] for reg in
                          regions_name]

            if objects_id and regions_id:
                # print("SAVE")
                scene_valid_semantic_nodes[scene].append({'node': node, 'objects_id': objects_id,
                                                          'regions_id': regions_id})
            # else:
            #     print("DO NOT SAVE")

        nodes_count = 0
        for scene2 in scene_valid_semantic_nodes:
            nodes_count += len(scene_valid_semantic_nodes[scene2])
        print("Total Nodes Count: ", nodes_count)

        print("min_rgb_value: ", min_rgb_value)
        print("max_rgb_value: ", max_rgb_value)
        print("min_depth_value: ", min_depth_value)
        print("max_depth_value: ", max_depth_value)
    modality_max_min_value = {'min_rgb_value': min_rgb_value, 'max_rgb_value': max_rgb_value,
                              'min_depth_value': min_depth_value, 'max_depth_value': max_depth_value}

    with open(os.path.join(r"data/metadata/mp3d_scene_valid_semantic_nodes.bin"), 'wb') as fo:
        pickle.dump(scene_valid_semantic_nodes, fo)
        pickle.dump(modality_max_min_value, fo)
