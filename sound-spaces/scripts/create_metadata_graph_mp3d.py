import csv
import os
import pickle

import habitat_sim


if __name__ == '__main__':

    mp3d_data_path = r"data/scene_datasets/mp3d"
    mp3d_metadata_filepath = r"data/metadata/mp3d_scenes_semantic_data.bin"

    if not os.path.exists(mp3d_metadata_filepath):
        print(mp3d_metadata_filepath + "does not exists, so computing it...")

        scenes = {}
        objects_id_name = {}
        regions_id_name = {}
        for root, dirs, filenames in os.walk(mp3d_data_path):
            for a_file in filenames:
                filename, fileext = os.path.splitext(a_file)

                if fileext != '.glb':
                    continue

                data_filepath = os.path.join(root, a_file)
                print("data_filepath: ", data_filepath)

                scene = {'objects': {}, 'regions': {}, 'regions_objects': {}}

                sim_cfg = habitat_sim.SimulatorConfiguration()
                agent_cfg = habitat_sim.AgentConfiguration()
                sim_cfg.scene_id = data_filepath
                sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

                for obj in sim.semantic_scene.objects:
                    # print("obj: ", obj, obj.id, obj.region.category.name(), obj.category.name())
                    scene['objects'].setdefault(obj.category.name(), 0)
                    scene['objects'][obj.category.name()] += 1

                    objects_id_name.setdefault(obj.category.index(), obj.category.name())

                for reg in sim.semantic_scene.regions:
                    # print("reg: ", reg, reg.id, reg.category.name())
                    scene['regions'].setdefault(reg.category.name(), 0)
                    scene['regions'][reg.category.name()] += 1

                    regions_id_name.setdefault(reg.category.index(), reg.category.name())

                    # print("reg.objects: ", len(reg.objects))
                    for obj in reg.objects:
                        # print("obj: ", obj, obj.id, obj.category.name())
                        scene['regions_objects'].setdefault(reg.category.name(), {})
                        scene['regions_objects'][reg.category.name()].setdefault(obj.category.name(), 0)
                        scene['regions_objects'][reg.category.name()][obj.category.name()] += 1

                sim.close()
                scenes[filename] = scene

        # print("scenes: ", scenes)
        print("object_id_name: ", objects_id_name)
        print("region_id_name: ", regions_id_name)

        output_file = open(mp3d_metadata_filepath, "wb")
        pickle.dump(scenes, output_file)
        pickle.dump(objects_id_name, output_file)
        pickle.dump(regions_id_name, output_file)
        output_file.close()
    else:
        bin_file = open(mp3d_metadata_filepath, "rb")
        scenes = pickle.load(bin_file)
        objects_id_name = pickle.load(bin_file)
        regions_id_name = pickle.load(bin_file)
        bin_file.close()

    print("scenes: ", len(scenes))
    # print("scenes: ", scenes)
    object_id_name_sorted = [(k, v) for k, v in sorted(objects_id_name.items(), key=lambda x: x[0], reverse=False)]
    print("objects: ", len(object_id_name_sorted))
    print("object_id_name_sorted: ", object_id_name_sorted)
    region_id_name_sorted = [(k, v) for k, v in sorted(regions_id_name.items(), key=lambda x: x[0], reverse=False)]
    print("regions: ", len(region_id_name_sorted))
    print("region_id_name_sorted: ", region_id_name_sorted)
    print("\n" + "-" * 100)

    objects = {}
    regions = {}
    regions_objects = {}
    for scene in scenes:
        for obj in scenes[scene]['objects']:
            objects.setdefault(obj, 0)
            objects[obj] += scenes[scene]['objects'][obj]
        for reg in scenes[scene]['regions']:
            regions.setdefault(reg, 0)
            regions[reg] += scenes[scene]['regions'][reg]
        for reg in scenes[scene]['regions_objects']:
            regions_objects.setdefault(reg, {})
            for obj in scenes[scene]['regions_objects'][reg]:
                regions_objects[reg].setdefault(obj, 0)
                regions_objects[reg][obj] += scenes[scene]['regions_objects'][reg][obj]

    print("objects: ", len(objects))
    # print("objects: ", sorted(objects.keys()))
    print([(k, v) for k, v in sorted(objects.items(), key=lambda x: x[1], reverse=True)])

    print("regions: ", len(regions))
    print([(k, v) for k, v in sorted(regions.items(), key=lambda x: x[1], reverse=True)])
    print("")

    print("regions_objects: ", len(regions_objects))
    # print("regions_objects: ", regions_objects)
    print("\n" + "-" * 100)

    objects_of_interest = ['bathtub', 'bed', 'cabinet', 'chair', 'chest_of_drawers', 'clothes', 'counter', 'cushion',
                           'fireplace', 'gym_equipment', 'picture', 'plant', 'seating', 'shower', 'sink', 'sofa',
                           'stool', 'table', 'toilet', 'towel', 'tv_monitor']  # semantic AVN's 21 objects
    ooi_objects_id_name = {}
    for obj in sorted(objects_of_interest):
        ooi_objects_id_name[len(ooi_objects_id_name)] = obj
    print("ooi_objects_id_name: ", len(ooi_objects_id_name))
    print("ooi_objects_id_name: ", ooi_objects_id_name)
    print("\n" + "-" * 100)

    # Computing object-to-object and object-to-region relations
    
    max_objects_in_a_region = 0
    min_objects_in_a_region = len(objects)
    min_num_regions_needed = 15
    regions_objects_frequency_percent = {}
    for reg in sorted(regions_objects):
        print("reg: ", reg, ", objects: ", len(regions_objects[reg]))
        print("There are " + str(regions[reg]) + " " + reg)
        print("There are " + str(len(regions_objects[reg])) + " objects in " + reg)
        if regions[reg] < min_num_regions_needed:
            print("There are very less examples of this region, so skipping\n")
            continue
        objects_sorted = []
        total_num_objects = 0
        for k, v in sorted(regions_objects[reg].items(), key=lambda x: x[1], reverse=True):
            if k in objects_of_interest:
                objects_sorted.append((k, v))
                total_num_objects += v
        print("total_num_objects: ", total_num_objects)
        print(objects_sorted)

        factor = 1.0 / total_num_objects
        objects_frequency_percent = {k: round(v * factor, 2) for k, v in objects_sorted}
        print("objects_frequency_percent: ", objects_frequency_percent)

        # Computing frequency compared to most frequent object
        objects_frequency_percent = {k: round(v/objects_frequency_percent[objects_sorted[0][0]], 2)
                                     for k, v in objects_frequency_percent.items()}
        print("objects_frequency_percent: ", objects_frequency_percent)
        regions_objects_frequency_percent[reg] = objects_frequency_percent

        if len(regions_objects[reg]) > max_objects_in_a_region:
            max_objects_in_a_region = len(regions_objects[reg])
        if len(regions_objects[reg]) < min_objects_in_a_region:
            min_objects_in_a_region = len(regions_objects[reg])
        print("")

    print("\n" + "-" * 100)
    print("max_objects_in_a_region: ", max_objects_in_a_region)
    print("min_objects_in_a_region: ", min_objects_in_a_region)
    print("\n" + "-" * 100)

    print("regions_objects_frequency_percent: ", len(regions_objects_frequency_percent))

    ooi_regions_id_name = {}
    for reg in sorted(regions_objects_frequency_percent):
        ooi_regions_id_name[len(ooi_regions_id_name)] = reg
    print("ooi_regions_id_name: ", ooi_regions_id_name)
    print("\n" + "-" * 100)

    mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
    output_file = open(mp3d_objects_of_interest_filepath, "wb")
    pickle.dump(ooi_objects_id_name, output_file)
    pickle.dump(ooi_regions_id_name, output_file)
    output_file.close()

    csv_path = r"data/metadata/mp3d_graph_object.csv"
    with open(csv_path, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Sounding Objects", "Objects", "Regions"])

    magic_num = 0.14
    cluster_regions = {}
    for obj in sorted(objects_of_interest):
        print("obj: ", obj)
        best_regions = {}
        best_objects = {}
        for reg in regions_objects_frequency_percent:
            # print("obj -> reg: ", obj, "->", reg)
            if obj in regions_objects_frequency_percent[reg]:
                for o in regions_objects_frequency_percent[reg]:
                    if o != obj and regions_objects_frequency_percent[reg][obj] >= magic_num and \
                            regions_objects_frequency_percent[reg][o] >= magic_num:
                        best_objects.setdefault(o, {})
                        best_objects[o][reg] = regions_objects_frequency_percent[reg][o]

                        best_regions.setdefault(reg, 0)
                        best_regions[reg] += 1

        if not len(best_objects):
            print("DECREASE magic_num or min_num_regions_needed")
            exit()

        print("best_objects: ", len(best_objects), best_objects)
        best_objects = {obj2: round(sum(best_objects[obj2].values())/len(best_objects[obj2]), 2)
                        for obj2 in best_objects}  # Averaging
        best_objects = [(k, v) for k, v in sorted(best_objects.items(), key=lambda x: x[1], reverse=True)]
        print("best_objects: ", len(best_objects), best_objects)

        total_best_regions = sum(best_regions.values())
        best_regions = {reg: round(best_regions[reg]/total_best_regions, 2) for reg in best_regions}
        best_regions_sorted = [(k, v) for k, v in sorted(best_regions.items(), key=lambda x: x[1], reverse=True)]
        print("best_regions_sorted: ", len(best_regions_sorted), best_regions_sorted)
        print("")

        for reg in best_regions:
            for r in regions:
                if reg != r and r in best_regions:
                    cluster_regions.setdefault(reg, {})
                    cluster_regions[reg].setdefault(r, 0)
                    cluster_regions[reg][r] += 1

        with open(csv_path, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([obj, ", ".join([o_n[0] for o_n in best_objects]),
                             ", ".join([o_n[0] for o_n in best_regions_sorted])])

    print("\n" + "-" * 100)
    print("cluster_regions: ", cluster_regions)
    print("\n" + "-" * 100)

    # Computing region-to-region and region-to-object relations

    csv_path = r"data/metadata/mp3d_graph_region.csv"
    with open(csv_path, 'w') as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Regions", "Objects", "Other Regions"])

    cluster_regions2 = {}
    magic_num_region = 1.0
    for reg in sorted(cluster_regions):
        print("reg: ", reg)
        # cluster_regions_sorted = [(k, v) for k, v in sorted(cluster_regions[reg].items(), key=lambda x: x[1], reverse=True)]
        # print("cluster_regions_sorted: ", len(cluster_regions_sorted), cluster_regions_sorted)
        cluster_regions2[reg] = [k for k, v in sorted(cluster_regions[reg].items(), key=lambda x: x[1], reverse=True)]

        cluster_regions_sorted = []
        total_num_regions = 0
        for k, v in sorted(cluster_regions[reg].items(), key=lambda x: x[1], reverse=True):
            cluster_regions_sorted.append((k, v))
            total_num_regions += v
        print("total_num_regions: ", total_num_regions)
        print("cluster_regions_sorted: ", len(cluster_regions_sorted), cluster_regions_sorted)
        # print(cluster_regions_sorted)

        factor = 1.0 / total_num_regions
        regions_frequency_percent = {k: round(v * factor, 2) for k, v in cluster_regions_sorted}
        print("regions_frequency_percent: ", regions_frequency_percent)

        # Computing frequency compared to most frequent region
        regions_frequency_percent = {k: round(v / regions_frequency_percent[cluster_regions_sorted[0][0]], 2)
                                     for k, v in regions_frequency_percent.items()}
        print("regions_frequency_percent: ", regions_frequency_percent)

        best_regions = []
        for r in regions_frequency_percent:
            if regions_frequency_percent[r] >= magic_num_region:
                best_regions.append(r)
        print("best_regions: ", best_regions)

        if not len(best_regions):
            print("INCREASE magic_num_region")
            exit()

        objects_frequency_percent_sorted = []
        for k, v in sorted(regions_objects_frequency_percent[reg].items(), key=lambda x: x[1], reverse=True):
            if v >= magic_num:
                objects_frequency_percent_sorted.append(k)
        print("objects_frequency_percent_sorted: ", objects_frequency_percent_sorted)

        with open(csv_path, 'a') as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow([reg, ", ".join(objects_frequency_percent_sorted), ", ".join(best_regions)])

        print("")

    print("cluster_regions2: ", cluster_regions2)
    print("\n" + "-" * 100)
    
    # Clustering regions

    def fill_set(my_reg, idx, my_cluster_regions2, my_cluster_regions_sets):

        for i in range(idx):
            while True:
                my_reg2 = my_cluster_regions2[my_reg][i]
                my_in_a_region_set = False
                for my_r_set in my_cluster_regions_sets:
                    if my_reg2 in my_r_set:
                        my_in_a_region_set = True
                        break
                if not my_in_a_region_set:
                    # print("adding: ", my_reg2)
                    my_in_a_region_set = False
                    for my_r_set in my_cluster_regions_sets:
                        if my_reg in my_r_set:
                            my_in_a_region_set = True
                            my_r_set.add(my_reg2)
                            break
                    if not my_in_a_region_set:
                        my_cluster_regions_sets.append({my_reg2})
                    my_reg = my_reg2
                else:
                    break

        return my_cluster_regions_sets

    first_how_many = 2
    cluster_regions_sets = []  # list of sets
    for reg in sorted(cluster_regions2):
        # print("reg: ", reg)

        in_a_region_set = False
        for r_set in cluster_regions_sets:
            if reg in r_set:
                in_a_region_set = True
                break
        if not in_a_region_set:
            cluster_regions_sets.append({reg})
            cluster_regions_sets = fill_set(reg, first_how_many, cluster_regions2, cluster_regions_sets)

        # print("cluster_regions_sets: ", cluster_regions_sets)
        # print("")

    print("cluster_regions_sets: ", len(cluster_regions_sets), cluster_regions_sets)
    print("\n" + "-" * 100)
