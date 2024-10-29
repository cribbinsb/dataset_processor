from dataset_processor import DatasetProcessor
import dataset_util as dsu
import os
import argparse
import json
import math
import time
import copy
from tqdm import tqdm

def get_param(processor, config, param, default_value):
    if param+"_val" in config and processor.task=="val":
        return config[param+"_val"]
    if param+"_train" in config and processor.task=="train":
        return config[param+"_train"]
    if param in config:
        return config[param]
    return default_value

def build_task_import(processor, import_config):

    task=processor.task
    class_names=processor.class_names
    name=processor.dataset_name

    processor.log(f"import; class_names={class_names}")

    loader=import_config["loader"]
    max_images=get_param(processor, import_config, "max_images", 10000000)
    filter_single_large=get_param(processor, import_config, "filter_single_large", False)
    
    loader_fn=dsu.get_loader(loader)
    o=loader_fn(class_names=class_names, task=task, ds_params=import_config)

    processor.log("origin info: "+o.get_info())
    maps=o.get_category_maps()
    for c in class_names:
        if c in maps:
            o.add_category_mapping(maps[c], c)

    ids=o.get_image_ids()
    if len(ids) != len(set(ids)):
        print("WARNING: Duplicate ids")
    processor.log(f"found {len(ids)} ids, limiting to {max_images}")

    base=name+"_"+task[0]

    index=0
    no_annotations=0
    filtered_single_large=0

    max_index=min(len(ids), max_images)

    with tqdm(total=max_index, desc=name+"/"+task+"/Generate") as pbar:
        for i in ids:
            img_path=o.get_img_path(i)
            if not os.path.isfile(img_path):
                continue
            dets=o.get_annotations(i)
            if dets==None:
                no_annotations+=1
                continue

            if filter_single_large==True:
                if len(dets)==1:
                    if dsu.box_a(dets[0]["box"])>0.20:
                        filtered_single_large+=1
                        continue
                
            n=base+"{:07d}".format(index)
            index+=1
            pbar.update(1)
            processor.add(n, img_path, dets)
            if index>=max_index:
                break
 
    processor.log(f"imported {index} images")
    if no_annotations!=0:
        processor.log(f"import_dataset: filtered {no_annotations} images due to 'None' annotations")
    if filtered_single_large!=0:
        processor.log(f"import_dataset: filtered {filtered_single_large} images due to 'filtered_single_large'")
    processor.reload_files()
    processor.log_stats()

def build_task_make_hard(processor, hard_config):
    max_images=get_param(processor, hard_config, "max_images", 20000)
    model=hard_config["model"]
    rare_classes=[]
    if "rare_classes" in hard_config:
        rare_classes=hard_config["rare_classes"]

    processor.log(f"make_hard: reduce {processor.num_files} images to {max_images}")
    hardness=[]
    num=processor.num_files
    if num<=max_images:
        processor.log("make_hard: nothing to do")
        return

    processor.set_yolo_detector(model, imgsz=480, thr=0.01, batch_size=48)    
    class_names=processor.class_names
    nc=len(class_names)
    rare_class_map=[0.0]*nc
    for index,c in enumerate(class_names):
        if c in rare_classes:
            rare_class_map[index]=1.0

    for i in tqdm(range(num), desc=processor.dataset_name+"/"+processor.task+"/measure hardness"):
        gts=processor.get_gt(i)
        dets=processor.get_detections(i)
        det_matched, gt_matched=dsu.match_boxes(dets, gts, 0.5)        
        hardness_fn=0.0
        hardness_fp=0.0
        hardness_rare_class=0.0
        for g in gts:
            hardness_rare_class+=rare_class_map[g["class"]]
        num_gt=0
        for j in range(len(gt_matched)):
            cls=gts[j]["class"]
            if processor.can_detect[cls]==False: # dpn't count a penalty for GTs we couldn't possibly match
                assert(gt_matched[j]==-1)
                continue
            num_gt+=1
            hardness_fn+=1.0
            if gt_matched[j]!=-1:
                hardness_fn-=dets[gt_matched[j]]["confidence"]
        for j in range(len(det_matched)):
            if det_matched[j]==-1:
                if dets[j]["confidence"]>0.5:
                    hardness_fp+=(2*dets[j]["confidence"]-1.0)
        hardness_tot=hardness_fp+hardness_fn+4*hardness_rare_class
        # the following is just so we don't end up picking all the images with loads of GTS
        if num_gt>0:
            hardness_tot/=math.sqrt(float(num_gt)) 
        hardness.append(hardness_tot)
    
    l=list(range(processor.num_files))
    l=[x for _, x in sorted(zip(hardness, l), reverse=True)]
        
    base_name=processor.dataset_name+"_"+processor.task[0]+"h"
    for i in tqdm(range(max_images, processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/deleting"):
        processor.delete(l[i])
        
    for i in tqdm(range(max_images), desc=processor.dataset_name+"/"+processor.task+"/renaming"):
        new_name=base_name+"{:07d}".format(i)
        processor.rename(l[i], new_name)

    processor.reload_files()
    processor.log_stats()

def build_task_add_objects(processor, add_object_config):
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    thr=get_param(processor, add_object_config, "thr", 0.9)
    iou_thr=get_param(processor, add_object_config, "iou_thr", 0.5)
    min_sz=get_param(processor, add_object_config, "min_sz", 0.05*0.05)
    per_class_thr=get_param(processor, add_object_config, "per_class_thr", [thr]*len(processor.class_names))
    
    processor.set_yolo_detector(model, imgsz=imgsz, thr=thr, half=True, rect=False, batch_size=16)
    added=[0]*len(processor.class_names)
    for i in tqdm(range(processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/add objects"):
        gts=processor.get_gt(i)
        if gts==None:
            continue
        detections=processor.get_detections(i)
        missing_detections=[]
        for d in detections:
            best_iou, best_match, match_index=dsu.best_iou_match(d, gts)
            thr_use=per_class_thr[d["class"]]
            if best_iou<iou_thr and d["confidence"]>thr_use and dsu.box_a(d["box"])>min_sz:
                missing_detections.append(d)
                added[d["class"]]+=1
        processor.replace_annotations(i, gts+missing_detections)
    processor.log(f" add_objects: det={model} sz={imgsz} thr={thr}: {added} detections added")
    processor.log_stats()

def build_task_add_pose(processor, add_object_config):
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    thr=get_param(processor, add_object_config, "thr", 0.2)
    min_sz=get_param(processor, add_object_config, "min_sz", 0.05*0.05)

    processor.set_yolo_detector(model, imgsz=imgsz, batch_size=16, thr=thr)
    added=0
    desc=processor.dataset_name+"/"+processor.task+"/add pose"
    for i in tqdm(range(processor.num_files), desc=desc):
        gts=processor.get_gt(i)
        if gts==None:
            continue
        detections=processor.get_detections(i)
        added+=dsu.add_pose_points_to_gts(detections, gts, min_sz=min_sz, class_names=processor.get_class_names())
        processor.replace_annotations(i, gts)
    processor.log(desc+f" added {added} pose point sets with det {model} sz {imgsz}")
    processor.log_stats()
    
def build_task_add_faces(processor, add_object_config):
    model=add_object_config["model"]
    imgsz=add_object_config["sz"]
    kp_thr=get_param(processor, add_object_config, "kp_thr", 0.9)
    box_thr=get_param(processor, add_object_config, "box_thr", 0.9)

    processor.set_yolo_detector(model, imgsz=imgsz, thr=min(kp_thr, box_thr), batch_size=16)

    faces_added=0
    face_kp_added=0
    for i in tqdm(range(processor.num_files), desc=processor.task+f"/add faces bthr:{box_thr} kpthr:{kp_thr}"):
        gts=processor.get_gt(i)
        if gts==None: 
            continue       

        detections=processor.get_detections(i)
        faces=dsu.filter_faces_by_persons(detections, gts, class_names=processor.get_class_names())
        for f in faces:
            # check for existing face
            iou, gt, index=dsu.best_iou_match(f, gts) 
            if iou<0.5:
                # new face
                if f["confidence"]>box_thr:
                    gts.append(f)
                    faces_added+=1
            else:
                # existing face; consider adding the face points from detection into that face
                # if it doesn't already have face points
                if dsu.has_face_points(gt)==False and f["confidence"]>kp_thr:
                    gt["face_points"]=f["face_points"]
                    face_kp_added+=1
        processor.replace_annotations(i, gts)
    processor.log(f" add_faces det={model} sz={imgsz} thrs={box_thr},{kp_thr}: {faces_added} faces added; {face_kp_added} face keypoint sets added")
    processor.log_stats()

def build_task_normalise(processor):
    """
    Normalise does some random cleanup 
    - make sure all the co-ordinates are in range
    - make sure there aren't multiple overlapping instances of the same class
    """
    for i in tqdm(range(processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/normalise"):
        gts=processor.get_gt(i)
        if gts==None:
            continue
        for g in gts:
            if "confidence" in g:
                g["confidence"]=dsu.clip01(g["confidence"])
            if "box" in g:
                for i in range(4):
                    g["box"][i]=dsu.clip01(g["box"][i])
            if "face_points" in g:
                for i in range(15):
                    g["face_points"][i]=dsu.clip01(g["face_points"][i])
            if "pose_points" in g:
                for i in range(51):
                    g["pose_points"][i]=dsu.clip01(g["pose_points"][i])
        # dedup does a kind of NMS style removal of multiple overlapping instances of the same class
        gts=dsu.dedup_gt(gts, iou_thr=0.5)
        processor.replace_annotations(i, gts)

def build_task_generate_backgrounds(class_names, config, face_kp=True, pose_kp=True):

    loader=config["loader"]
    model=config["model"]
    check_model=None
    if "check_model" in config:
        check_model=config["check_model"]
    check_thr=0.85
    if "check_thr" in config:
        check_thr=config["check_thr"]
    all_classes=class_names+["confused", "background"]
    name=loader.lower()
    if "loader" in name:
        name=name[0:name.find("loader")]
    unique_dataset_name=dsu.unique_dataset_name(name+"_bgtemp")
    yaml_path=dsu.make_dataset_yaml(unique_dataset_name, class_names=class_names, face_kp=face_kp, pose_kp=pose_kp)

    for task in ["val","train"]:

        processor=DatasetProcessor(yaml_path, task=task, append_log=False)
        max_images=get_param(processor, config, "max_images", 1000)

        if max_images==0:
            continue

        loader_fn=dsu.get_loader(loader)
        o=loader_fn(class_names=all_classes, task=task, ds_params=None)

        maps=o.get_category_maps()
        for c in all_classes:
            if c in maps:
                o.add_category_mapping(maps[c], c)

        ids=o.get_image_ids()
        num=min(len(ids), max_images)
        max_index=min(len(ids), 10*num)
        print(f"generate_backgrounds: max_images {max_images} imported {len(ids)} images; max_index={max_index}")
        index=0
        progress=0
        prog_i=0
        prog_idx=0
        with tqdm(total=1000, desc=name+"/"+task+"/Generate BG") as pbar:
            for i in ids:
                prog_i=(i*1000)//len(ids)
                prog=max(prog_i, prog_idx)
                if prog>progress:
                    pbar.update(prog-progress)
                    progress=prog

                img_path=o.get_img_path(i)
                if not os.path.isfile(img_path):
                    continue
                dets=o.get_annotations(i)
                if dets==None:
                    continue

                non_bg=0
                bg_class=len(all_classes)-1
                for d in dets:
                    if d["class"]!=bg_class:
                        non_bg+=1
                if non_bg!=0:
                    continue

                n="t"+"{:07d}".format(index)
                index+=1
                prog_idx=(index*1000)//max_index
                processor.add(n, img_path, [])

                if index>=max_index:
                    break

        processor.reload_files()

        processor.set_yolo_detector(model, imgsz=640, thr=0.1, half=True, rect=False, batch_size=32)
        
        fp_score=[0.0]*processor.num_files
        for i in tqdm(range(processor.num_files), desc=name+"/"+task+"/measure"):    
            dets=processor.get_detections(i)
            fp_tot=0.0
            for d in dets:
                fp_tot+=d["confidence"]
            fp_score[i]=fp_tot
    
        l=list(range(processor.num_files))
        l=[x for _, x in sorted(zip(fp_score, l), reverse=True)]
        
        base_name="bg_"+name+"_"+processor.task[0]

        num=min(processor.num_files, max_images)
        for i in tqdm(range(num, processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/deleting"):
            processor.delete(l[i])
        
        for i in tqdm(range(num), desc=processor.dataset_name+"/"+processor.task+"/renaming"):
            new_name=base_name+"{:07d}".format(i)
            processor.rename(l[i], new_name)

        processor.reload_files()

        if "person" in class_names and check_model!=None:
            processor.set_yolo_detector(check_model, imgsz=640, thr=0.1, half=True, rect=False, batch_size=32)
            num_deleted=0
            for i in tqdm(range(processor.num_files), desc=task+"/BG check"):
                dets=processor.get_detections(i)
                very_conf=False
                for d in dets:
                    if d["confidence"]>check_thr:
                        very_conf=True
                if very_conf:
                    processor.delete(i)
                    num_deleted+=1

            print(f"deleted {num_deleted} of {processor.num_files} (concern of mis-labelling)")
    return yaml_path

def merge(a: dict, b: dict, path=[]):
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

def process_dataset(config):
    with open(config) as json_file:
        config = json.load(json_file)

    for dataset_name in config["datasets"]:

        # create new empty dataset

        if "default" in config:
            dataset_config=copy.deepcopy(config["default"])
            merge(dataset_config, config["datasets"][dataset_name])
        else:
            dataset_config=config["datasets"][dataset_name]

        general_config=dataset_config["general"]
        class_names=general_config["class_names"]
        face_kp=general_config["face_kp"]
        pose_kp=general_config["pose_kp"]
        tasks=["val","train"]
        chunk_size=10000
        if "tasks" in general_config:
            tasks=general_config["tasks"]
        if "chunk_size" in general_config:
            chunk_size=general_config["chunk_size"]

        unique_dataset_name=dsu.unique_dataset_name(dataset_name)
        yaml_path=dsu.make_dataset_yaml(unique_dataset_name, class_names=class_names, face_kp=face_kp, pose_kp=pose_kp)

        for task in tasks:
            
            processor=DatasetProcessor(yaml_path, task=task, append_log=True)

            # import 

            import_config=dataset_config["import"]
            build_task_import(processor, import_config)

            # hard subset

            if "make_hard" in dataset_config:
                hard_config=dataset_config["make_hard"]
                build_task_make_hard(processor, hard_config)
            
            start_time=time.time()
            processor.chunk_size=chunk_size
            processor.reload_files() 

            for chunk in range(processor.num_chunks):
                print(f"\n==== {processor.dataset_name} {processor.task} : chunk {chunk} of {processor.num_chunks} ====")
                t=time.time()
                processor.set_chunk(chunk)

                # add objects

                if "add_objects" in dataset_config:
                    add_object_config=dataset_config["add_objects"]
                    for detector_config in add_object_config:
                        build_task_add_objects(processor, detector_config)

                # add pose points

                if "add_pose" in dataset_config:
                    add_object_config=dataset_config["add_pose"]
                    for detector_config in add_object_config:
                        build_task_add_pose(processor, detector_config)
                
                # add faces

                if "add_faces" in dataset_config:
                    add_object_config=dataset_config["add_faces"]
                    for detector_config in add_object_config:
                        build_task_add_faces(processor, detector_config)

                # normalise

                build_task_normalise(processor)
                
                elapsed=time.time()-t
                total_elapsed=time.time()-start_time
                remaining=(processor.num_chunks-chunk-1)*total_elapsed/(chunk+1.0)
                
                print(f"chunk took {(int(elapsed))} seconds; total {(int)(total_elapsed/60.0)}mins remaining {(int(remaining/60.0))}mins")
        
        # generate background images

        if "add_backgrounds" in dataset_config:
            bg_config=dataset_config["add_backgrounds"]
            bg_yaml=build_task_generate_backgrounds(class_names, bg_config, face_kp=face_kp, pose_kp=pose_kp) 
            
            for task in tasks:
                processor=DatasetProcessor(bg_yaml, task=task, append_log=False, class_names=class_names)
                dest_path=dsu.get_dataset_path(yaml_path)+"/"+task
                for i in tqdm(range(processor.num_files), desc="BG"+"/"+task+"/copying"):
                    src_img=processor.get_img_path(i)
                    src_label=processor.get_label_path(i)
                    dst_img=dest_path+"/images/"+os.path.split(src_img)[1]
                    dst_label=dest_path+"/labels/"+os.path.split(src_label)[1]
                    dsu.rename(src_img, dst_img)
                    dsu.rename(src_label, dst_label)
            path=dsu.get_dataset_path(bg_yaml)
            print(f"generate_backgrounds: delete folder {path}...")
            dsu.rmdir(path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--config', type=str, default="dataset_config.json", help='JSON configuration to use')
    opt = parser.parse_args()
    process_dataset(opt.config)