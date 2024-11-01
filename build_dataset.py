from src.dataset_processor import DatasetProcessor
from src.dataset_processor import dataset_merge_and_delete
import src.dataset_util as dsu
import os
import argparse
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

def build_task_import(processor, import_config, test=False):

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
    if test:
        max_index=min(max_index, 1000)

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

def build_task_make_hard(processor, hard_config, test=False):
    max_images=get_param(processor, hard_config, "max_images", 20000)
    if test:
        max_images=min(max_images, 500)
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
        processor.append_exif_comment(l[i], "hardness="+str(format(hardness[l[i]], ".2f")))
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
    processor.log(f" add_objects: det={model} sz={imgsz} thr={per_class_thr}: {added} detections added")
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
    for idx in tqdm(range(processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/normalise"):
        gts=processor.get_gt(idx)
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
        processor.replace_annotations(idx, gts)

def build_task_generate_backgrounds(class_names, config, face_kp=True, pose_kp=True, test=False):

    loader=config["loader"]
    model=config["model"]
    check_model=None
    if "check_model" in config:
        check_model=config["check_model"]
    check_thr=0.80
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
        if test:
            max_images=min(max_images, 500)
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
        len_ids=len(ids)
        with tqdm(total=max_index, desc=name+"/"+task+"/Generate BG") as pbar:
            for n,i in enumerate(ids):
                prog=max((n*max_index+len_ids-1)//len_ids, index)
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
                processor.add(n, img_path, [])

                if index>=max_index:
                    break
            if max_index>progress:
                pbar.update(max_index-progress)

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
        
        for i in tqdm(range(processor.num_files), desc=processor.dataset_name+"/"+processor.task+"/renaming"):
            new_name=base_name+"{:07d}".format(i)
            processor.append_exif_comment(l[i], "bghardness="+str(format(fp_score[l[i]], ".2f")))
            processor.rename(l[i], new_name)

        processor.reload_files()

        base_name="bgc_"+name+"_"+processor.task[0]

        do_check="person" in class_names and check_model!=None
        if do_check:
            processor.set_yolo_detector(check_model, imgsz=960, thr=0.1, half=True, rect=False, batch_size=24)
        num_deleted=0
        num_ok=0
        for i in tqdm(range(processor.num_files), desc=task+"/BG check"):
            if num_ok>max_images:
                processor.delete(i)
                continue
            if do_check:
                dets=processor.get_detections(i)
                max_conf=0
                num_high=0
                for d in dets:
                    max_conf=max(max_conf, d["confidence"])
                    if d["confidence"]>0.5:
                        num_high+=1
                        very_conf=True
                if max_conf>check_thr or num_high>=4:
                    processor.delete(i)
                    num_deleted+=1
                    continue
                processor.append_exif_comment(i, "check="+str(format(max_conf, ".3f"))+":"+str(num_high))
            new_name=base_name+"{:07d}".format(num_ok)
            processor.rename(i, new_name)
            num_ok+=1

        processor.reload_files()    
        print(f"deleted {num_deleted} of {processor.num_files} (concern of mis-labelling) backgrounds output {num_ok}")
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

def get_expanded_config(path, config_filename, dataset_name):
    config = dsu.load_dictionary(os.path.join(path, config_filename))
    dataset_config=config["datasets"][dataset_name]
    if "general" in dataset_config:
        if "inherit_config" in dataset_config["general"]:
            inherit_from=dataset_config["general"]["inherit_config"]
            if "/" in inherit_from:
                ih_config_filename, ih_dataset_name=inherit_from.split("/")
            else:
                ih_config_filename, ih_dataset_name=config_filename, inherit_from
            ih_config=get_expanded_config(path, ih_config_filename, ih_dataset_name)
            ih_config=copy.deepcopy(ih_config)
            merge(ih_config, dataset_config)
            return ih_config
    return dataset_config

def generate_dataset(path, config_filename, dataset_name, force_generate=False, test=False):
    config = dsu.load_dictionary(os.path.join(path, config_filename))
    
    # create new empty dataset
    dataset_config=get_expanded_config(path, config_filename, dataset_name)
    
    general_config=dataset_config["general"]
    class_names=general_config["class_names"]
    face_kp=general_config["face_kp"]
    pose_kp=general_config["pose_kp"]
    tasks=["val","train"]
    chunk_size=10000
    if "generate" in general_config:
        if general_config["generate"]==False and force_generate==False:
            return None
    if "tasks" in general_config:
        tasks=general_config["tasks"]
    if "chunk_size" in general_config:
        chunk_size=general_config["chunk_size"]

    unique_dataset_name=dsu.unique_dataset_name(dataset_name)
    yaml_path=dsu.make_dataset_yaml(unique_dataset_name, class_names=class_names, face_kp=face_kp, pose_kp=pose_kp)

    if "merge" in dataset_config:
        datasets_to_merge=dataset_config["merge"]
        for d in datasets_to_merge:
            to_merge_yaml=generate_dataset(path, config_filename, d, force_generate=True, test=test)
            dataset_merge_and_delete(to_merge_yaml, yaml_path)
        for task in ["val","train"]:
            x=DatasetProcessor(yaml_path, task=task, append_log=True)
            x.log("======================")
            x.log(" Final merged stats: "+str(x.basic_stats()))
        return yaml_path 

    for task in tasks:
        
        processor=DatasetProcessor(yaml_path, task=task, append_log=True, face_kp=face_kp, pose_kp=pose_kp)

        # import 

        import_config=dataset_config["import"]
        build_task_import(processor, import_config, test=test)

        # hard subset

        if "make_hard" in dataset_config:
            hard_config=dataset_config["make_hard"]
            build_task_make_hard(processor, hard_config, test=test)
        
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
        if bg_config["loader"]!="None":
            bg_yaml=build_task_generate_backgrounds(class_names, bg_config, face_kp=face_kp, pose_kp=pose_kp, test=test) 
            
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
    
    total_elapsed=int(time.time()-start_time)
    x=DatasetProcessor(yaml_path, task="val", append_log=True)

    x.log("======================")
    x.log(f" Total elapsed time {total_elapsed//3600} hours {(total_elapsed//60)%60} mins")
    x.log(" Final merged stats: "+str(x.basic_stats()))
    return yaml_path

def process_dataset(config_filename, test=False):
    config = dsu.load_dictionary(config_filename)

    for dataset_name in config["datasets"]:
        generate_dataset(os.path.split(config_filename)[0], os.path.split(config_filename)[1], dataset_name, test=opt.test)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='process_dataset.py')
    parser.add_argument('--config', type=str, default="dataset_config.yaml", help='Configuration to use (json/yaml)')
    parser.add_argument('--test', action='store_true', help='run in test mode (limit number of images)')
    opt = parser.parse_args()
    process_dataset(opt.config, test=opt.test)