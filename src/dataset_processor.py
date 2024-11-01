import src.dataset_util as dsu
import ultralytics
import cv2
import os
import random
import shutil
import pandas as pd
import torch
import concurrent.futures
import gc

class DatasetProcessor:
    def __init__(self, ds_yaml, task="val", class_names=None, chunk_size=0, append_log=False, face_kp=True, pose_kp=True):
        self.chunk_size=chunk_size
        self.num_chunks=0
        self.ds_yaml=ds_yaml
        self.task=task
        self.reload_files()
        assert(self.ds_files!=None)
        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.dataset_path=os.path.dirname(ds_yaml)
        self.append_log=append_log
        if class_names==None:
            class_names=self.ds_class_names
        self.class_names=class_names
        self.yolo=None
        self.yolo_nms_iou=0.6
        self.yolo_max_det=600
        self.yolo_det_conf=0.001

    def reload_files(self):
        self.ds_files, self.ds_class_names, self.dataset_name=dsu.get_all_dataset_files(self.ds_yaml, self.task, "image")
        self.num_files=len(self.ds_files)
        if self.chunk_size!=0:
            self.num_chunks=(self.num_files+self.chunk_size-1)//self.chunk_size
            self.current_chunk=0
            self.ds_files_all=self.ds_files
            self.num_files_all=self.num_files
            self.set_chunk(0)

    def __len__(self):
        return len(self.ds_files)
    
    def __getitem__(self, index):
        return self.ds_files[index]

    def log(self, txt):
        print("log: "+str(txt))
        if self.append_log:
            dsu.append_comments(self.ds_yaml, txt)

    def basic_stats(self):
        class_counts=[0]*len(self.class_names)
        image_count=0
        background_count=0
        face_point_count=0
        pose_point_count=0
        small_persons_count=0
        person_class=self.get_class_index("person")
        for i in range(self.num_files):
            gts=self.get_gt(i)
            has_small_persons=False
            for g in gts:
                class_counts[g["class"]]+=1
                if dsu.has_face_points(g):
                    face_point_count+=1
                if dsu.has_pose_points(g):
                    pose_point_count+=1
                if g["class"]==person_class:
                    if not dsu.is_large(g):
                        has_small_persons=True
            if has_small_persons:
                small_persons_count+=1
            image_count+=1
        return {"task":self.task, "class_counts":class_counts, "image_count":image_count, "image_small_person_count":small_persons_count, "has_face_kp":face_point_count, "has_pose_kp":pose_point_count}

    def log_stats(self):
        stats=self.basic_stats()
        self.log(str(stats))

    def set_chunk(self, cn):
        self.current_chunk=cn
        start=cn*self.chunk_size
        end=min((cn+1)*self.chunk_size, self.num_files_all)
        self.num_files=end-start
        self.ds_files=self.ds_files_all[start:end]

    def get_gt(self, index):
        label_file=self.ds_files[index]["label"]
        gts=dsu.load_ground_truth_labels(label_file)
        if gts==None:
            return None
        gt_class_remap=[self.class_names.index(x) if x in self.class_names else -1 for x in self.ds_class_names]
        for g in gts:
            g["class"]=gt_class_remap[g["class"]]
        ret=[g for g in gts if g["class"]!=-1]
        return ret
    
    def get_gt_file_path(self, index):
        label_file=self.ds_files[index]["label"]
        return label_file
    
    def get_class_names(self):
        return self.class_names
    
    def get_class_index(self, class_name):
        if class_name in self.class_names:
            return self.class_names.index(class_name)
        return -1
    
    def add(self, name, img_file, dets):
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        shutil.copyfile(img_file, dst_img)
        dsu.image_append_exif_comment(dst_img, "origin="+img_file)
        an_txt=dsu.write_annotations(dets, include_face=self.face_kp, include_pose=self.pose_kp)
        with open(dst_label, 'w') as file:
            file.write(an_txt)

    def delete(self, index):
        dsu.rm(self.ds_files[index]["label"])
        dsu.rm(self.ds_files[index]["image"])
        self.ds_files[index]["label"]=None
        self.ds_files[index]["image"]=None

    def rename(self, index, name):
        dst_img=self.dataset_path+"/"+self.task+"/images/"+name+".jpg"
        dst_label=self.dataset_path+"/"+self.task+"/labels/"+name+".txt"
        dsu.rename(self.ds_files[index]["label"], dst_label)
        dsu.rename(self.ds_files[index]["image"], dst_img)
        self.ds_files[index]["label"]=dst_label
        self.ds_files[index]["image"]=dst_img

    def append_exif_comment(self, index, comment):  
        dsu.image_append_exif_comment(self.ds_files[index]["image"], comment)
    
    def set_yolo_detector(self, yolo, imgsz=640, thr=0.001, rect=False, half=True, batch_size=32, augment=False):
        
        if self.yolo!=None:
            del self.yolo
            self.yolo=None

        gc.collect()
        torch.cuda.empty_cache() # try hard not to run out of GPU memory
        self.yolo_cache={}

        if yolo==None:
            return
    
        self.imgsz=imgsz
        self.yolo_batch_size=batch_size
        self.yolo_det_conf=thr
        self.yolo_rect=rect
        self.yolo_half=half
        self.num_gpus=torch.cuda.device_count()
        self.yolo_task="detect"
        if type(yolo) is str:
            if "pose" in yolo or "face" in yolo or "full" in yolo:
                self.yolo_task="pose"
            if ":" in yolo:
                t=yolo.split(":")
                yolo=t[0]
                self.imgsz=int(t[1])
                self.yolo_batch_size=self.yolo_batch_size//2

            if yolo.endswith(".engine"):
                if not os.path.isfile(yolo):
                    yolo=yolo[:-len(".engine")]+".pt"
                    if os.path.isfile(yolo):
                        self.generate_engine(yolo)

            if self.num_gpus==1:
                self.yolo=ultralytics.YOLO(yolo, task=self.yolo_task, verbose=False)
                assert self.yolo!=None
                self.yolo_class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
            else:
                self.yolo=[None]*self.num_gpus
                assert self.yolo!=None
                for i in range(self.num_gpus):
                    self.yolo[i]=ultralytics.YOLO(yolo, task=self.yolo_task, verbose=False)
                    self.yolo[i]=self.yolo[i].to("cuda:"+str(i))
                self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        elif type(yolo) is list:
            self.yolo=yolo
            self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        else:
            self.yolo=yolo
            self.yolo_class_names=[self.yolo.names[i] for i in range(len(self.yolo.names))]
        self.yolo_cache={}

        # map the detected classes back to our class set
        # assume if our class set has things like 'vehicle' then we would want any standard coco classes
        # like 'car' to map to that
        self.det_class_remap=[-1]*len(self.yolo_class_names)
        synonyms={'vehicle':['car','bicycle','motorcycle','train','truck','bus','airplane','boat'],
                  'animal':['cat','dog','horse','sheep','cow','bird','elephant','bear','zebra','giraffe']}
        self.can_detect=[False]*len(self.class_names)
        for i,x in enumerate(self.yolo_class_names):
            if x in self.class_names:
                self.det_class_remap[i]=self.class_names.index(x)
                self.can_detect[self.class_names.index(x)]=True
            else:
                for y in synonyms:
                    if y in self.class_names:
                        if x in synonyms[y]:
                            self.det_class_remap[i]=self.class_names.index(y) 
                            self.can_detect[self.class_names.index(y)]=True   
    
    def get_img(self, index):
        return cv2.imread(self.ds_files[index]["image"])
    
    def get_img_path(self, index):
        return self.ds_files[index]["image"]
    
    def get_label_path(self, index):
        return self.ds_files[index]["label"]
    
    def get_img_name(self, index):
        label_file=self.ds_files[index]["label"]
        return dsu.name_from_file(label_file)

    def replace_annotations(self, index, an):
        txt=dsu.write_annotations(an, include_face=self.face_kp, include_pose=self.pose_kp)
        label_file=self.get_gt_file_path(index)
        with open(label_file, 'w') as file:
            file.write(txt)

    def get_detections(self, index, det_thr=0.01):
        img=self.ds_files[index]["image"]
        out_det=[]
        if self.yolo!=None:
            if not index in self.yolo_cache:
                if self.num_gpus==1:
                    input_frames=[self.get_img(x) for x in range(index, min(self.num_files, index+self.yolo_batch_size))]
                    #print(f"run yolo on {input_frames}")
                    batch_result=self.yolo(input_frames, conf=self.yolo_det_conf, iou=self.yolo_nms_iou, max_det=self.yolo_max_det, agnostic_nms=False, half=self.yolo_half, imgsz=self.imgsz, verbose=False, rect=self.yolo_rect)
                    self.yolo_cache={(i+index):batch_result[i] for i in range(len(input_frames))}
                else:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        thread_state=[]
                        for i in range(self.num_gpus):
                            start=min(self.num_files, index+i*self.yolo_batch_size)
                            end=min(self.num_files, index+(i+1)*self.yolo_batch_size)
                            if start!=end:
                                input_frames=[self.get_img(x) for x in range(start, end)]
                                future = executor.submit(self.yolo[i], input_frames, conf=self.yolo_det_conf, iou=self.yolo_nms_iou, max_det=self.yolo_max_det, agnostic_nms=False, half=self.yolo_half, imgsz=self.imgsz, verbose=False, rect=self.yolo_rect)
                                thread_state.append({"frames":input_frames, "future":future, "start":start})
                        self.yolo_cache={}
                        for i in range(len(thread_state)):
                            batch_result=thread_state[i]["future"].result()
                            input_frames=thread_state[i]["frames"]
                            
                            for j in range(len(input_frames)):
                                self.yolo_cache[j+thread_state[i]["start"]]=batch_result[j]

            assert(index in self.yolo_cache)
            
            results=self.yolo_cache[index]
            
            det_boxes = results.boxes.xyxyn.tolist() # center
            det_classes = results.boxes.cls.tolist()
            det_confidences = results.boxes.conf.tolist()

            det_kp_list=None
            det_kp_conf_list=None
            if hasattr(results, "keypoints") and results.keypoints!=None:
                det_kp_list=results.keypoints.xyn.tolist()
                if results.keypoints.has_visible:
                    det_kp_conf_list=results.keypoints.conf.tolist()

            num_det=len(det_boxes)
            indexes=[i for i in range(num_det) if self.det_class_remap[int(det_classes[i])]!=-1 and det_confidences[i]>det_thr] # remove detected classes we are not interested in
            
            for i in indexes:
                det={"box":det_boxes[i], 
                     "class":self.det_class_remap[int(det_classes[i])],
                     "confidence":det_confidences[i],
                     "face_points":[0]*15,
                     "pose_points":[0]*51}
                
                fp, pp=dsu.unpack_yolo_keypoints(det_kp_list, det_kp_conf_list, i)
                if fp!=None and self.face_kp:
                    det["face_points"]=fp
                if pp!=None and self.pose_kp:
                    det["pose_points"]=pp
                if det["class"]!=-1:
                    out_det.append(det)

        return out_det
    
def dataset_merge_and_delete(src, dest):
    dsu.append_comments(dest, "====== Merge dataset "+src+" ======")
    with open(src, 'r') as file:
        for line in file:
            if line.startswith("#"):
                dsu.append_comments(dest, line[1:].strip())    
    
    dest_path=dsu.get_dataset_path(dest)
    
    for task in ["val","train"]:
        x=DatasetProcessor(src, task=task)
        for i in range(x.num_files):    
            img=x.get_img_path(i)
            label=x.get_label_path(i)
            dsu.rename(img, dest_path+"/"+task+"/images/"+os.path.basename(img))
            dsu.rename(label, dest_path+"/"+task+"/labels/"+os.path.basename(label))
    
    dsu.rmdir(os.path.dirname(src))