import os
import yaml
import json
import cv2
import math
import exif
from pathlib import Path
import numpy as np
import exif
import shutil
from datetime import datetime
from src.loaders.coco_loader import CocoLoader
from src.loaders.openimages_loader import OpenImagesLoader
from src.loaders.o365_loader import O365Loader
from src.loaders.widerface_loader import WiderfaceLoader
from src.loaders.roboflow_loader import RoboflowLoader
from src.loaders.weapondetection_loader import WeaponDetectionLoader

def mldata_folder():
    d=os.environ.get('MLDATA_LOCATION')
    if d==None:
        return "/mldata"
    return d

def unique_dataset_name(dataset_name):
    ver=0
    while True:
        name=dataset_name
        name+="-"+datetime.today().strftime('%y%m%d')
        if ver!=0:
            name+="-v"+str(ver)
        if not os.path.isdir(mldata_folder()+"/"+name):
            break
        ver+=1
    return name

def name_from_file(x):
    ext=""
    if isinstance(x, str):
        if ":" in x:
            t=x.split(":")
            x=t[0]
            ext+=t[1]+" "
        if x.endswith(".engine"):
            ext+="TRT "
        if len(ext)>0:
            ext="("+ext[:-1]+")"
    return os.path.splitext(os.path.basename(x))[0]+ext

def makedir(x):
    Path(x).mkdir(parents=True, exist_ok=True)

def rename(x, y):
    os.rename(x, y)

def rm(x):
    os.remove(x)

def rmdir(x):
    if os.path.isdir(x):
        try:
            shutil.rmtree(x)
        except OSError as e:
            print("rmdir error: %s - %s." % (e.filename, e.strerror))
     
def get_all_dataset_files(dataset, task="val", file_key="both"):
    """
    'Load' a yaml dataset and return arrays of all the corresponding image and label files

    Args:
        dataset: path of yaml file
        task: train or val
        file_key: if 'both' returns img/label where both exist, if 'image' returns all images, if 'label' returns all labels
        
    Returns:
        list of dict, each containing "label" and "image" keys with paths
        list of class names
    """
    
    assert(file_key=="both" or file_key=="image" or file_key=="label")

    with open(dataset) as stream:
        try:
            ds=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None, None, None

    if not "names" in ds:
        print(f"Could not class names in dataset {dataset} yaml")
        return None, None, None
    
    names=ds["names"]

    if "dataset_name" in ds:
        dataset_name=ds["dataset_name"]
    else:
        dataset_name=name_from_file(dataset)

    if not task in ds:
        print(f"Could not find {task} in dataset {dataset} yaml")
        return None, None
    
    ds[task]=str(ds[task])

    val=ds[task].strip()

    if val.startswith('[') and val.endswith(']'):
        val=val[1:-1].split(",")
        val=[v.strip() for v in val]
    else:
        val=[val]

    for i,v in enumerate(val):
        if v.startswith('"') and v.endswith('"'):
            v=v[1:-1]
        if v.startswith("'") and v.endswith("'"):
            v=v[1:-1]
        val[i]=v

    if "path" in ds:
        path=ds["path"]
        if path==".":
            path=os.path.dirname(dataset)
        val=[os.path.join(path, v) for v in val]

    files=[]

    for v in val:
        for x in ["/images","/labels"]:
            if v.endswith(x):
                v=v[:-len(x)]
        img_path=os.path.join(v, "images")
        label_path=os.path.join(v, "labels")
        if file_key=="image" or file_key=="both":
            if os.path.isdir(img_path):
                imgs=os.listdir(img_path)
                for i in imgs:
                    l=os.path.splitext(i)[0]+".txt"
                    img=os.path.join(img_path, i)
                    lab=os.path.join(label_path,l)
                    f={}
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="image" and os.path.isfile(img):
                        f={"image":img, "label":None}
                        files.append(f)
        else:
            if os.path.isdir(label_path):
                labels=os.listdir(label_path)
                for l in labels:
                    i=os.path.splitext(i)[0]+".jpg"
                    img=os.path.join(img_path, i)
                    lab=os.path.join(label_path,l)
                    if os.path.isfile(img) and os.path.isfile(lab):
                        f={"image":img, "label":lab}
                        files.append(f)
                    elif file_key=="label" and os.path.isfile(lab):
                        f={"image":None, "label":lab}
                        files.append(f)

    return files, names, dataset_name

def make_dataset_yaml(name, class_names=None, face_kp=True, pose_kp=True):
    """
    Write out a basic dataset yaml description
    """
    path=mldata_folder()+"/"+name

    txt="# autogenerated dataset YAML file for "+name+"\n"
    txt+="# created: "+datetime.now().strftime("%d/%m/%Y %H:%M:%S")+"\n"
    txt+="dataset_name: "+name+"\n"
    txt+="path: "+path+"\n"
    txt+="train: train/images\n"
    txt+="val: val/images\n"
    txt+="nc: "+str(len(class_names))+"\n"
    if face_kp and pose_kp:
        txt+="kpt_shape: [22, 3]\n"
        txt+="flip_idx: [1, 0, 2, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20]\n"
    elif face_kp:
        txt+="kpt_shape: [5, 3]\n"
        txt+="flip_idx: [1, 0, 2, 4, 3]\n"
    elif pose_kp:
        txt+="kpt_shape: [17, 3]\n"
        txt+="flip_idx: [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]\n"
    txt+="names: "+str(class_names)+"\n"
    
    makedir(path)
    makedir(path+"/train/images")
    makedir(path+"/train/labels")
    makedir(path+"/val/images")
    makedir(path+"/val/labels")
    yaml_file_name=path+"/dataset.yaml"
    with open(yaml_file_name, 'w') as file:
        file.write(txt)
    return yaml_file_name

def load_ground_truth_labels(label_file):
    """
    load one label file

    Args:
        path of label .txt file
        
    Returns:
        list of boxes co-ords [x1,x2,y1,y2]
        list of box classes
    """
    if label_file==None:
        return None
    try:
        if not os.path.isfile(label_file):
            return None
    except:
        print("Exception loading gt "+str(label_file))
        return None
    
    out=[]
    with open(label_file, 'r') as f:
        boxes=[]
        classes=[]
        for line in f:
            vals=[float(x) for x in line.strip().split()]
            box=[vals[1]-0.5*vals[3], vals[2]-0.5*vals[4], vals[1]+0.5*vals[3], vals[2]+0.5*vals[4]]
            cls=int(vals[0]) 
            face_points=[0]*15
            pose_points=[0]*51
            if len(vals)==20 or len(vals)==71:
                face_points=vals[5:20]
            if len(vals)==56:
                pose_points=vals[5:56]
            if len(vals)==71:
                pose_points=vals[20:71]
            gt={"box":box,
                "class":cls,
                "confidence":1.0,
                "face_points":face_points,
                "pose_points":pose_points}
            out.append(gt)
            
    return out

def load_cached_detections(model_path, dataset, f):
    """
    load_cached_detections

    Args:
        path to model .pt file
        path to dataset yaml file
        path to where a label .txt file would be if it existed
        
    Returns:
        flat list of boxes co-ords x1,x2,y1,y2
        list of box classes
        list of confidences
    """
    model=name_from_file(model_path)
    dataset=name_from_file(dataset)
    f=os.path.basename(f)
    path=os.path.join("/mldata/detections",dataset,model,f)

    if not os.path.isfile(path):
        return None, None, None
    
    boxes=[]
    classes=[]
    confidences=[]

    with open(path, 'r') as f:
        for line in f:
            vals=[float(x) for x in line.strip().split()]
            box=[vals[1]-0.5*vals[3], vals[2]-0.5*vals[4], vals[1]+0.5*vals[3], vals[2]+0.5*vals[4]]
            boxes.append(box)
            classes.append(vals[0])
            confidences.append(vals[5])
    return boxes, classes, confidences

def box_i(b1, b2):
    """
    Return area of box intersection
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    return ai

def box_a(b1):
    """
    Return area of box
    """
    return (b1[3]-b1[1])*(b1[2]-b1[0])

def box_w(b1):
    """
    Return width of box
    """
    return b1[2]-b1[0]

def box_iou(b1, b2):
    """
    Computes the iou between two boxes

    Args:
        b1, b2: list [x1,y1,x2,y2] in xyxy format
        
    Returns:
        float iou
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    iou=(iw*ih)/(a1+a2-ai+1e-7)
    return iou

def is_large(x):
    return x["box"][2]-x["box"][0]>=0.1

def has_face_points(x):
    """
    Return True if the annotation has non-trivial face points (i.e. not all 0)
    """
    if not "face_points" in x:
        return False
    t=0
    for i in range(5):
        t+=x["face_points"][3*i+2]
    if t<0.1:
        return False
    return True

def has_pose_points(x):
    """
    Return True if the annotation has non-trivial pose points (i.e. not all 0)
    """
    if not "pose_points" in x:
        return False
    t=0
    for i in range(17):
        t+=x["pose_points"][3*i+2]
    if t<0.1:
        return False
    return True

def better_annotation(a1, a2):
    """
    return True if a1 is 'better' than a2
    better is currently the one that has pose, face points if the other doesn't else the largest
    """
    a1_pp=has_pose_points(a1)
    a2_pp=has_pose_points(a2)
    if a1_pp!=a2_pp:
        return a1_pp
    a1_fp=has_face_points(a1)
    a2_fp=has_face_points(a2)
    if a1_fp!=a2_fp:
        return a1_fp
    return box_a(a1["box"])>box_a(a2["box"])
    
def dedup_gt(gts, iou_thr=0.6):
    """
    Remove multiple overlapping instances of the same class
    This can happen because things that are different in the original dataset may be mapped to the same label
    e.g. head,face or boy,person
    """
    for i,g in enumerate(gts):
        for j,g2 in enumerate(gts):
            if i==j or g2["confidence"]==0 or g["confidence"]==0:
                continue
            if g["class"]==g2["class"]:
                if box_iou(g["box"], g2["box"])>iou_thr:
                    if better_annotation(g, g2): # delete the 'worst' one
                        g2["confidence"]=0
                    else:
                        g["confidence"]=0
    out=[]
    for g in gts:
        if g["confidence"]>0:
            out.append(g)
    return out

def best_iou_match(x, to_check):
    """
    check against list and find match of same class with largest IOU. Retun triple iou, matching item, item index
    """
    best_iou=0
    best_g=None
    best_i=-1
    for i,g in enumerate(to_check):
        if x["class"]==g["class"]:
            iou=box_iou(x["box"],g["box"])
            if iou>best_iou:
                best_iou=iou
                best_g=g
                best_i=i
    return best_iou, best_g, best_i

def kp_iou(kp_gt, kp_det, s, num_pt):
    """
    Computes the keypoint iou between two boxes

    Args:
        kpgt, gpdet: list [x,y,conf]*npoint
        s: GT box area
    Returns:
        float iou 0-1
    """
    # https://learnopencv.com/object-keypoint-similarity/
    if num_pt==5:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025]
    elif num_pt==17:
        scales=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    elif num_pt==22:
        scales=[0.025, 0.025, 0.026, 0.025, 0.025, 0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072, 0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
    else:
        scales=[0.025]*num_pt

    scales=[x*2 for x in scales] # scale=2*sigma

    s=s*0.53 # approximation of shape area from box area
    num=0
    denom=0
    for i in range(num_pt):
        if kp_gt[i*3+2]>0.3: # is point labelled
            dx=kp_gt[i*3+0]-kp_det[i*3+0]
            dy=kp_gt[i*3+1]-kp_det[i*3+1]
            num+=math.exp(-(dx*dx+dy*dy)/(2.0*s*scales[i]*scales[i]+1e-7))
            denom+=1.0
    iou=num/(denom+1e-7)
    return iou

def match_boxes(dets, gts, iou_thr=0.5):
    """
    Matches predictions to ground truth values

    Args:
        det_boxes, gt_boxes list of [x1,y1,x2,y2] in xyxy format
        det_classes, gt_classes: class indexes
        iou_thr: min IOU to count as a match
        note: det boxes/classes should be pre-sorted in descending confidence order
        
    Returns:
        det_matched, gt_matched: corresponding indexes of matches, -1 means for no match
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j in range(len(dets)):
        for i in range(len(gts)):
            if gt_matched[i]==-1 and gts[i]['class']==dets[j]['class'] and box_iou(gts[i]['box'], dets[j]['box'])>iou_thr:
                gt_matched[i]=j
                det_matched[j]=i
                break
    return det_matched, gt_matched

def match_boxes_kp(dets, gts, iou_thr=0.5, face_class=None, person_class=None):
    """
    Match detections to GTs using keypoint IOU
    """
    gt_matched=[-1]*len(gts)
    det_matched=[-1]*len(dets)
    for j in range(len(dets)):
        for i in range(len(gts)):
            if gt_matched[i]==-1 and gts[i]['class']==dets[j]['class']:
                a=box_a(gts[i]['box'])
                if gts[i]['class']==face_class:
                    #biou=box_iou(gts[i]['box'], dets[j]['box'])

                    #fp1=[0.095, 0.152, 1.0, 0.103, 0.154, 1.0, 0.098, 0.158, 1.0, 0.094, 0.162, 1.0, 0.102, 0.164, 1.0]
                    #fp2=[0.09501953423023224, 0.15155945718288422, 0.990234375, 0.10390625149011612, 0.1520467847585678, 0.98974609375, 0.09921874850988388, 0.15789473056793213, 0.990234375, 0.09526367485523224, 0.16252437233924866, 0.9912109375, 0.10234375298023224, 0.16301169991493225, 0.99072265625]
                    #fiou=kp_iou(fp1, fp2, a, 5)

                    #fiou=kp_iou(gts[i]['face_points'], dets[j]['face_points'], a, 5)
                    #if (biou>0.8):
                    #    print(fp1)
                    #    print(fp2)    
                    #    print(f" {biou} {fiou} {a}")
                    
                    if kp_iou(gts[i]['face_points'], dets[j]['face_points'], a, 5)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
                if gts[i]['class']==person_class:
                    if kp_iou(gts[i]['pose_points'], dets[j]['pose_points'], a, 17)>iou_thr:
                        gt_matched[i]=j
                        det_matched[j]=i
                        break
    return det_matched, gt_matched

def display_image_wait_key(image, scale=0, title="no title"):
    h, w, c = image.shape
    if scale==0:
        scale=min(720.0/h, 1280.0/w)
    scaled = cv2.resize(image, (0,0), fx=scale, fy=scale)
    cv2.imshow(title+" "+str(w)+"x"+str(h), scaled)
    r=cv2.waitKey(0)  # Press any key to move to the next image
    cv2.destroyAllWindows()
    if r==27:
        print("Quitting")
        quit()
    return r

def clip01(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return x

def append_comments(fn, x):
    x="#"+x
    x.replace('\n', '\n#')
    if not os.path.isfile(fn):
        print(f"Error: file {fn} does not exist")
        return 
    with open(fn, "a") as f:
        f.write(x+"\n")

def kp_line(img, kp, a, b, c=None):
    height, width, ch = img.shape
    x0=kp[3*a+0]
    y0=kp[3*a+1]
    v0=kp[3*a+2]
    x1=kp[3*b+0]
    y1=kp[3*b+1]
    v1=kp[3*b+2]
    if v0==0:
        return
    if v1==0:
        return
    if c!=None:
        x2=kp[3*c+0]
        y2=kp[3*c+1]
        v2=kp[3*c+2]
        if v2==0:
            return
        x1=0.5*(x1+x2)
        y1=0.5*(y1+y2)
    x0=int(clip01(x0)*width)
    y0=int(clip01(y0)*height)
    x1=int(clip01(x1)*width)
    y1=int(clip01(y1)*height)
    cv2.line(img, (x0, y0), (x1, y1), (255, 0, 0), thickness=2)
    
def draw_boxes(img, an, class_names=None, alt_clr=False):
    height, width, c = img.shape
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 0.5
    fontColor              = (255,255,255)
    thickness              = 2
    lineType               = 2
    
    for a in an:
        b=a["box"]
        x1=int(clip01(b[0])*width)
        y1=int(clip01(b[1])*height)
        x2=int(clip01(b[2])*width)
        y2=int(clip01(b[3])*height)
        if alt_clr:
            clr=(255,255,0)
        else:
            clr=(0, 255, 0)
        if "test" in a:
            clr=(0,255,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), clr, 2)
        label=class_names[a["class"]] if not class_names==None else f"Class_{i}"
        label+=" "
        label+="{:4.3f}  ".format(a["confidence"])

        fp=a["face_points"]
        for i in range(5):
            if fp[3*i+2]!=0:
                x=fp[3*i+0]*width
                y=fp[3*i+1]*height
                clr=(0,0,255)
                if i==0 or i==2: # RIGHT points
                    clr=(0,255,255)
                cv2.circle(img, (int(x), int(y)), 2, clr, -1) 
            
        if "pose_points" in a:
            kp=a["pose_points"]
            kp_line(img, kp, 0, 1)
            kp_line(img, kp, 0, 2) 
            kp_line(img, kp, 0, 5, 6)

            kp_line(img, kp, 1, 3)
            kp_line(img, kp, 2, 4)

            kp_line(img, kp, 5, 6)
            kp_line(img, kp, 5, 11)
            kp_line(img, kp, 6, 12)
            kp_line(img, kp, 11, 12)

            kp_line(img, kp, 5, 7)
            kp_line(img, kp, 7, 9)

            kp_line(img, kp, 6, 8)
            kp_line(img, kp, 8, 10)

            kp_line(img, kp, 11, 13)
            kp_line(img, kp, 13, 15)

            kp_line(img, kp, 12, 14)
            kp_line(img, kp, 14, 16)

        cv2.putText(img,
                    label, 
                    (x1,y2), 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        
def add_pose_points_to_gts(dets, gts, iou_thr=0.6, min_sz=0.5*0.5, class_names=None, remove_existing=False):
    """
    Given existing GTs and detections from a pose point detector, add the pose values into the GTs
    does this by matching the box to find a good box in the GTs and copying the pose points to that box
    -i.e. assumption is the GTs already have full and complete labelling for the person boxes
    """
    person_class=class_names.index("person")
    matched=[False]*len(dets)
    added=0
    if remove_existing:
        for g in gts:
            g["pose_points"]=[0]*51

    for g in gts:
        if box_a(g["box"])>min_sz: 
            best_iou, best_match, i=best_iou_match(g, dets)
            if best_iou>iou_thr and matched[i]==False:
                matched[i]=True
                if has_pose_points(g)==False:
                    g["pose_points"]=best_match["pose_points"]
                    added+=1
    return added

def filter_faces_by_persons(faces, persons, ioa_thr=0.9, class_names=None):
    assert(class_names!=None)
    if not "person" in class_names:
        return faces # can't do any filtering
    assert("face" in class_names)
    face_class=class_names.index("face")
    person_class=class_names.index("person")

    out=[]
    matched=[False]*len(persons)
    keep_indexes=[]

    for j,f in enumerate(faces):
        if face_class!=None and f["class"]!=face_class:
            keep_indexes.append(j)
            continue
        best_ioa=0
        best_index=-1

        for i,p in enumerate(persons):
            if p["class"]!=person_class:
                continue
            if matched[i]==False:
                ioa=box_i(f["box"], p["box"])/(box_a(f["box"])+1e-10)
                iopa=box_i(f["box"], p["box"])/(box_a(p["box"])+1e-10)
                width_test=box_w(f["box"])>0.10*box_w(p["box"]) # face must be not too small relative to body width
                top_test=f["box"][1]<(0.5*p["box"][1]+0.5*p["box"][3]) # top of face has to be in top half of body
                if ioa>best_ioa and top_test and width_test and iopa>0.005:
                    best_ioa=ioa
                    best_index=i

        if best_ioa>ioa_thr:
            matched[best_index]=True
            keep_indexes.append(j)

    ret=[faces[i] for i in keep_indexes]
    return ret

def sstr(x):
    if x==int(x):
        return str(int(x))+" "
    return "{:4.3f}  ".format(x)

def write_annotations(an, include_face=True, include_pose=False):
    an_txt=""
    for a in an:
        b=a["box"]
        cx=(b[2]+b[0])*0.5
        cy=(b[3]+b[1])*0.5
        w=b[2]-b[0]
        h=b[3]-b[1]
        an_txt+=str(a["class"])+" {0:.4f} {1:.4f} {2:.4f} {3:.4f} ".format(cx,cy,w,h)

        if include_face:
            if "face_points" in a:
                fp=a["face_points"]
            else:
                fp=[0]*15
            for i in range(5*3):
                #if fp[i]<0 or fp[i]>1.0:
                #    print(f"Warning: face point {i} out of range {fp[i]}")
                an_txt+=sstr(clip01(fp[i]))
        if include_pose:
            if "pose_points" in a:
                pp=a["pose_points"]
            else:
                pp=[0]*17*3
            for i in range(17*3):
                if pp[i]<0 or pp[i]>1.0:
                    print(f"Warning: pose point {i} out of range {pp[i]}")
                an_txt+=sstr(clip01(pp[i]))
        an_txt+="\n"
    return an_txt

def unpack_yolo_keypoints(det_kp_list, det_kp_conf_list, index):
    if det_kp_list==None:
        return None, None

    if det_kp_conf_list!=None:
        det_kp_conf=det_kp_conf_list[index]
    else:
        det_kp_conf=[1.0]*len(det_kp)
    det_kp=det_kp_list[index]
    flat_kp=[0]*3*len(det_kp)
    for j in range(len(det_kp)):
        flat_kp[3*j+0]=det_kp[j][0]
        flat_kp[3*j+1]=det_kp[j][1]
        flat_kp[3*j+2]=det_kp_conf[j]
        if det_kp[j][0]<=0 and det_kp[j][1]<=0:
            flat_kp[3*j+2]=0

    if len(flat_kp)==51:
        return None, flat_kp # pose points only
    elif len(flat_kp)==66:
        return flat_kp[0:15], flat_kp[15:66]
    elif len(flat_kp)==15:
        return flat_kp[0:15], None # face points only
    else:
        print("Bad number of yolo keypoints "+str(len(flat_kp)))
    return None, None

def get_dataset_path(ds_yaml):
    return os.path.dirname(ds_yaml)

def get_dataset_name(ds_yaml):
    with open(ds_yaml) as stream:
        try:
            ds=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            ds=None

    if ds!=None:
        if "dataset_name" in ds:
            return ds["dataset_name"]
    return name_from_file(ds_yaml)

def get_loader(name):
    loaders={"CocoLoader":CocoLoader,
             "OpenImagesLoader": OpenImagesLoader,
             "O365Loader": O365Loader,
             "WiderfaceLoader":WiderfaceLoader,
             "RoboflowLoader":RoboflowLoader,
             "WeaponDetectionLoader":WeaponDetectionLoader}
    if not name in loaders:
        print("Could not find loader "+name)
        exit()
    return loaders[name]

def load_dictionary(name):
    if name.endswith(".json"):
        with open(name) as json_file:
            dict = json.load(json_file)
        return dict
    if name.endswith(".yml") or name.endswith(".yaml"):
        with open(name) as stream:
            try:
                dict=yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                exit()
        return dict
    return None

def image_add_exif_comment(image_file, comment):
    image=exif.Image(image_file)
    image["user_comment"]=comment
    with open(image_file, 'wb') as file:
        new_image_file.write(image.get_file())

def image_append_exif_comment(image_file, comment):
    try:
        image=exif.Image(image_file)
        old_comment=None
        if image.has_exif:
            old_comment=image.get("user_comment")
        if old_comment!=None:
            image.delete("user_comment")
            comment=old_comment+";"+comment
        image["user_comment"]=comment
        with open(image_file, 'wb') as file:
            file.write(image.get_file())
    except Exception as e:
        pass #print("image_append_exif_comment exception")

def image_get_exif_comment(image_file):
    image=exif.Image(image_file)
    comment=image.get("user_comment")
    return comment