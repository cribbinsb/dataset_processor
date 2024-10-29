import dataset_util as du
import os
import yaml
import PIL.Image
import PIL.ExifTags
import xml.etree.ElementTree as ET
import hashlib

class WeaponDetectionLoader:
    def __init__(self,
                 path="/mldata/downloaded_datasets/OD-WeaponDetection",
                 task="val", class_names=["face","person"], ds_params=None):
        
        self.base_path=path
        self.knife_path=path+"/Knife_detection"
        self.pistol_path=path+"/Pistol detection"
        self.class_names=class_names

        knives=os.listdir(self.knife_path+"/Images")
        pistols=os.listdir(self.pistol_path+"/Weapons")

        val_percent=15

        if task=="train":
            knives=[x for x in knives if (hash(x)%100)>=val_percent]
            pistols=[x for x in pistols if (hash(x)%100)>=val_percent]
        else:
            knives=[x for x in knives if (hash(x)%100)<val_percent]
            pistols=[x for x in pistols if (hash(x)%100)<val_percent]

        self.knife_images=["/Knife_detection/Images/"+x for x in knives]
        self.pistol_images=["/Pistol detection/Weapons/"+x for x in pistols]
        self.all_images=self.pistol_images+self.knife_images
        
        self.pistol_out_class=-1
        self.knife_out_class=-1

    def get_info(self):
        return "weapondetection https://github.com/ari-dasci/OD-WeaponDetection https://github.com/ari-dasci/OD-WeaponDetection?tab=CC-BY-4.0-1-ov-file#"

    def add_category_mapping(self, source_class, dest_class):
        if isinstance(source_class, list):
            for c in source_class:
                self.add_category_mapping(c, dest_class)
            return
        if source_class=="knife":
            self.knife_out_class=self.class_names.index(dest_class)
        if source_class=="pistol":
            self.pistol_out_class=self.class_names.index(dest_class)

    def get_annotations(self, img_id):
        img=self.all_images[img_id]
        out_class=-1
        if "Knife_detection" in img:
            out_class=self.knife_out_class
            img=img.replace("/Images/", "/annotations/")
        else:
            out_class=self.pistol_out_class
            img=img.replace("/Weapons/", "/xmls/")
        label=os.path.splitext(img)[0]+".xml"
        tree = ET.parse(self.base_path+label)
        root = tree.getroot()
        img_width=int((root.find('size').find('width').text))
        img_height=int((root.find('size').find('height').text))
        gts=[]
        for object in root.findall('object'):
            x0=(float)(object.find('bndbox').find('xmin').text) / img_width
            x1=(float)(object.find('bndbox').find('xmax').text) / img_width
            y0=(float)(object.find('bndbox').find('ymin').text) / img_height
            y1=(float)(object.find('bndbox').find('ymax').text) / img_height
            gt={"box":[x0,y0,x1,y1], "class":out_class, "confidence":1.0, "face_points":[0.0]*15}
            if out_class!=-1:
                gts.append(gt)

        return gts

    def get_image_ids(self):
        return range(len(self.all_images))

    def get_img_path(self, img_id):
        return self.base_path+"/"+self.all_images[img_id]
    
    def get_category_maps(self):
        return { 'person': [], 
                 'vehicle':[],
                 'animal':[],
                 'face': [],
                 'weapon': ["pistol", "knife"]
                 }