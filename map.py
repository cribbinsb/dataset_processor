import sys
import ultralytics
from ultralytics import YOLO
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import copy
from datetime import datetime
import src.dataset_util as dsu
import src.ultralytics_ap as uap
import src.dataset_processor as dp
import json
from datetime import datetime, timedelta
import time

def fstr(x):
    if x==None:
        return "        "
    x=float(x)
    if x<0.0001:
        return "        "
    return "{:5.4f}  ".format(x)

def compute_one_ap(dataset, model_path, yolo_model=None, batchsize=16, max_det=600, iou_thr=0.5, nms_iou=0.6, det_conf=0.001, classes=None, augment=False):
    
    x=dp.DatasetProcessor(dataset, task="val", class_names=classes)
    nc=len(classes)
    if not yolo_model==None:
        x.set_yolo_detector(yolo_model)
    else:
        x.set_yolo_detector(model_path)

    target_class=[]
    conf=[]
    tp=[]
    pred_class=[]

    target_class_large=[]
    conf_large=[]
    tp_large=[]
    pred_class_large=[]

    target_class_kp=[]
    conf_kp=[]
    tp_kp=[]
    pred_class_kp=[]

    target_class_kp_large=[]
    conf_kp_large=[]
    tp_kp_large=[]
    pred_class_kp_large=[]

    measure_kp=True

    desc=x.dataset_name+" / "+dsu.name_from_file(model_path)
    if augment:
        desc+=" / AUG"
    else:
        desc+=" / NOAUG"

    for i in tqdm(range(x.num_files), desc=desc):
        gts=x.get_gt(index=i)
        dets=x.get_detections(index=i,det_thr=det_conf)   
        if gts==None or dets==None:
            continue
     
        det_matched, gt_matched=dsu.match_boxes(dets, gts, iou_thr)

        for j in range(len(gts)):
            target_class.append(gts[j]['class'])
            if dsu.is_large(gts[j]):
                target_class_large.append(gts[j]['class'])


        for j in range(len(dets)):
            pred_class.append(dets[j]['class'])
            conf.append(dets[j]['confidence'])
            tp.append(0 if det_matched[j]==-1 else 1)
            if dsu.is_large(dets[j]):
                pred_class_large.append(dets[j]['class'])
                conf_large.append(dets[j]['confidence'])
                tp_large.append(0 if det_matched[j]==-1 else 1)

        if measure_kp:
            face_class=x.get_class_index("face")
            person_class=x.get_class_index("person")
            det_matched, gt_matched=dsu.match_boxes_kp(dets, gts, 
                                                   iou_thr=0.5, 
                                                   face_class=face_class, 
                                                   person_class=person_class)
            for j in range(len(gts)):
                target_class_kp.append(gts[j]['class'])
                if dsu.is_large(gts[j]):
                    target_class_kp_large.append(gts[j]['class'])

            for j in range(len(dets)):
                pred_class_kp.append(dets[j]['class'])
                conf_kp.append(dets[j]['confidence'])
                tp_kp.append(0 if det_matched[j]==-1 else 1)
                if dsu.is_large(dets[j]):
                    pred_class_kp_large.append(dets[j]['class'])
                    conf_kp_large.append(dets[j]['confidence'])
                    tp_kp_large.append(0 if det_matched[j]==-1 else 1)
    
    augstr="+aug" if augment else ""
    result={"model":dsu.name_from_file(model_path), 
            "dataset": x.dataset_name,
            "datasetaug": x.dataset_name+augstr,
            "augment": augment,
            "time": datetime.now()}

    ap, p, r = uap.ap_calc(conf, tp, pred_class, target_class, nc)
    ap_large, p_large, r_large = uap.ap_calc(conf_large, tp_large, pred_class_large, target_class_large, nc)

    if measure_kp:
        ap_kp, p_kp, r_kp = uap.ap_calc(conf_kp, tp_kp, pred_class_kp, target_class_kp, nc)
        ap_kp_large, p_kp_large, r_kp_large = uap.ap_calc(conf_kp_large, tp_kp_large, pred_class_kp_large, target_class_kp_large, nc)
    
    for i in range(len(classes)):
        c=classes[i]
        result[c+"_ap50"]=float(ap[i])
        result[c+"-p"]=float(p[i])
        result[c+"-r"]=float(r[i])
        result[c+"_ap50_large"]=float(ap_large[i])
        result[c+"-p_large"]=float(p_large[i])
        result[c+"-r_large"]=float(r_large[i])
        if measure_kp:
            result[c+"_ap50_kp"]=float(ap_kp[i])
            result[c+"-p_kp"]=float(p_kp[i])
            result[c+"-r_kp"]=float(r_kp[i])
            result[c+"_ap50_kp_large"]=float(ap_kp_large[i])
            result[c+"-p_kp_large"]=float(p_kp_large[i])
            result[c+"-r_kp_large"]=float(r_kp_large[i])

    return result

def compute_multiple_aps(datasets, model_path, classes=["person", "face", "vehicle", "animal", "weapon"], augment=False, batchsize=16):
    yolo_model = YOLO(model_path)
    results=[]
    for d in datasets:
        results.append(compute_one_ap(d, model_path, yolo_model=yolo_model, classes=classes, augment=augment, batchsize=batchsize))
    return results

def compare(scorea, scoreb):
    #return scorea['person_ap50']-scoreb['person_ap50']
    sa=scorea['person_ap50']
    sb=scoreb['person_ap50']
    if sa>sb:
        return 1
    elif sb>sa:
        return -1
    else:
        return 0

def get_avg_scores(results, model, param):
    t=1.0
    n=0
    for r in results:
        if r["model"]==model:
            if param in r:
                if isinstance(r[param], int) or isinstance(r[param], float):
                    if r[param]>0.001:
                        t*=r[param]
                        n=n+1
    if n>0:
        t=pow(t, 1.0/n)
        return t
    else:
        return 0

def add_averages(results):
    datasets = sorted(list(set(entry['dataset'] for entry in results)), reverse=True)

    models = list(set(entry['model'] for entry in results))
    paramset=set([])
    for r in results:
        paramset=paramset.union(set(r.keys()))

    params = list(paramset)
    params.remove("model")
    params.remove("dataset")
    params.remove("datasetaug")
    params.remove("augment")
    params.remove("time")

    results2=[]
    for m in models:
        e={"model":m, "dataset":"_geomean", "datasetaug":"_geomean", "augment":False, "time":datetime.now()}
        for p in params:
            e[p]=get_avg_scores(results, m, p)
        results2.append(e)
    results+=results2
    
def results_chart(results, param_to_show, result_location):
    datasets = sorted(list(set(entry['datasetaug'] for entry in results)), reverse=True)
    models = [entry['model'] for entry in results if entry["dataset"]=="_geomean"]
    models_scores = [entry[param_to_show] for entry in results if entry["dataset"]=="_geomean"]
    models = [x for _,x in sorted(zip(models_scores, models))]
    # Organize data for plotting
    ap_data = {model: [] for model in models}

    min_score=100.0
    max_score=-100
    for dataset in datasets:
        for model in models:
            # Get AP score for each model-dataset pair
            ap_score = next((entry[param_to_show] for entry in results if entry['datasetaug'] == dataset and entry['model'] == model), None)
            if ap_score==None:
                continue
            ap_data[model].append(ap_score)
            if ap_score>0.1:
                min_score=min(min_score, ap_score)
            max_score=max(max_score, ap_score)

    # Plotting
    bar_width = 0.1
    gap=0.3

    index = np.arange(len(datasets))*(len(models)*bar_width+gap)

    plt.figure(figsize=(12, 6))

    # Create bars for each model
    for i, model in enumerate(models):
        plt.bar(index + i * bar_width, ap_data[model], bar_width, label=model)

    # Add labels, title, and adjust y-axis scale
    plt.xlabel('Datasets')
    plt.ylabel('Score')
    plt.title('Object Detector '+param_to_show+' Scores by Model '+datetime.today().strftime('%Y-%m-%d'))
    plt.xticks(index + (len(models)*bar_width)/2, datasets, rotation=45)
    plt.ylim(min_score*0.975, max_score*1.025)  # Adjust y-axis to better fit the range of AP scores

    # Move the legend to the upper right outside of the plot
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make space for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Display the plot
    #plt.show()
    directory=os.path.join(result_location, datetime.today().strftime('%Y-%m-%d'))
    dsu.makedir(directory)

    plt.savefig(os.path.join(directory, param_to_show+".pdf"), format="pdf", bbox_inches="tight")

def cfloat(x):
    try:
        ret=float(x)
    except ValueError as e:
        ret=0.0
    return ret

def show_results_plt(results, columns, column_text, sort_key, result_location):
    dss=[]
    for r in results:
        if not r["dataset"] in dss:
            dss.append(r["dataset"])
    
    directory=os.path.join(result_location, datetime.today().strftime('%Y-%m-%d'))
    dsu.makedir(directory)
    
    with PdfPages(os.path.join(directory, "results_table.pdf")) as pdf:
        for ds in dss:
            r_ds=[r for r in results if r["dataset"]==ds]
            r_ds=sorted(r_ds, key=lambda x: x[sort_key], reverse=True)
            
            data_row0=["Model"]
            for c in column_text:
                data_row0.append(c)
            data=[data_row0]
            for r in r_ds:
                data_row=[r["model"]]
                for c in columns:
                    data_row.append(fstr(r[c] if c in r else None))
                data.append(data_row)

            fig, ax = plt.subplots(figsize=(16, 0.3*len(r_ds)+4))
            ax.axis('tight')
            ax.axis('off')
            table = ax.table(cellText=data, loc='center', cellLoc='center')
            table.scale(1, 2)  # Adjust the size of the table cells
            table.auto_set_font_size(False)  # Disable automatic font scaling
            table.set_fontsize(12)
            #table[(row_idx, col_idx)].set_text_props(weight='bold')
            #table[(row_idx, col_idx)].set_facecolor('#FFDDC1')  # Set a light color for the row

            for col in range(1, len(data[0])):
                coldata=[cfloat(data[i][col]) for i in range(1, len(data))]
                maxcolval=max(coldata)
                if maxcolval<=0:
                    mincolval=maxcolval
                else:
                    mincolval=min([x for x in coldata if x>0])
                for row in range(1, len(data)):
                    if cfloat(data[row][col])==0.0:
                        pass
                    elif cfloat(data[row][col])>=maxcolval-0.001:
                        table[(row, col)].set_facecolor('#44FF44')
                    elif cfloat(data[row][col])>=maxcolval-0.01:
                        table[(row, col)].set_facecolor('#aaFFaa')
                    elif cfloat(data[row][col])>=maxcolval-0.02:
                        table[(row, col)].set_facecolor('#ddFFdd')
                    elif cfloat(data[row][col])<=mincolval+0.001:
                        table[(row, col)].set_facecolor('#FF4444')
                    elif cfloat(data[row][col])<=mincolval+0.01:
                        table[(row, col)].set_facecolor('#FFaaaa')
                    elif cfloat(data[row][col])<=mincolval+0.02:
                        table[(row, col)].set_facecolor('#FFdddd')

            for row in range(1, len(data)):
                if "face" in data[row][0] or "full" in data[row][0]:
                   for col in range(0, len(data[0])):
                       table[(row, col)].set_text_props(weight='bold')
            
            plt.text(0.5, 0.83, ds, ha='center', va='bottom', fontsize=16, weight='bold', transform=ax.transAxes)

            for key, cell in table.get_celld().items():
                # key[1] is the column index
                if key[1] == 0:  # First column
                    cell.set_width(0.25)  # Wider first column (adjust value as needed)
                else:  # Other columns
                    cell.set_width(0.08)  # Narrower other columns (adjust value as needed)
                if key[0] == 0:  # First row
                    cell.set_height(0.06)  # Wider first column (adjust value as needed)
                else:  # Other columns
                    cell.set_height(0.030)  # Narrower other columns (adjust value as needed)

            #table[(row_idx, col_idx)].set_text_props(weight='bold') 
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

def show_results(results, columns, column_text, sort_key):
    ds={}
    n=0
    for r in results:
        if not r["dataset"] in ds:
            ds[r["dataset"]]=n*100
            n+=1
    out_txt=[]
    out_sort=[]
    for r in results:
        label=r["dataset"]
        if r["augment"]==True:
            label+="+aug"
        out="{:20s} {:26s} ".format(label[0:19],r["model"])
        for c in columns:
            if not c in r:
                out+="        "
            else:
                out+=fstr(r[c])
        
        sortscr=ds[r["dataset"]]*1000+r[sort_key]
        if r["augment"]==True:
            sortscr+=100
        out_sort.append(sortscr)
        out_txt.append(out)

    l="\n{:20s} {:26s} ".format("Dataset","Model")
    for c in column_text:
        l+="{:8s}".format(c.split("\n")[0])
    l+="\n{:20s} {:26s} ".format("","")
    for c in column_text:
        l+="{:8s}".format(c.split("\n")[1])

    print(l)
    Z = [x for _,x in sorted(zip(out_sort,out_txt), reverse = True)]
    for z in Z:
        print(z)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='data/map_ultralytics.json', help='config file to use (json or yml)')
    parser.add_argument('--batch-size', type=int, default=16)
    opt = parser.parse_args()

    config = dsu.load_dictionary(opt.config)

    result_location=config["results_location"]
    resultfile=config["results_cache_file"]
    classes=config["classes"]
    datasets=config["datasets"]
    models=config["models"]
    regen_models=config["regen_models"]
    regen_datasets=config["regen_datasets"]
    max_age_days=config["max_age_days"]
    columns=config["columns"]
    sort_key=config["sort_key"]
    chart_list=config["chart_list"]

    cached_results=[]
    if os.path.isfile(resultfile):
        with open(resultfile, 'rb') as handle:
            cached_results = pickle.load(handle)

    augs=[False]

    column_txt=[]
    for c in columns:
        txt="??"
        for cl in classes:
            if c==cl+'_ap50':
                txt=cl+"\n"+"AP50"
            elif c==cl+'-p':
                txt=cl+"\n"+"Prec"
            elif c==cl+'-r':
                txt=cl+"\n"+"Rec"
            elif c==cl+'_ap50_kp':
                if cl=='person':
                    txt="Pose KP\nAP50"
                elif cl=='face':
                    txt="Face KP\nAP50"
                else:
                    txt=cl+"\n"+"KP"
            elif c==cl+'_ap50_large':
                txt=cl+"\nAP50 L"
            elif c==cl+'_ap50_kp_large':
                if c=='person':
                    txt="Pose KP\nAP L"
                elif c=='face':
                    txt="Face KP\nAP L"
                else:
                    txt=cl+"\n"+"KP L"
        column_txt.append(txt[0].upper()+txt[1:])
    
    for r in cached_results:
        if not "augment" in r:
            r["augment"]=False
        if not "datasetaug" in r:
            if r["augment"]==True:
                r["datasetaug"]=r["dataset"]+"+aug"
            else:
                r["datasetaug"]=r["dataset"]
        if not "inference" in r:
                r["inference"]="pytorch"
        if not "precision" in r:
                r["precision"]="fp16"
        if not "time" in r:
            r["time"]=datetime.now()
        if not "classes" in r:
            r["classes"]=classes

    results=[]
    #for r in cached_results:
    #    print(f"{r['dataset']} {r['model']} {r['augment']} {r['datasetaug']}")

    min_time=datetime.now() - timedelta(days=max_age_days)

    #for model in models:
    for dataset in datasets:
        #for dataset in datasets:
        for model in models:
            for augment in augs:
                dataset_name=dsu.get_dataset_name(dataset)
                model_name=dsu.name_from_file(model)

                cached_result_to_use=None
                for r in cached_results:
                    if r["model"]==model_name and r["dataset"]==dataset_name and r["augment"]==augment and set(classes)<=set(r["classes"]):
                        if cached_result_to_use==None or r["time"]>cached_result_to_use["time"]:
                            if r["time"]>min_time:
                                cached_result_to_use=r

                if model_name in regen_models or dataset_name in regen_datasets:
                    cached_result_to_use=None
                
                #if cached_result_to_use!=None:
                #    if "person_ap50" in cached_result_to_use:
                #        if cached_result_to_use["person_ap50"]<0.01:
                #            cached_result_to_use=None
                    
                if cached_result_to_use!=None:
                    results.append(cached_result_to_use)
                else:
                    #show_results_plt(results)
                    result=compute_one_ap(dataset, model, classes=classes, augment=augment)
                    results.append(result)
                    cached_results.append(result)
                    show_results(results, columns, column_txt, sort_key)
                    
                    print("Writing "+str(len(cached_results))+" cached results")
                    if os.path.isfile(resultfile+".tmp"):
                        dsu.rm(resultfile+".tmp")
                    with open(resultfile+".tmp", 'wb') as handle:
                        pickle.dump(cached_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    dsu.rename(resultfile+".tmp", resultfile) # atomic, replaces existing file


    add_averages(results)
    show_results(results, columns, column_txt, sort_key)
    for r in chart_list:
        results_chart(results, r, result_location)
    show_results_plt(results, columns, column_txt, sort_key, result_location)
            

        
        

