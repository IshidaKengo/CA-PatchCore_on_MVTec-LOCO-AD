import numpy as np
import pickle
import csv
from sklearn.metrics import roc_auc_score
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from load_dataset import LoadDataset

def evaluation(args, category):
    ###Preparation####
    save_path = os.path.join(args.evaluation_path, f'num_clusters={args.num_cluster}', category)
    os.makedirs(save_path, exist_ok=True)
    
    load_path = os.path.join(args.output_path, f'num_clusters={args.num_cluster}', category, 'results')

    ####Load Results###
    with open(os.path.join(load_path,'pred_list.txt'), 'rb') as f:
        scores_all = pickle.load(f)

    with open(os.path.join(load_path, 'anomaly_maps.txt'), 'rb') as f:
        anomaly_maps = pickle.load(f)

    test_datasets = LoadDataset(os.path.join(args.dataset_path, category),  transform = [], phase='test')

    scores_good = []
    scores_logical = []
    scores_structural = []

    for pred, type in zip(scores_all,  test_datasets.types):
        if type == 'good':
            scores_good.append(pred)
        elif type == 'logical_anomalies':
            scores_logical.append(pred)
        else:
            scores_structural.append(pred)
    
    ###Caliculate AUROC###
    auc_str = caliculate_AUROC(scores_good, scores_structural)
    auc_log = caliculate_AUROC(scores_good, scores_logical)
    auc_mean = 0.5 * (auc_str + auc_log) 
    
    print(f'AUC  str:{auc_str:.3f}   log:{auc_log:.3f}   mean:{auc_log:.3f}')
    
    with open(os.path.join(os.path.join(args.evaluation_path, f'num_clusters={args.num_cluster}'), r'AUROC.csv'), 'a') as f:
        writer = csv.writer(f)
        writer.writerow([category, auc_str, auc_log, auc_mean])    

    ###########
    
    ###Save histgrams###
    ##Save the histgrams of image-level anomaly scores
    bins = 32
    plt.clf()
    plt.hist([scores_good, scores_logical, scores_structural], bins, label=['normal', 'logical anomaly', 'structural anomaly'], color=['green', 'blue', 'red'])
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.title(category)
    plt.savefig(os.path.join(save_path, 'image-level_score_histogram.png'))
    
    ##Save the histgrams of pixel-level anomaly scores
    good_index = [index for index, defect_type in enumerate(test_datasets.types) if defect_type == 'good']
    abnormal_index = [index for index, defect_type in enumerate(test_datasets.types) if defect_type != 'good']

    good_maps = [anomaly_maps[i] for i in good_index]
    abnormal_maps =  [anomaly_maps[i] for i in abnormal_index]
    
    good_maps = np.array(good_maps)
    abnormal_maps = np.array(abnormal_maps)

    good_maps_flatten = np.ravel(good_maps)
    good_maps_flatten.tolist()
    abnormal_maps_flatten = np.ravel(abnormal_maps)
    abnormal_maps_flatten.tolist()
    
    bins = 1024
    plt.clf()
    plt.hist([good_maps_flatten, abnormal_maps_flatten], bins, label=['normal', 'abnormal'], color=['green', 'red'])
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.grid(axis='y')
    plt.legend(loc='upper right')
    plt.title(category)
    plt.savefig(os.path.join(save_path, 'pixel-level_score_histogram.png'))
    ##############
    
    ###Caliculate precision, recall and F1 value###
    th_list = np.linspace(min(scores_all), max(scores_all), 100)
    th_list = th_list.tolist()
    gt_list = test_datasets.labels
    
    precision_list = []
    recall_list = []
    f1_list = []
        
    for th in th_list:
        pred_list = []
        for score in scores_all:
            pred_list.append(1 if score >= th else 0)

        precision_list.append(precision_score(gt_list, pred_list))
        recall_list.append(recall_score(gt_list, pred_list))
        f1_list.append(f1_score(gt_list, pred_list))
    
    plt.figure()
    plt.plot(th_list, precision_list, color='cyan', label='precision')
    plt.plot(th_list, recall_list, color='lime', label='recall')
    plt.plot(th_list, f1_list, color='red', label='f1 value')
    plt.xlabel('threshold')
    plt.legend()
    plt.title(category)
    plt.savefig(os.path.join(save_path, 'P_R_f1.png'))
    ##################################
    
    ###Caliculate a threshold to save anomaly map###
    th_list = np.linspace(min(scores_all), max(scores_all), 100)
    th_list = th_list.tolist()
    gt_list = test_datasets.labels
    f1_list = []
        
    for th in th_list:
        pred_list = []
        for score in scores_all:
            pred_list.append(1 if score >= th else 0)

        f1_list.append(f1_score(gt_list, pred_list))

    th_f1valuemax = th_list[np.argmax(f1_list)]         #Threshold for maximun F1 value
    ##############################################
    
    #####Save Anomaly Maps########
    temp_map = np.array(anomaly_maps)
    max_num, min_num = temp_map.max(), temp_map.min()
    saturate_hi_num, saturate_lw_num  = 0.9*max_num, th_f1valuemax #Upper and Lower limit of anomaly maps
    
    save_anomap = SaveAnomaryMap(test_datasets.img_paths, anomaly_maps, test_datasets.types, os.path.join(save_path, 'anomaly_maps'))
    #save_anomap.save_anomap_1()
    save_anomap.save_anomap_2(saturate_hi_num, saturate_lw_num)
    ####################


def caliculate_AUROC(score_of_good, score_of_anomaly):
    gt = np.zeros(len(score_of_good))
    gt = np.concatenate([gt, np.ones(len(score_of_anomaly))],axis=0)
    
    pred = np.concatenate([score_of_good, score_of_anomaly], axis=0)
    
    auroc = roc_auc_score(gt, pred)
    return auroc

class SaveAnomaryMap:
    def __init__(self, img_paths, anomaps, defect_types, save_dir):
        self.img_paths, self.anomaps, self.defect_types, self.save_dir = img_paths, anomaps,  defect_types, save_dir
        os.makedirs(save_dir, exist_ok = True)
    
    ##Min-Max Normalized Heatmap per Image
    def save_anomap_1(self):
        for i in range(len(self.img_paths)):
            input_img = cv2.imread(self.img_paths[i])
            
            anomap = cv2.resize(self.anomaps[i], (input_img.shape[1], input_img.shape[0]))
            #anomaly_map_resized_blur = gaussian_filter(anomap, sigma=4)
            
            anomaly_map_norm = self.min_max_norm(anomap)
            anomaly_map_norm_hm = self.cvt2heatmap(anomaly_map_norm*255)

            # anomaly map on image
            hm_on_img = self.heatmap_on_image(anomaly_map_norm_hm , input_img)
            
            # save images
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}.jpg'), input_img)
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}_amap.jpg'), anomaly_map_norm_hm)
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}_amap_on_img.jpg'), hm_on_img)
            #cv2.imwrite(os.path.join(self.result_path, f'{x_type}_{file_name}_gt.jpg'), gt_img)
    
    ##Setting the maximun and minimum values of normalization for each image optionally
    def save_anomap_2(self, max, min):
        for i in range(len(self.img_paths)):
            input_img = cv2.imread(self.img_paths[i])
            
            anomap = cv2.resize(self.anomaps[i], (input_img.shape[1], input_img.shape[0]))
            #anomaly_map_resized_blur = gaussian_filter(anomap, sigma=4)
            
            anomaly_map_norm = self.select_min_max_norm(anomap, max, min)
            anomaly_map_norm_hm = self.cvt2heatmap(anomaly_map_norm*255)

            # anomaly map on image
            hm_on_img = self.heatmap_on_image(anomaly_map_norm_hm , input_img)
            
            # save images
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}.jpg'), input_img)
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}_amap.jpg'), anomaly_map_norm_hm)
            cv2.imwrite(os.path.join(self.save_dir, f'{self.defect_types[i]}_{str(i)}_amap_on_img.jpg'), hm_on_img)

    def min_max_norm(self, image):
        a_min, a_max = image.min(), image.max()
        return (image-a_min)/(a_max - a_min)    

    def select_min_max_norm(self, image, saturate_hi_num, saturate_lw_num):
        norm_img = np.where(image > saturate_hi_num, 1, image)
        norm_img = np.where(image < saturate_lw_num, 0, norm_img)
        norm_img = np.where((image >= saturate_lw_num) & (image <= saturate_hi_num), (norm_img-saturate_lw_num)/(saturate_hi_num-saturate_lw_num), norm_img)
        #a_min, a_max = image.min(), image.max()
        return norm_img

    def cvt2heatmap(self, gray):
        heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
        return heatmap

    def heatmap_on_image(self, heatmap, image):
        if heatmap.shape != image.shape:
            heatmap = cv2.resize(heatmap, (image.shape[0], image.shape[1]))
        out = np.float32(heatmap)/255 + np.float32(image)/255
        out = out / np.max(out)
        return np.uint8(255 * out)
