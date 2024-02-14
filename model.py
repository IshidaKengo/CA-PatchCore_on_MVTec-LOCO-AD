import torch
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from PIL import Image
from torch import nn
import pytorch_lightning as pl
import pickle
from sampling_methods.kcenter_greedy import kCenterGreedy
from sklearn.random_projection import SparseRandomProjection

from load_dataset import LoadDatasetWithSeg
from utils_patchcore import *   

def cross_attention_per_channel(q, k ,v):
    attention_in1 = q #B, C, H*W
    attention_in2 = k
    attention_v = v 

    attention_in1 = torch.reshape(attention_in1, (attention_in1.shape[0], attention_in1.shape[1], attention_in1.shape[2], 1)) ## B, C, H*W, 1
    attention_in2 = torch.reshape(attention_in2, (attention_in2.shape[0], attention_in2.shape[1], 1, attention_in2.shape[2]))  ## B, C, 1, H*W
    attention_v = torch.reshape(attention_v, (attention_v.shape[0], attention_v.shape[1], attention_v.shape[2], 1)) ## B, C, H*W, 1
    attention1 = torch.matmul(attention_in1, attention_in2) 
    attention2 = torch.nn.functional.softmax(attention1, dim=-1) 
    attention_out = torch.matmul(attention2, attention_v) 
    attention_out = attention_out.view(attention_out.shape[0], attention_out.shape[1], attention_out.shape[2]) ##B, C, H*W
    return attention_out
        
class MODEL(pl.LightningModule):
    def __init__(self, args, category, save_dir):
        super(MODEL, self).__init__()

        #self.save_hyperparameters(hparams)

        self.args = args
        self.category = category
        self.save_dir = save_dir
        self.segimg_path = os.path.join('./segimgs', args.dataset_name, f'num_clusters={args.num_cluster}', category)

        self.init_features()
        def hook_t(module, input, output):
            self.features.append(output)

        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'wide_resnet50_2', pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.layer2[-1].register_forward_hook(hook_t)
        self.model.layer3[-1].register_forward_hook(hook_t)
        #print(self.model)

        self.criterion = torch.nn.MSELoss(reduction='sum')

        self.init_results_list()
        
        mean_train = [0.485, 0.456, 0.406]
        std_train = [0.229, 0.224, 0.225]
        
        self.data_transforms = transforms.Compose([
                        transforms.Resize((self.args.input_size, self.args.input_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        #transforms.CenterCrop(self.args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
        self.gt_transforms = transforms.Compose([
                        transforms.Resize((self.args.input_size, self.args.input_size)),
                        transforms.ToTensor(),
                        #transforms.CenterCrop(self.args.input_size)])
                        ])

        self.inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    def init_results_list(self):
        self.gt_list_px_lvl = []
        self.pred_list_px_lvl = []
        self.gt_list_img_lvl = []
        self.pred_list_img_lvl = []
        self.img_path_list = []      
        self.anomaly_map_all = []    
        self.input_x_list =[]
        self.defect_types = []

    def init_features(self):
        self.features = []

    def forward(self, x_t):
        self.init_features()
        _ = self.model(x_t)
        return self.features

    def train_dataloader(self):
        image_datasets = LoadDatasetWithSeg(root=os.path.join(self.args.dataset_path,self.category), segimg_path=self.segimg_path, transform=self.data_transforms, phase='train')
        train_loader = DataLoader(image_datasets, batch_size=1, shuffle=False, num_workers=4) #, pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_datasets = LoadDatasetWithSeg(root=os.path.join(self.args.dataset_path,self.category), segimg_path=self.segimg_path, transform=self.data_transforms, phase='test')
        test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=4) #, pin_memory=True) # only work on batch_size=1, now.
        return test_loader

    def configure_optimizers(self):
        return None

    def on_train_start(self):      
        self.model.eval() # to stop running_var move (maybe not critical)
        self.embedding_dir_path, self.result_path= prep_dirs(os.path.join(self.save_dir, self.category))
        self.embedding_list = []
    
    def on_test_start(self):
        self.init_results_list()
        self.embedding_dir_path, self.result_path= prep_dirs(os.path.join(self.save_dir, self.category))
        
    def training_step(self, batch, batch_idx): # save locally aware patch features
        x, _, file_name, _, segimg_path = batch
        features = self(x)
        embeddings = []
               
        for feature in features:
            #print(feature.shape)
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)
            
            ##original patchcore features
            conv_out = avep(feature)
            try:
                conv_embedding = embedding_concat(conv_embedding, conv_out)
            except:
                conv_embedding = conv_out
        
            ##cross-attention with segmentaition mask
            segimgs = []
            for i in range(self.args.num_cluster):
                path = os.path.join(segimg_path[0], f'heatresult{i}.jpg')
                segimg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
                segimg = cv2.resize(segimg, (feature.shape[2], feature.shape[3]))
                _, segimg = cv2.threshold(segimg, 0, 255, cv2.THRESH_OTSU)
                segimg = torch.from_numpy(segimg).unsqueeze(0).float() /255.0
                segimgs.append(segimg)
            
            feature = maxp(feature)
            B, C, H, W= feature.shape
            flat_feature = feature.view(B, C, H*W) 
            
            for seg in segimgs:
                seg = seg.view(1,1,seg.shape[1]*seg.shape[2])
                sa_out = cross_attention_per_channel(flat_feature, seg, flat_feature)
                sa_out = sa_out.view(B, C, H, W)
                try:
                    sa_embedding = embedding_concat(sa_embedding, sa_out)
                except:
                    sa_embedding = sa_out
        
        embedding = embedding_concat(conv_embedding, sa_embedding)
        self.embedding_list.extend(reshape_embedding(np.array(embedding)))
        
        #self.embedding_list.extend(reshape_embedding(np.array(embeddings[0])))

    def training_epoch_end(self, outputs): 
        total_embeddings = np.array(self.embedding_list)
        print(total_embeddings.shape)
        # Random projection
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9) # 'auto' => Johnson-Lindenstrauss lemma  
        self.randomprojector.fit(total_embeddings)
        # Coreset Subsampling
        print(total_embeddings.shape)
        selector = kCenterGreedy(total_embeddings,0,0)
        selected_idx = selector.select_batch(model=self.randomprojector, already_selected=[], N=int(total_embeddings.shape[0]*self.args.coreset_sampling_rate))
        self.embedding_coreset = total_embeddings[selected_idx]
        
        #print('initial embedding size : ', total_embeddings.shape)
        #print('final embedding size : ', self.embedding_coreset.shape)
        with open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'wb') as f:
            pickle.dump(self.embedding_coreset, f)

    def test_step(self, batch, batch_idx): # Nearest Neighbour Search
        
        self.embedding_coreset = pickle.load(open(os.path.join(self.embedding_dir_path, 'embedding.pickle'), 'rb'))
        x, label, file_name, x_type, segimg_path = batch
        # extract embedding
        features = self(x)
        embeddings = []
        for feature in features:
            avep = torch.nn.AvgPool2d(3, 1, 1)
            maxp = torch.nn.MaxPool2d(3, 1, 1)
            
            ##original patchcore features
            conv_out = avep(feature)
            try:
                conv_embedding = embedding_concat(conv_embedding, conv_out)
            except:
                conv_embedding = conv_out
        
            ##cross-attention with segmentaition mask
            segimgs = []
            for i in range(self.args.num_cluster):
                path = os.path.join(segimg_path[0], f'heatresult{i}.jpg')
                segimg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  
                segimg = cv2.resize(segimg, (feature.shape[2], feature.shape[3]))
                _, segimg = cv2.threshold(segimg, 0, 255, cv2.THRESH_OTSU)
                segimg = torch.from_numpy(segimg).unsqueeze(0).float() /255.0
                segimgs.append(segimg)
            
            feature = maxp(feature)
            B, C, H, W= feature.shape
            flat_feature = feature.view(B, C, H*W) 
            
            for seg in segimgs:
                seg = seg.view(1,1,seg.shape[1]*seg.shape[2])
                sa_out = cross_attention_per_channel(flat_feature, seg, flat_feature)
                sa_out = sa_out.view(B, C, H, W)
                try:
                    sa_embedding = embedding_concat(sa_embedding, sa_out)
                except:
                    sa_embedding = sa_out
        
        embedding = embedding_concat(conv_embedding, sa_embedding)
        H = embedding.shape[3]
        embedding_test = np.array(reshape_embedding(np.array(embedding)))  

        knn = KNN(torch.from_numpy(self.embedding_coreset).cuda(), k=9)
        score_patches = knn(torch.from_numpy(embedding_test).cuda())[0].cpu().detach().numpy()

        anomaly_map = score_patches[:,0].reshape((H, H))  #Pixel-level anomaly scores = Anomaly map
        score = max(score_patches[:,0]) # Image-level anomaly score  
        
        self.anomaly_map_all.append(anomaly_map)
        self.pred_list_img_lvl.append(score)  

    def test_epoch_end(self, outputs):
        #Output image-level anomaly scores
        with open(os.path.join(self.result_path,'pred_list.txt'), 'wb') as f:
            pickle.dump(self.pred_list_img_lvl, f)
        
        #Output anomaly maps
        with open(os.path.join(self.result_path, 'anomaly_maps.txt'), 'wb') as f:
            pickle.dump(self.anomaly_map_all, f)
