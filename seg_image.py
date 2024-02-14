from modules import DinoFeaturizer
from dataset import MVTecLocoDataset
from torch.utils.data import DataLoader
import torch
from sampler import GreedyCoresetSampler
import torch.nn.functional as F
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from crf import dense_crf
from torchvision import transforms

class Segmentation():
    def __init__(self, args, category):
        self.num_cluster = args.num_cluster
        self.dataset_name = args.dataset_name
        self.dataset_root = f'{args.dataset_path}/{category}/'
        self.image_size = args.input_size
        self.category = category
    
    def run(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        i_m = np.array(IMAGENET_MEAN)
        i_m = i_m[:, None, None]
        i_std = np.array(IMAGENET_STD)
        i_std = i_std[:, None, None]
        
        unloader = transforms.ToPILImage()
        StartTrain = True
        color_list = [[127, 123, 229], [195, 240, 251], [146, 223, 255], [243, 241, 230], [224, 190, 144], [178, 116, 75]]
        color_tensor = torch.tensor(color_list)
        color_tensor = color_tensor[:, :, None, None]
        color_tensor = color_tensor.repeat(1, 1, self.image_size , self.image_size )
        num_cluster = self.num_cluster

        dataset_train = MVTecLocoDataset(root_dir=self.dataset_root, category='train/good', resize_shape=self.image_size )
        dataloader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=4)

        net = DinoFeaturizer()
        net = net.cuda()
        train_feature_list=[]
        greedsampler_perimg = GreedyCoresetSampler(percentage=0.01,device='cuda')
        
        mid_output_path = './seg_mid_output'
        os.makedirs(mid_output_path, exist_ok=True)
        
        if StartTrain:
            for i, Img in enumerate(dataloader):
                with torch.no_grad():
                    image = Img['image']
                    image = image.cuda()
                    feats0, f_lowdim = net(image)
                    feats = feats0.squeeze()
                    feats = feats.reshape(feats0.shape[1],-1).permute(1,0)
                    feats_sample = greedsampler_perimg.run(feats)
                    train_feature_list.append(feats_sample)


            train_features = torch.cat(train_feature_list,dim=0)
            train_features = F.normalize(train_features, dim=1)
            torch.save(train_features.cpu(),f'{mid_output_path}/{self.category}.pth')
            train_features = train_features.cpu().numpy()
            kmeans=KMeans(init='k-means++',n_clusters=num_cluster)
            c = kmeans.fit(train_features)
            cluster_centers = torch.from_numpy(c.cluster_centers_)
            torch.save(cluster_centers.cpu(),f'{mid_output_path}/{self.category}_k{num_cluster}.pth')
            train_features_sampled = cluster_centers.cuda()
            train_features_sampled = train_features_sampled.unsqueeze(0).unsqueeze(0)
            train_features_sampled = train_features_sampled.permute(0, 3, 1, 2)
        else:
            train_features = torch.load(f'{mid_output_path}/{self.category}.pth').cuda()
            train_features = train_features.cpu().numpy()
            kmeans=KMeans(init='k-means++',n_clusters=num_cluster)
            c = kmeans.fit(train_features)
            cluster_centers = torch.from_numpy(c.cluster_centers_)
            train_features_sampled = cluster_centers.cuda()
            train_features_sampled = train_features_sampled.unsqueeze(0).unsqueeze(0)
            train_features_sampled = train_features_sampled.permute(0, 3, 1, 2)

        ##SAVE SegIMAGES##
        savepath = os.path.join('./segimgs', self.dataset_name, f'num_clusters={num_cluster}', self.category )

        #train_savepath = f'{savepath}/train'
        train_savepath = os.path.join(savepath, 'train', 'good')
        if not os.path.exists(train_savepath):
            os.makedirs(train_savepath)
        self.save_img(dataloader, train_features_sampled, net, train_savepath)

        defect_types = os.listdir(os.path.join(self.dataset_root, 'test'))
        for type in defect_types:
            dataset_test = MVTecLocoDataset(root_dir=self.dataset_root, category=f'test/{type}',resize_shape=self.image_size )
            dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=4)
        
            #test_savepath = f'{savepath}/test/logical_anomalies'
            test_savepath = os.path.join(savepath, 'test', type)
            if not os.path.exists(test_savepath):
                os.makedirs(test_savepath)
            self.save_img(dataloader_test, train_features_sampled, net, test_savepath)
        
    def save_img(self, dataloader, train_features_sampled, net, savapath):
        for i, Img in enumerate(dataloader):
            image = Img['image']
            #heatmap,heatmap_intra = get_heatmaps(image, train_features_sampled, net)
            heatmap_intra = self.get_heatmaps(image, train_features_sampled, net)

            img_savepath = f'{savapath}/{i:03}'
            if not os.path.exists(img_savepath):
                os.makedirs(img_savepath)
            #imageo.save(f'{img_savepath}/imgo.jpg')
            #see_image(image, heatmap,img_savepath,heatmap_intra)
            self.see_image(image, img_savepath,heatmap_intra)

    def get_heatmaps(self, img,query_feature,net):
        with torch.no_grad():
            feats1, f1_lowdim = net(img.cuda())
        sfeats1 = query_feature
        attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
        attn_intra -= attn_intra.mean([3, 4], keepdims=True)
        attn_intra = attn_intra.clamp(0).squeeze(0)
        heatmap_intra = F.interpolate(
            attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
        img_crf = img.squeeze()
        crf_result = dense_crf(img_crf,heatmap_intra)
        heatmap_intra = torch.from_numpy(crf_result)
        #return seg_map,heatmap_intra
        return heatmap_intra

    #def see_image(data,heatmap,savepath,heatmap_intra):
    def see_image(self, data,savepath,heatmap_intra):
        for i in range(heatmap_intra.shape[0]):
            heat=heatmap_intra[i,:,:].cpu().numpy()
            heat = np.round(heat*128).astype(np.uint8)
            cv2.imwrite(f'{savepath}/heatresult{i}.jpg', heat)

