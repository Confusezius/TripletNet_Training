############### LIBRARIES ###################################
import numpy as np, os, sys, pandas as pd, csv
import torch, torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


################ FUNCTIONS TO RETURN TRAINING AND VALIDATION DATASETS ############################
def give_CUB200_training_sets(source_path, perc_data, tv_split, seed):
    image_sourcepath  = source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x], key=lambda x: int(x.split('.')[0]))
    conversion    = {int(x.split('.')[0]):x.split('.')[-1] for x in image_classes}
    image_list    = {int(key.split('.')[0]):sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key) if '._' not in x]) for key in image_classes}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    random.seed(seed)
    random.shuffle(image_list)

    image_list    = image_list[:int(len(image_list)*perc_data)]

    datalen = int(len(image_list)*tv_split)
    train_image_list = image_list[:datalen]
    val_image_list   = image_list[datalen:]

    train_image_dict, val_image_dict  = {},{}

    for key, img_path in train_image_list:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(img_path)


    for key, img_path in val_image_list:
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(img_path)

    return BaseTripletDataset(train_image_dict, dataset='cub200'), BaseTripletDataset(val_image_dict, dataset='cub200')


def give_CelebA_training_sets(source_path, perc_data, tv_split, seed):
    image_sourcepath  = source_path+'/images'
    images_dataframe  = pd.read_table(source_path+'/feature_targets.txt', header=1, delim_whitespace=True)
    classnames, imagenames = images_dataframe.columns, np.array(images_dataframe.index)
    images_dataframe  = np.array(images_dataframe)

    image_dict = {idx:[image_sourcepath+'/'+imagenames[x] for x in np.where(np.array(images_dataframe)[:,idx]==1)[0]] for idx in range(np.array(images_dataframe).shape[-1])}

    random.seed(seed)

    train_image_dict, val_image_dict = {},{}
    for key,items in image_dict.items():
        random.shuffle(items)
        datalen = int(len(items)*perc_data)
        items   = items[:datalen]
        datalen = int(len(items)*tv_split)
        train_image_dict[key] = items[:datalen]
        val_image_dict[key] = items[datalen]

    return BaseTripletDataset(train_image_dict, dataset='celeba'), BaseTripletDataset(val_image_dict, dataset='celeba')


def give_CARS196_training_sets(source_path, perc_data, tv_split, seed):
    #Get Foldernames, i.e. classnames, and generate list of all images and classes.
    image_sourcepath  = source_path+'/images'
    image_classes = sorted([x for x in os.listdir(image_sourcepath)])
    conversion    = {i:x for i,x in enumerate(image_classes)}
    image_list    = {i:sorted([image_sourcepath+'/'+key+'/'+x for x in os.listdir(image_sourcepath+'/'+key)]) for i,key in enumerate(image_classes)}
    image_list    = [[(key,img_path) for img_path in image_list[key]] for key in image_list.keys()]
    image_list    = [x for y in image_list for x in y]

    #Remove Grayscale Images
    idxs_2_rem = [445,451,467,476,479,480,483,484,503,511,524,951,1018,1708,2502,2883,7403,7457]

    for rem_idx in idxs_2_rem[::-1]:
        del image_list[rem_idx]

    #Shuffle and divide (depending on percentage of data to use and training/validation split)
    random.seed(seed)
    random.shuffle(image_list)
    image_list    = image_list[:int(len(image_list)*perc_data)]

    datalen = int(len(image_list)*tv_split)
    train_image_list = image_list[:datalen]
    val_image_list   = image_list[datalen:]


    #Generate Image Dictionaries
    train_image_dict, val_image_dict  = {},{}

    for key, img_path in train_image_list:
        if not key in train_image_dict.keys():
            train_image_dict[key] = []
        train_image_dict[key].append(img_path)


    for key, img_path in val_image_list:
        if not key in val_image_dict.keys():
            val_image_dict[key] = []
        val_image_dict[key].append(img_path)


    return BaseTripletDataset(train_image_dict, dataset='cars196'), BaseTripletDataset(val_image_dict, dataset='cars196')




class BaseTripletDataset(Dataset):
    def __init__(self, image_dict, dataset):
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.image_dict = image_dict

        self.avail_classes = list(self.image_dict.keys())

        ### Set of precomputed dataset means and stds, as well as useful transformations.
        transf_list = [transforms.RandomHorizontalFlip(0.5)]
        if dataset=='cub200':
            means = np.array([0.47819992, 0.49399305, 0.4262326 ])
            stds  = np.array([0.05760238, 0.05675151, 0.06677961])
            transf_list.append(transforms.CenterCrop([256,256]))
        if dataset=='cars196':
            means = np.array([0.4706145 , 0.46000465, 0.45479808])
            stds  = np.array([0.04725483, 0.04907224, 0.04912915])
            transf_list.extend([transforms.Resize((700,480)), transforms.CenterCrop([256,256])])
        if dataset=='celeba':
            means   = np.array([0.5064, 0.4263, 0.3845])
            stds    = np.array([0.1490, 0.1443, 0.1469])

        transf_list.extend([transforms.ToTensor(), transforms.Normalize(mean=means, std=stds)])

        self.transform = transforms.Compose(transf_list)



    def __getitem__(self, idx):
        anchor_class    = random.choice(list(self.image_dict.keys()))
        positive_class  = anchor_class
        negative_class  = random.choice([x for x in self.image_dict.keys() if x!=anchor_class])

        anchor_path     = random.choice(self.image_dict[anchor_class])
        positive_path   = random.choice([x for x in self.image_dict[positive_class] if x!=anchor_path])
        negative_path   = random.choice(self.image_dict[negative_class])

        anchor_img   = self.transform(Image.open(anchor_path))
        pos_img      = self.transform(Image.open(positive_path))
        neg_img      = self.transform(Image.open(negative_path))

        self.descriptor = 'Anchor Class: {} | Negative Class: {}'.format(anchor_class, negative_class)

        return anchor_img,pos_img,neg_img

    def __len__(self):
        return self.n_files



################## WRITE TO CSV FILE #####################
class CSV_Writer():
    def __init__(self, save_path, columns):
        self.save_path = save_path
        self.columns   = columns

        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=",")
            writer.writerow(self.columns)

    def log(self, inputs):
        with open(self.save_path, "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(inputs)


################# ACQUIRE NUMBER OF WEIGHTS #################
def gimme_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

################## PLOT SUMMARY IMAGE #####################
class InfoPlotter():
    def __init__(self, save_path, title='Training Log', figsize=(15,10)):
        self.save_path = save_path
        self.title     = title
        self.figsize   = figsize

    def make_plot(self, x, y1, y2):
        plt.style.use('ggplot')
        f,ax = plt.subplots(1)
        ax.set_title(self.title)
        ax.plot(x, y1, '-k', label='Training')
        axx = ax.twinx()
        axx.plot(x, y2, '-r', label='Validation')
        f.set_size_inches(self.figsize[0], self.figsize[1])
        f.savefig(self.save_path)
        plt.close()
