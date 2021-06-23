import pandas as pd
import os
import cv2
import numpy as np
# np.set_printoptions(threshold=np.inf)
import xml.etree.ElementTree as ET
import pickle

def load_yolo(root,filename):
    images,images_path,labels,labels_path=[],[],[],[]
    with open(os.path.join(root, filename)) as f:
        rows=f.readlines()
        for row in rows:
            images_path.append(root+'images/'+filename.rstrip('.txt')+'2021/'+row.rstrip('\n')+'.jpg')
            labels_path.append(root+'labels/'+filename.rstrip('.txt')+'2021/'+row.rstrip('\n')+'.txt')
    for i in range(6):
        image=cv2.imread(images_path[i],cv2.IMREAD_GRAYSCALE)
        with open(labels_path[i]) as f:
            row=f.readline()
            while row:
                cols=row.split(' ')
                label=int(cols[0])
                x,y,w,h=[float(col) for col in cols[1:]]
                labels.append(label)
                x1,y1=int(image.shape[1]*x-image.shape[1]*w/2),int(image.shape[0]*y-image.shape[0]*h/2)
                x2,y2=int(image.shape[1]*x+image.shape[1]*w/2),int(image.shape[0]*y+image.shape[0]*h/2)
                images.append(image[x1:x1+x2,y1:y1+y2])
                row=f.readline()
        # print('index:',i,'total:',len(images_path))
    return labels,images

def load_pascal_voc(root='./VOCdevkit/VOC2007/',filename='trainval.txt'):
    images,images_path,labels,labels_path=[],[],[],[]
    with open(root+'ImageSets/Main/'+filename) as f:
        rows=f.readlines()
        for row in rows:
            images_path.append(root+'JPEGImages/'+row.rstrip('\n')+'.jpg')
            labels_path.append(root+'Annotations/'+row.rstrip('\n')+'.xml')
    for i in range(len(images_path)):
        image=cv2.imread(images_path[i],cv2.IMREAD_GRAYSCALE)
        element=ET.parse(labels_path[i]).getroot()
        objects=element.findall('object')
        for obj in objects:
            x1,y1=int(obj.find('bndbox').find('xmin').text),int(obj.find('bndbox').find('ymin').text)
            x2,y2=int(obj.find('bndbox').find('xmax').text),int(obj.find('bndbox').find('ymax').text)
            bnd_image=image[y1:y2,x1:x2]
            # 排除错误数据
            if bnd_image.shape[0]!=0 and bnd_image.shape[1]!=0:
                labels.append(obj.find('name').text)
                # print(obj.find('name').text,image.shape,image[y1:y2,x1:x2].shape)
                images.append(bnd_image)
        print('index:',i,'totals:',len(images_path))
    return labels,images

def _change_one_hot_label(X):
    label_list=[]
    for label in X:
        if label not in label_list:
            label_list.append(label)
    # print(label_list)
    T = np.zeros(shape=(len(X), len(label_list)),dtype=int)
    for idx, row in enumerate(T):
        row[label_list.index(X[idx])] = 1
    return T
def load_dataset(filename='./dataset.pkl',flatten=False,resize=(0,0),onehot=True,normalize=True,concat=False):
    with open(filename,'rb') as f:
        dataset=pickle.load(f)
    if resize[0]!=0 and resize[1]!=0:
        for key in ('train_images','test_images'):
            for i in range(len(dataset[key])):
                dataset[key][i]=cv2.resize(dataset[key][i],resize)
    if flatten:
        for key in ('train_images','test_images'):
            dataset[key]=np.array(dataset[key]).transpose((0,2,1)).reshape(len(dataset[key]),1,-1)
    if normalize:
        for key in ('train_images','test_images'):
            dataset[key]=np.array(dataset[key],dtype=np.float)
            dataset[key] /= 255.0
    if onehot:
        dataset['train_labels'] = _change_one_hot_label(dataset['train_labels'])
        dataset['test_labels'] = _change_one_hot_label(dataset['test_labels'])
    if concat:
        return (np.concatenate((dataset['train_labels'],dataset['test_labels']),axis=0),np.concatenate((dataset['train_images'],dataset['test_images']),axis=0))
    return (dataset['train_labels'],dataset['train_images']),(dataset['test_labels'],dataset['test_images'])


def get_range(filename='./dataset.pkl'):
    max_w=0
    max_h=0
    min_w=100000
    min_h=100000
    with open(filename,'rb') as f:
        dataset=pickle.load(f)
    for key in ('train_images','test_images'):
        for i in range(len(dataset[key])):
            w,h=dataset[key][i].shape[1],dataset[key][i].shape[0]
            if w>max_w:
                max_w=w
            if w<min_w:
                min_w=w
            if h>max_h:
                max_h=h
            if h<min_h:
                min_h=h
    # (7, 854, 6, 791)
    return min_w,max_w,min_h,max_h



if __name__=='__main__':
    train_labels,train_images=load_pascal_voc()
    test_labels,test_images=load_pascal_voc(filename='test.txt')
    print('train_iamges_size:',len(train_images))
    with open ('./dataset.pkl','wb') as f:
        pickle.dump({'train_labels':train_labels,'train_images':train_images,'test_labels':test_labels,'test_images':test_images},f)
        f.close()

    # print(get_range())

