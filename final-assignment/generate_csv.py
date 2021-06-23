import pandas as pd
import os
import cv2
# image_name,image_width,image_height,center_x,center_y,bnd_width,bnd_height,label

def load_yolo(root,filename,data):
    images,images_path,labels,labels_path=[],[],[],[]
    with open(os.path.join(root, filename)) as f:
        rows=f.readlines()
        for row in rows:
            images_path.append(root+'images/'+filename.rstrip('.txt')+'2021/'+row.rstrip('\n')+'.jpg')
            labels_path.append(root+'labels/'+filename.rstrip('.txt')+'2021/'+row.rstrip('\n')+'.txt')
    for i in range(len(images_path)):
        image=cv2.imread(images_path[i],cv2.IMREAD_GRAYSCALE)
        # print(image.shape)
        with open(labels_path[i]) as f:
            row=f.readline()
            while row:
                data['image_name'].append(images_path[i].split("/")[-1])
                data['image_width'].append(image.shape[1])
                data['image_height'].append(image.shape[0])
                cols=row.split(' ')
                label=int(cols[0])
                x,y,w,h=[float(col) for col in cols[1:]]
                data['center_x'].append(x)
                data['center_y'].append(y)
                data['bnd_width'].append(w)
                data['bnd_height'].append(h)
                data['label'].append(label)
                # labels.append(label)
                # x1,y1=int(image.shape[1]*x-image.shape[1]*w/2),int(image.shape[0]*y-image.shape[0]*h/2)
                # x2,y2=int(image.shape[1]*x+image.shape[1]*w/2),int(image.shape[0]*y+image.shape[0]*h/2)
                # images.append(image[x1:x1+x2,y1:y1+y2])

                # image.release()
                # del image
                # gc.collect()
                row=f.readline()
        print('index:',i,'total:',len(images_path))
    return labels,images


if __name__=='__main__':
    data={
        'image_name':[],
        'image_width':[],
        'image_height':[],
        'center_x':[],
        'center_y':[],
        'bnd_width':[],
        'bnd_height':[],
        'label':[]
    }
    train_labels,train_images=load_yolo('E://Yangdingming/Documents/tube_yolo/tube/','train.txt',data)
    test_labels,test_images=load_yolo('E://Yangdingming/Documents/tube_yolo/tube/','val.txt',data)
    df2=pd.DataFrame(data)
    df2.to_csv('dataset2.csv')
