import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import pickle
import copy
import sys
import matplotlib.pyplot as plt
import scipy.misc as misc
# 确保yolo包能被正确导入
# 不再同一个目录下的python脚本需要添加路径
sys.path.insert(0,'../')
import yolo.config as cfg


class pascal_voc(object):
    
    def __init__(self, phase='train', rebuild=False):
        # voc数据路径
        if phase=='train':
            self.devkil_path = os.path.join(cfg.PASCAL_PATH, 'VOCdevkit')
        else:
            self.devkil_path = os.path.join(cfg.PASCAL_PATH,'VOCdevkit_test')
            cfg.FLIPPED=False
        
        self.data_path = os.path.join(self.devkil_path, 'VOC2007')
        
        # self.data_path = cfg.PASCAL_PATH
        self.cache_path = cfg.CACHE_PATH
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        # 网格数量
        self.cell_size = cfg.CELL_SIZE
        # 类别名称
        self.classes = cfg.CLASSES
        # 类别到标号索引的字典
        self.class_to_ind = dict(zip(self.classes, range(len(self.classes))))
        # 是否开启图像翻转
        self.flipped = cfg.FLIPPED
        self.phase = phase
        self.rebuild = rebuild
        # 当前数据集的游标
        self.cursor = 0
        self.epoch = 1
        # GT label information
        self.gt_labels = None
        
        self.prepare()
    
    # 得到一个batch的图片和GT
    def get(self):
        images = np.zeros(
            (self.batch_size, self.image_size, self.image_size, 3))
        labels = np.zeros(
            (self.batch_size, self.cell_size, self.cell_size, 5+len(self.classes)))
        count = 0
        while count < self.batch_size:
            imname = self.gt_labels[self.cursor]['imname']
            flipped = self.gt_labels[self.cursor]['flipped']
            images[count, :, :, :] = self.image_read(imname, flipped)
            labels[count, :, :, :] = self.gt_labels[self.cursor]['label']
            count += 1
            self.cursor += 1
            if self.cursor >= len(self.gt_labels):
                np.random.shuffle(self.gt_labels)
                self.cursor = 0
                self.epoch += 1
        return images, labels

    def image_read(self, imname, flipped=False):
        image = cv2.imread(imname)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image = (image / 255.0) * 2.0 - 1.0
        # 左右翻转
        if flipped:
            image = image[:, ::-1, :].copy()
        return image

    def prepare(self):
        gt_labels = self.load_labels()
        if self.flipped:
            print('Appending horizontally-flipped training examples ...')
            gt_labels_cp = copy.deepcopy(gt_labels)
            for idx in range(len(gt_labels_cp)):
                gt_labels_cp[idx]['flipped'] = True
                gt_labels_cp[idx]['label'] =\
                    gt_labels_cp[idx]['label'][:, ::-1, :]
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        if gt_labels_cp[idx]['label'][i, j, 0] == 1:
                            # filp axis x coordinator
                            gt_labels_cp[idx]['label'][i, j, 1] = \
                                self.image_size - 1 -\
                                gt_labels_cp[idx]['label'][i, j, 1]
            # 如果图像翻转，GT label数量扩大一倍
            gt_labels += gt_labels_cp
        if self.phase=='train':
            np.random.shuffle(gt_labels)
        self.gt_labels = gt_labels
        return gt_labels
    
    # 得到真实Label
    def load_labels(self):
        # 缓存文件
        cache_file = os.path.join(
            self.cache_path, 'pascal_' + self.phase + '_gt_labels.pkl')
        
        # 判断一个路径下是否是一个文件，如果该文件路径不存在
        # 则判断为false,相当于os.path.exists()
        if os.path.isfile(cache_file) and not self.rebuild:
            print('Loading gt_labels from: ' + cache_file)
            with open(cache_file, 'rb') as f:
                gt_labels = pickle.load(f)
            return gt_labels

        print('Processing gt_labels from: ' + self.data_path)

        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        if self.phase == 'train':
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'trainval.txt')
        else:
            txtname = os.path.join(
                self.data_path, 'ImageSets', 'Main', 'test.txt')
                
        # 得到训练或者测试的图片名字索引
        with open(txtname, 'r') as f:
            self.image_index = [x.strip() for x in f.readlines()]
        
        # 真实标记信息
        gt_labels = []
        for index in self.image_index:
            # 得到一张图片的label信息
            label, num = self.load_pascal_annotation(index)
            if num == 0:
                continue
            imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
            gt_labels.append({'imname': imname,
                              'label': label,
                              'flipped': False})
        print('Saving gt_labels to: ' + cache_file)
        with open(cache_file, 'wb') as f:
            pickle.dump(gt_labels, f)
        return gt_labels

    # 加载voc标注信息
    def load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """

        imname = os.path.join(self.data_path, 'JPEGImages', index + '.jpg')
        im = cv2.imread(imname)
        # 得到宽高缩放比例
        h_ratio = 1.0 * self.image_size / im.shape[0]
        w_ratio = 1.0 * self.image_size / im.shape[1]
        # im = cv2.resize(im, [self.image_size, self.image_size])
    
        # 一张图片GTbox label
        label = np.zeros((self.cell_size, self.cell_size, 5+len(self.classes)))
        # xml文件路径
        filename = os.path.join(self.data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        # return a list
        objs = tree.findall('object')

        for obj in objs:
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            # 将坐标值按照图像缩放比例进行缩放
            # 减1得到像素在原图索引
            x1 = max(min((float(bbox.find('xmin').text) - 1) * w_ratio, self.image_size - 1), 0)
            y1 = max(min((float(bbox.find('ymin').text) - 1) * h_ratio, self.image_size - 1), 0)
            x2 = max(min((float(bbox.find('xmax').text) - 1) * w_ratio, self.image_size - 1), 0)
            y2 = max(min((float(bbox.find('ymax').text) - 1) * h_ratio, self.image_size - 1), 0)
            
            # 类别id
            cls_ind = self.class_to_ind[obj.find('name').text.lower().strip()]
            # gt box 
            # center_x,center_y,w,h
            boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
            # 计算GTbox中心行列索引位于第几个网格中
            # int向下取整
            x_ind = int(boxes[0] * self.cell_size / self.image_size)
            y_ind = int(boxes[1] * self.cell_size / self.image_size)
            # bounding box 置信度标记
            # 每个网格是否有目标
            # ----------------------------------------------------------------#
            # --------------注意每个grid最多只对应一个真实的box----------------#
            # ----------------------------------------------------------------#
            if label[y_ind, x_ind, 0] == 1:
                continue
            label[y_ind, x_ind, 0] = 1
            label[y_ind, x_ind, 1:5] = boxes
            label[y_ind, x_ind, 5 + cls_ind] = 1

        return label, len(objs)
    
    def visualize(self):
        # visual one image and its bounding box
        img_info = self.gt_labels[109]
        img_path = img_info['imname']
        flip = img_info['flipped']
        print('imgpath',img_path,'flipped',flip)
        bbox = img_info['label'][:,:,1:5]
        # convert bbox to (xmin,ymin,xmax,ymax)
        # left corner
        lf = np.int32(np.round(bbox[:,:,0:2]-0.5*bbox[:,:,2:4]))
        # right corner
        rg = np.int32(np.round(bbox[:,:,0:2]+0.5*bbox[:,:,2:4]))
        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = cv2.resize(img,(self.image_size,self.image_size))
        
        if flip:
            print('flipped image')
            # note must copy a bug for opencv
            img = img[:,::-1,:].copy()
            # or using builtin function
            # img = cv2.flip(img,1)
        
        for i in range(self.cell_size):
            for j in range(self.cell_size):
                if img_info['label'][i,j,0]==1:
                    
                    print('x1,y1',lf[i,j,:],'x2,y2',rg[i,j,:])
                    cv2.rectangle(img,tuple(lf[i,j,:]),tuple(rg[i,j,:]),(255,0,0))
                    # plt.ion()
                    plt.imshow(img)
                    plt.show()
            

if __name__ == '__main__':
    voc = pascal_voc()
    voc.visualize()
