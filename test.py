import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
import matplotlib.pyplot as plt
from collections import Counter

class Detector(object):

    def __init__(self, net, weight_file):
        self.net = net
        # 权重文件名
        self.weights_file = weight_file
        # 类别名称
        self.classes = cfg.CLASSES
        # 类别数量
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        # 每个特定类别的得分阈值
        self.threshold = cfg.THRESHOLD
        # IOU阈值
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)
    
    # 显示检测结果
    def draw_result(self, img, result):
        for i in range(len(result)):
            x = int(round(result[i][1]))
            y = int(round(result[i][2]))
            w = int(round(result[i][3] / 2))
            h = int(round(result[i][4] / 2))
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(img, (x - w, y - h - 20),
                          (x + w, y - h), (125, 125, 125), -1)
            lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
            # 输出类别加上概率值
            cv2.putText(
                img, result[i][0] + ' : %.2f' % result[i][5],
                (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, lineType)
    
    # 得到检测结果
    def detect(self, img):
        # 保存原始检测图像的宽度和高度信息
        img_h, img_w, _ = img.shape
        # 输入缩放到448X448X3
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        # 构成一个batch
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]
        
        # bbox坐标变换得到对应原始图像大小的BBOX坐标
        for i in range(len(result)):
            result[i][1] *= (1.0 * img_w / self.image_size)
            result[i][2] *= (1.0 * img_h / self.image_size)
            result[i][3] *= (1.0 * img_w / self.image_size)
            result[i][4] *= (1.0 * img_h / self.image_size)

        return result
    
    # get shape 448x448 coordinate
    def detect_without_resize(self, img):
        # 输入缩放到448X448X3
        inputs = cv2.resize(img, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
        inputs = (inputs / 255.0) * 2.0 - 1.0
        # 构成一个batch
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))

        result = self.detect_from_cvmat(inputs)[0]
        return result


    def detect_from_cvmat(self, inputs):
        # 1x1470
        net_output = self.sess.run(self.net.logits,
                                   feed_dict={self.net.images: inputs})
        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret_output(net_output[i]))

        return results
    
    # 得到检测结果的插值输出
    def interpret_output(self, output):
    
        # 7x7x2x20
        # 每个grid20个条件类别概率乘以每个Bbox的置信度
        # 得到每个Bbox相对于每个特定类别的得分
        probs = np.zeros((self.cell_size, self.cell_size,
                          self.boxes_per_cell, self.num_class))
        # 条件类别概率预测
        # 7x7x20
        class_probs = np.reshape(
            output[0:self.boundary1],
            (self.cell_size, self.cell_size, self.num_class))
        # bbox得分预测
        # 7x7x2
        scales = np.reshape(
            output[self.boundary1:self.boundary2],
            (self.cell_size, self.cell_size, self.boxes_per_cell))
        # bbox相对坐标预测输出
        boxes = np.reshape(
            output[self.boundary2:],
            (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        
        # bbox偏移
        # x方向偏移
        offset = np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell)
        offset = np.transpose(
            np.reshape(
                offset,
                [self.boxes_per_cell, self.cell_size, self.cell_size]),
            (1, 2, 0))
        
        # 得到正真的BBOX预测结果
        # [x,y,w,h]
        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        # 将x,y坐标除以self.cell_size归一化到[0-1]
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
        # h,w的平方归一化到[0-1]
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        
        # 恢复到网络输入时候的尺寸大小
        boxes *= self.image_size
        
        # 计算每个gird的每个bbox特定类别的得分
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                probs[:, :, i, j] = np.multiply(
                    class_probs[:, :, j], scales[:, :, i])
        
        # 根据得分阈值过滤概率
        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        # np.nonzero()返回非0元素的索引
        # 四元组list
        filter_mat_boxes = np.nonzero(filter_mat_probs)
        
        # 7x7x2x4
        # 得到满足得分阈值的BOX坐标
        # shape = nums*4
        boxes_filtered = boxes[filter_mat_boxes[0],
                               filter_mat_boxes[1], filter_mat_boxes[2]]
#        print('remain boxes',boxes_filtered.shape)
#        print(boxes_filtered)                    
        # 得到满足得分阈值的BOX类别概率
        # shape = nums*1
        probs_filtered = probs[filter_mat_probs]
#        print('pros remian',probs_filtered.shape)
#        print(probs_filtered)
        
        # 觉得使用probs更好
        # 得到满足条件的类别索引
        # filter_mat_probs is a bool array 
        # can only find the first True
        
        # 得到挑选出的类别标号
        classes_num_filtered = np.argmax(
            probs, axis=3)[
            filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]
        
#        print('class_num',classes_num_filtered.shape)
#        print(classes_num_filtered)
        
#        # 按照概率从大到小排序
#        argsort = np.array(np.argsort(probs_filtered))[::-1]
#        
#        boxes_filtered = boxes_filtered[argsort]
#        probs_filtered = probs_filtered[argsort]
#        classes_num_filtered = classes_num_filtered[argsort]
#        
#        # NMS
#        for i in range(len(boxes_filtered)):
#            if probs_filtered[i] == 0.0:
#                continue
#            for j in range(i + 1, len(boxes_filtered)):
#                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
#                    probs_filtered[j] = 0.0
#
#        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
#        
#        # 得到经过NMS抑制之后的得到的BOX和类别概率
#        boxes_filtered = boxes_filtered[filter_iou]
#        probs_filtered = probs_filtered[filter_iou]
#        classes_num_filtered = classes_num_filtered[filter_iou]
        boxes_filtered,classes_num_filtered,probs_filtered = self.NMS(boxes_filtered,
                                                                      classes_num_filtered,
                                                                      probs_filtered)

        result = []
        for i in range(len(boxes_filtered)):
            result.append(
                [self.classes[classes_num_filtered[i]],
                 boxes_filtered[i][0],
                 boxes_filtered[i][1],
                 boxes_filtered[i][2],
                 boxes_filtered[i][3],
                 probs_filtered[i]])

        return result
    
    
    def NMS(self,predict_bboxes,predict_classes,predict_probs):
        '''
        predict_bboxes:remained bboxes after filter some boxes dont satisfy score IOU
        predict_classes:remained bboxes classes
        predict_probs:remained bboxes score
        '''
        counter = Counter(predict_classes)
        #print(counter)
        res_bboxes=[]
        res_classes = []
        res_probs = []
        for cls in counter.keys():
            cur_index = np.where(predict_classes==cls)[0]
            #print(cur_index,len(cur_index))
            cur_class = predict_classes[cur_index]
            #print('cur_class',cur_class)
            cur_bboxes = predict_bboxes[cur_index]
            #print('cur_bboxes',cur_bboxes)
            cur_probs = predict_probs[cur_index]
            #print('cur_probs',cur_probs)
            assert len(cur_index)==counter[cls],'Nums of current class not equal'
            argsort = np.argsort(cur_probs)[::-1]
            sorted_bboxes = cur_bboxes[argsort]
            #print('sorted_box',sorted_bboxes)
            sorted_classes = cur_class[argsort]
            #print('sorted_class',sorted_classes)
            sorted_probs = cur_probs[argsort]
            #print('sorted_probs',sorted_probs)
            
            for i in range(len(sorted_bboxes)):
                if sorted_probs[i] == 0.0:
                    continue
                for j in range(i+1,len(sorted_bboxes)):
                    if self.iou(sorted_bboxes[i],sorted_bboxes[j])>self.iou_threshold:
                       sorted_probs[j] = 0.0
            filter_iou = np.array(sorted_probs > 0.0, dtype='bool')
            bboxes_filtered = sorted_bboxes[filter_iou].tolist()
            classes_filtered = sorted_classes[filter_iou].tolist()
            probs_filtered = sorted_probs[filter_iou].tolist()
            
            res_classes.extend(classes_filtered)
            res_probs.extend(probs_filtered)
            for bbox in bboxes_filtered:
                res_bboxes.append(bbox)
        res_bboxes = np.array(res_bboxes)
        #print(res_bboxes,res_classes,res_probs,sep='\n')
        return res_bboxes,res_classes,res_probs
                
            
    
    def iou(self, box1, box2):
        tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
            max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
        lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
            max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
        inter = 0 if tb < 0 or lr < 0 else tb * lr
        return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)


    def camera_detector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()

        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            result = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time: {:.3f}s'.format(
                detect_timer.average_time))

            self.draw_result(frame, result)
            cv2.imshow('Camera', frame)
            cv2.waitKey(wait)
            ret, frame = cap.read()
            
    
    # 图片检测器
    def image_detector(self, imname, wait=0):
        detect_timer = Timer()
        image = cv2.imread(imname)

        detect_timer.tic()
        result = self.detect(image)
        detect_timer.toc()
        print('Average detecting time: {:.3f}s'.format(
            detect_timer.average_time))

        self.draw_result(image, result)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image)
        # cv2.waitKey(wait)
        plt.imshow(image)
        plt.show()
        if not os.path.exists(cfg.OUTPUT_DIR):
            os.makedirs(cfg.OUTPUT_DIR)
        plt.imsave(os.path.join(cfg.OUTPUT_DIR,os.path.basename(imname)),image)
        print('saving done')
    
    # get detect result for compute MAP    
    def get_dectect_result(self):
        self.voc_test = generate_GTbox_labels(save=True)
        infos = self.voc_test.gt_labels
        class_to_ind = dict(zip(self.classes,range(self.num_class)))
        predict_bboxes = []
        predict_labels = []
        predict_scores = []
        nums = len(infos)
        k = 0
        test_timer = Timer()
        for info in infos:
            k += 1
            test_timer.tic()
            testname = info['imname']
            #print(testname)
            img = cv2.imread(testname)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            
            res = self.detect_without_resize(img)
            if len(res)!=0:
                for i in res:
                    i[0] = class_to_ind[i[0]]
                res = np.array(res)
                bbox = res[:,1:5]
                # convert to [ymin,xmin,ymax,xmax]
                xmin = bbox[:,0]-0.5*bbox[:,2]
                ymin = bbox[:,1]-0.5*bbox[:,3]
                xmax = bbox[:,0]+0.5*bbox[:,2]
                ymax = bbox[:,1]+0.5*bbox[:,3]
                bbox_tran = np.stack((ymin,xmin,ymax,xmax),axis=-1)
                
                label = res[:,0]
                score = res[:,-1]
                predict_bboxes.append(bbox_tran)
                predict_labels.append(label)
                predict_scores.append(score)
            else:
                predict_bboxes.append(np.array([0,0,0,0]))
                predict_labels.append(res)
                predict_scores.append(np.array([0]))
            test_timer.toc()
            # print(predict_bboxes,predict_labels,predict_scores,sep='\n')
            r = '\r>>>Image:{:4d}/{:4d} Test Time:{:.3f}s Total Time:{:5.2f}s'.format(
                    k,nums,test_timer.average_time,test_timer.total_time)
            print(r,end='',flush=True)

        np.save('./result/predict_bboxes.npy',predict_bboxes)
        np.save('./result/predict_labels.npy',predict_labels)
        np.save('./result/predict_scores.npy',predict_scores)
        print('write done!')


def generate_GTbox_labels(rebuild=False,save=True):
    
    voc_test = pascal_voc(phase='test',rebuild=rebuild)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    labels_info = voc_test.gt_labels
    nums = len(labels_info)
    # print(nums)
    GT_box = []
    GT_labels = []
    if save:
        for i,info in enumerate(labels_info):
            # print(info['imname'])
            lab = info['label'][...,1:]
            response = info['label'][...,0]
            index = np.nonzero(response)
            # convert to (ymin,xmin,ymax,xmax)
            xmin = lab[...,0]-0.5*lab[...,2]
            ymin = lab[...,1]-0.5*lab[...,3]
            xmax = lab[...,0]+0.5*lab[...,2]
            ymax = lab[...,1]+0.5*lab[...,3]
            bboxes = np.stack([ymin,xmin,ymax,xmax],axis=-1)[index]
            GT_box.append(bboxes)
            # print(GT_box)
            classes = np.argmax(lab[...,4:],axis=-1)[index]
            GT_labels.append(classes)
            #print(GT_box,GT_labels,sep='\n')
            r = '\r>>>>>>Generate:{:4d}/{}'.format(i+1,nums)
            print(r,end='',flush=True)
            
        np.save(os.path.join(result_dir,'GT_box.npy'),GT_box)
        np.save(os.path.join(result_dir,'GT_labels.npy'),GT_labels)
    return voc_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--weight_dir', default='weights', type=str)
    parser.add_argument('--data_dir', default="data/customed_data", type=str)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    yolo = YOLONet(is_training=False)
    weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
    detector = Detector(yolo, weight_file)

    # detect from camera
    # cap = cv2.VideoCapture(0)
    # detector.camera_detector(cap)

    # detect from image file
    # imname = 'test/4.jpg'
    # detector.image_detector(imname)
    detector.get_dectect_result()
    

def compute_map():
    from eval_tool import eval_detection_voc
    predict_boxs = np.load('./result/predict_bboxes.npy')
    predict_labels = np.load('./result/predict_labels.npy')
    print(len(predict_boxs))
    predict_scores = np.load('./result/predict_scores.npy')
    #print(predict_labels.shape)
    gt_boxs = np.load('./result/GT_box.npy')
    gt_labels = np.load('./result/GT_labels.npy')
    print(len(gt_boxs))
    res = eval_detection_voc(predict_boxs,predict_labels,predict_scores,gt_boxs,gt_labels)
    print(res)

if __name__ == '__main__':
    main()
    compute_map()
    #generate_GTbox_labels()
