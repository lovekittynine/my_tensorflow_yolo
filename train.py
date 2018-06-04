import os
import argparse
import datetime
import tensorflow as tf
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc import pascal_voc
import logging
slim = tf.contrib.slim


class Solver(object):

    def __init__(self, net, data):
        self.net = net
        self.data = data
        # 权重文件路径
        self.weights_file = cfg.WEIGHTS_FILE
        # 迭代步数
        self.max_iter = cfg.MAX_ITER
        self.initial_learning_rate = cfg.LEARNING_RATE
        self.decay_steps = cfg.DECAY_STEPS
        # 学习衰减因子
        self.decay_rate = cfg.DECAY_RATE
        self.staircase = cfg.STAIRCASE
        self.summary_iter = cfg.SUMMARY_ITER
        # 保存模型的步数
        self.save_iter = cfg.SAVE_ITER
        # update by date
        self.output_dir = os.path.join(
            cfg.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # save model configuration
        self.save_cfg()

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=1)
        # 保存模型的文件名
        self.ckpt_file = os.path.join(self.output_dir, 'yolo')
        self.summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.output_dir, flush_secs=60)

        self.global_step = tf.train.create_global_step()
        # 指数衰减学习率
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate')
        # 定义优化器
        self.optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate)
        # 训练OP
        # default execute some necessary update_ops
        self.train_op = slim.learning.create_train_op(
            self.net.total_loss, self.optimizer, global_step=self.global_step)

        gpu_options = tf.GPUOptions(allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        # default initial all variables
        # using in this way is very convenient
        self.sess.run(tf.global_variables_initializer())

        if self.weights_file is not None:
            print('Restoring weights from: ' + self.weights_file)
            # load partial necessary parameters from ckpt file
            self.saver.restore(self.sess, self.weights_file)

        self.writer.add_graph(self.sess.graph)

    # 训练
    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self.max_iter + 1):

            load_timer.tic()
            # 加载一个batch的训练数据
            images, labels = self.data.get()
            load_timer.toc()
            feed_dict = {self.net.images: images,
                         self.net.labels: labels}

            if step % self.summary_iter == 0:
                
                summary_str, _ = self.sess.run(
                    [self.summary_op,self.train_op],
                    feed_dict=feed_dict)
                self.writer.add_summary(summary_str, step)

            else:
                train_timer.tic()
                summary_str,loss, _ = self.sess.run(
                    [self.summary_op,self.net.total_loss,self.train_op],
                    feed_dict=feed_dict)
                train_timer.toc()
                r = ('\r{} Epoch:{:03d},Step:{:06d},Loss:{:5.3f},Speed:{:.3f}s/iter,'+\
                    'Load:{:.3f}s/iter,Remain:{:10s}').format(
                            datetime.datetime.now().strftime('%m-%d-%H:%M:%S'),
                            self.data.epoch,
                            step,
                            loss,
                            train_timer.average_time,
                            load_timer.average_time,
                            train_timer.remain(step,self.max_iter))
                logging.info(r)
                print(r,end='',flush=True)

            if step % self.save_iter == 0:
                print('{} Saving checkpoint file to: {}'.format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir))
                self.saver.save(
                    self.sess, self.ckpt_file, global_step=self.global_step,
                    write_meta_graph=False)

    def save_cfg(self):

        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = cfg.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'voc2007')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')

    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)


def initLogging():
    log_file = './train_records.log'
    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        filemode='a',
                        format='%(asctime)s-%(levelname)s-%(message)s',
                        datefmt='%Y-%m-%d-%H:%M:%S')
    
    # set standard output stream
#    console = logging.StreamHandler()
#    console.setLevel(logging.INFO)
#    fmt = '%(asctime)s-%(message)s'
#    datefmt = '%m-%d-%H:%M:%S'
#    # console output format
#    console_fmt = logging.Formatter(fmt=fmt,datefmt=datefmt)
#    console.setFormatter(console_fmt)
#    # get root logger
#    logging.getLogger('').addHandler(console)
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('--data_dir', default="data", type=str)
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU

    yolo = YOLONet()
    pascal = pascal_voc('train')

    solver = Solver(yolo, pascal)
    
    initLogging()
    print('Start training ...')
    solver.train()
    print('Done training.')


if __name__ == '__main__':

    # python train.py --weights YOLO_small.ckpt --gpu 0
    main()
