import os

#
# path and dataset parameter
#

DATA_PATH = 'data'

PASCAL_PATH = os.path.join(DATA_PATH, 'voc2007')

CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')

OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')

WEIGHTS_DIR = os.path.join(PASCAL_PATH, 'weights')

# 权重文件名
# WEIGHTS_FILE = None
WEIGHTS_FILE = os.path.join(PASCAL_PATH, 'weights', 'YOLO_small.ckpt')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

# CLASSES = ['rbc']

FLIPPED = True


#
# model parameter
#

# 模型参数
# 图片大小448x448x3
IMAGE_SIZE = 448

# 网格7x7
CELL_SIZE = 7

# 每个网格box数量
BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

# 损失函数惩罚系数
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.5
CLASS_SCALE = 2.0
COORD_SCALE = 5.0


#
# solver parameter
#

GPU = '0'

LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1

STAIRCASE = True

BATCH_SIZE = 8

MAX_ITER = 100000

SUMMARY_ITER = 10

SAVE_ITER = 1000


#
# test parameter
#

THRESHOLD = 0.2 # 0.2

IOU_THRESHOLD = 0.5

def test():
    pass