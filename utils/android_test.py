from torch.autograd import Variable
import torch.onnx
from src.nn.models import *
from torchvision import models
import onnx
import onnx_caffe2.backend as backend
from caffe2.python.predictor import mobile_exporter
# Some standard imports
from caffe2.proto import caffe2_pb2
from caffe2.python import core, net_drawer, net_printer, visualize, workspace, utils
from utils.pytorch_modelsize import SizeEstimator
from datetime import datetime

#model80, name = chinese_pruned_80(3156)
#model80.load_state_dict(torch.load(f'trained_models/temp/pruned_0.9_250it.weights'))
#rep = pytorch_to_onnx(model80, Variable(torch.FloatTensor(1, 1, 96, 96)), name)
#
#onnx_to_caffe2(rep, "trained_models", name)

#sq = models.squeezenet1_1()
#batch_size = 64
#t1 = datetime.now()
#sq(Variable(torch.FloatTensor(batch_size,3,224,224)))
#t2 = datetime.now()
#
#print("time taken: ", (t2-t1).seconds/batch_size)
