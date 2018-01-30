from src.nn.models import *
from utils.pytorch_modelsize import *

c0,_ = chinese_model(3000)
c8,_ = chinese_pruned_80(3000)
c9,_ = chinese_pruned_90(3000)

se0 = SizeEstimator(c0)
se8 = SizeEstimator(c8)
se9 = SizeEstimator(c9)

se0.get_parameter_sizes()
se0.calc_param_bits()
print(se0.param_bits)
se8.get_parameter_sizes()
se8.calc_param_bits()
print(se8.param_bits)
se9.get_parameter_sizes()
se9.calc_param_bits()
print(se9.param_bits)
