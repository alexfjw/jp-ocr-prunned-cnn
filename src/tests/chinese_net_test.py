from src.nn.models import ChineseNet
import torch.autograd

model = ChineseNet(3).cuda()
x = torch.autograd.Variable(torch.rand((16, 1, 96, 96)).cuda())
model(x)


