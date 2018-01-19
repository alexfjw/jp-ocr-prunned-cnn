from torch.autograd import Variable
import torch.onnx
from src.nn.models import ChineseNet
import onnx
import onnx_caffe2.backend as backend
from caffe2.python.predictor import mobile_exporter

def pytorch_to_onnx(torch_model, dummy_input, model_name):
    #dummy_input = Variable(torch.rand(1, 1, 96, 96))
    torch_model.cpu()
    torch_model.train(False)
    torch_model.convert_to_onnx = True
    torch.onnx.export(torch_model, dummy_input, f"trained_models/{model_name}.proto", verbose=False, export_params=True)

    # test if it works
    onnx_model = onnx.load(f"trained_models/{model_name}.proto")
    onnx.checker.check_model(onnx_model)
    print(onnx.helper.printable_graph(onnx_model.graph))
    rep = backend.prepare(onnx_model, device="CPU")
    print(rep.run(dummy_input.data.cpu().numpy())[0])

    return rep


def onnx_to_caffe2(onnx_rep, dest_dir, prefix):
    c2_workspace = onnx_rep.workspace
    c2_model = onnx_rep.predict_net

    init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

    with open(f'{dest_dir}/{prefix}init_net.pb', "wb") as fopen:
        fopen.write(init_net.SerializeToString())
    with open(f'{dest_dir}/{prefix}predict_net.pb', "wb") as fopen:
        fopen.write(predict_net.SerializeToString())

    return init_net, predict_net


