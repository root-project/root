#!/usr/bin/python3

### Helpers shared by the PyTorch model generators used in TestSofieModels.
###
### The generators (Conv1d/Conv2d/Conv3d/ConvTrans2d/Linear/Recurrent) only
### differ in the network they build and in the shape of the input tensor;
### the command line, the ONNX export and the writing of the expected output
### are identical, so they live here. Keeping a single copy avoids the drift
### that repeated copy-pasting had already introduced between them.

import argparse
import sys

import torch


def make_parser(params_help, pooling=False, recurrent=False):
    """Command line shared by all the generators.

    The positional "params" are generator specific and described by params_help.
    """
    parser = argparse.ArgumentParser(description='PyTorch model generator')
    parser.add_argument('params', type=int, nargs='+', help=params_help)
    if recurrent:
        parser.add_argument('--lstm', action='store_true', default=False,
                            help='For using LSTM layer')
        parser.add_argument('--gru', action='store_true', default=False,
                            help='For using GRU layer')
    else:
        parser.add_argument('--bn', action='store_true', default=False,
                            help='For using batch norm layer')
    if pooling:
        parser.add_argument('--maxpool', action='store_true', default=False,
                            help='For using max pool layer')
        parser.add_argument('--avgpool', action='store_true', default=False,
                            help='For using average pool layer')
    parser.add_argument('--v', action='store_true', default=False,
                        help='For verbose mode')
    return parser


def model_name(base, bsize, use_bn=False, use_maxpool=False, use_avgpool=False):
    """The naming convention TestSofieModels expects, e.g. Conv2dModel_BN_B5."""
    name = base
    if use_bn:
        name += "_BN"
    if use_maxpool:
        name += "_MAXP"
    if use_avgpool:
        name += "_AVGP"
    return name + "_B" + str(bsize)


def export_onnx(model, xinput, name, use_bn=False, dynamo=None):
    """Export model to <name>.onnx.

    dynamo=None selects the exporter automatically: the dynamo based one,
    except where it is known not to work. Pass an explicit value to override.
    """
    if dynamo is None:
        # dynamo export doesn't work on Python 3.14
        dynamo = (sys.version_info.major, sys.version_info.minor) != (3, 14)
        # the new ONNX exporter does not work for batch normalization
        if use_bn:
            dynamo = False

    from packaging.version import Version
    if Version(torch.__version__) >= Version("2.5.0"):
        torch.onnx.export(
            model,
            xinput,
            name + ".onnx",
            export_params=True,
            dynamo=dynamo,
            external_data=False,
        )
    else:
        torch.onnx.export(model, xinput, name + ".onnx", export_params=True)


def write_reference_output(model, xinput, name):
    """Evaluate the model and write its flattened output to <name>.out."""
    model.eval()
    y = model.forward(xinput)

    print("output data : shape, ", y.shape)
    print(y)

    yvec = y.reshape([y.nelement()])
    with open(name + ".out", "w") as f:
        for i in range(0, y.nelement()):
            f.write(str(float(yvec[i].detach())) + " ")
