#!/usr/bin/env python3
"""Generator for the ONNX input models of the SOFIE tests.

Each make_<Name>() function below builds one of the models in a
human-readable way with the onnx helper API. Large weight tensors whose
values carry no meaning (e.g. the weights of the Linear_* models) are
seeded-random via _random_tensor().

The script also computes the expected outputs for the value-based unit tests
(see TEST_INPUTS further below) and writes them to references/<Name>.ref in
the output directory, from where TestCustomModelsFromONNX.cxx reads them at
runtime.

This script is run as the SofieGenerateModels_ONNX unit test, which the
other SOFIE ONNX tests depend on (see CMakeLists.txt). To (re)generate the
models and reference files manually:

    python3 generate_input_models.py --outdir <dir> [model names...]

Listing the available model names does not require the onnx package:

    python3 generate_input_models.py --list
"""

import argparse
import os
import sys

try:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator
except ImportError:
    onnx = None

inf = float("inf")
nan = float("nan")


def _vi(name, dtype, shape):
    """Shorthand for a tensor value info (graph/node input or output)."""
    return helper.make_tensor_value_info(name, dtype, shape)


def _tensor(name, dtype, dims, vals):
    """Shorthand for a constant tensor with explicit values."""
    return helper.make_tensor(name, dtype, dims, vals)


def _random_tensor(name, dims, seed):
    """Uniform random float32 weight tensor in [-k, k] with k = 1/sqrt(fan_in),
    mimicking the default pytorch initialization that the original models were
    exported with. The fixed per-tensor seed keeps the generated model - and
    with it the reference outputs computed further below - reproducible."""
    rng = np.random.RandomState(seed)
    k = 1.0 / np.sqrt(dims[-1])
    vals = rng.uniform(-k, k, int(np.prod(dims))).astype(np.float32)
    return helper.make_tensor(name, TensorProto.FLOAT, dims, vals)


def _model(graph, opset, ir_version, **kwargs):
    """Wrap a graph into a ModelProto with the given opset and IR version."""
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)], **kwargs)
    model.ir_version = ir_version
    return model


if onnx is not None:
    BOOL = TensorProto.BOOL
    DOUBLE = TensorProto.DOUBLE
    FLOAT = TensorProto.FLOAT
    INT64 = TensorProto.INT64
    UINT8 = TensorProto.UINT8


def make_Abs():
    """Ops: Abs"""
    nodes = [
        helper.make_node('Abs', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'Abs',
        inputs=[
            _vi('input', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Add():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['onnx::Add_0', 'onnx::Add_1'], ['2'], name='Add_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Add_0', FLOAT, [2]),
            _vi('onnx::Add_1', FLOAT, [2]),
        ],
        outputs=[
            _vi('2', FLOAT, [2]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_AddBroadcast1():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [5]),
            _vi('B', FLOAT, [4, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [4, 5]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast2():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [5]),
            _vi('B', FLOAT, [2, 3, 4, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4, 5]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast3():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [2, 1, 1, 5]),
            _vi('B', FLOAT, [2, 3, 4, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4, 5]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast4():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [2, 1]),
            _vi('B', FLOAT, [2, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast5():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [2, 1, 4]),
            _vi('B', FLOAT, [2, 3, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast6():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [2, 1, 3, 1, 2]),
            _vi('B', FLOAT, [2, 2, 3, 2, 2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 2, 3, 2, 2]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AddBroadcast7():
    """Ops: Add"""
    nodes = [
        helper.make_node('Add', ['A', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Add',
        inputs=[
            _vi('A', FLOAT, [2, 1, 3, 1]),
            _vi('B', FLOAT, [1, 1, 3, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 1, 3, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_AvgPool():
    """Ops: AveragePool"""
    nodes = [
        helper.make_node(
            'AveragePool',
            ['onnx::Pad_0'],
            ['2'],
            name='AveragePool_1',
            kernel_shape=[3, 2],
            pads=[0, 0, 0, 0],
            strides=[2, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Pad_0', FLOAT, [1, 1, 5, 10]),
        ],
        outputs=[
            _vi('2', FLOAT, [1, 1, 2, 9]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Cast():
    """Ops: Cast"""
    nodes = [
        helper.make_node('Cast', ['onnx::Cast_0'], ['1'], name='Cast_0', to=11),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Cast_0', INT64, [2, 3]),
        ],
        outputs=[
            _vi('1', DOUBLE, [2, 3]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Clip():
    """Ops: Clip"""
    nodes = [
        helper.make_node('Clip', ['X', 'min', 'max'], ['Y'], name='clip_node'),
        helper.make_node('Clip', ['X', 'min'], ['Y2'], name='clip_node'),
    ]
    graph = helper.make_graph(
        nodes,
        'ClipGraph',
        inputs=[
            _vi('X', FLOAT, ['N', 2, 2]),
        ],
        outputs=[
            _vi('Y', FLOAT, ['N', 2, 2]),
            _vi('Y2', FLOAT, ['N', 2, 2]),
        ],
        initializer=[
            _tensor('min', FLOAT, [], [-1.0]),
            _tensor('max', FLOAT, [], [1.0]),
        ],
    )
    return _model(graph, opset=13, ir_version=8, producer_name='onnx-example')


def make_Comparison_broadcast():
    """Ops: Greater, Equal, Less"""
    nodes = [
        helper.make_node('Greater', ['A', 'B'], ['OutGreater']),
        helper.make_node('Equal', ['A', 'B'], ['OutEqual']),
        helper.make_node('Less', ['A', 'B'], ['OutLess']),
    ]
    graph = helper.make_graph(
        nodes,
        'ComparisonOpsWithBroadcast',
        inputs=[
            _vi('A', FLOAT, [1, 4]),
            _vi('B', FLOAT, [4]),
        ],
        outputs=[
            _vi('OutGreater', BOOL, [1, 4]),
            _vi('OutEqual', BOOL, [1, 4]),
            _vi('OutLess', BOOL, [1, 4]),
        ],
    )
    return _model(graph, opset=23, ir_version=11, producer_name='comparison_broadcast_demo')


def make_Comparison_broadcast_3d():
    """Ops: Greater, Equal, Less"""
    nodes = [
        helper.make_node('Greater', ['A', 'B'], ['OutGreater']),
        helper.make_node('Equal', ['A', 'B'], ['OutEqual']),
        helper.make_node('Less', ['A', 'B'], ['OutLess']),
    ]
    graph = helper.make_graph(
        nodes,
        'ComparisonOpsBroadcast',
        inputs=[
            _vi('A', FLOAT, [2, 2, 4]),
            _vi('B', FLOAT, [1, 4]),
        ],
        outputs=[
            _vi('OutGreater', BOOL, [2, 2, 4]),
            _vi('OutEqual', BOOL, [2, 2, 4]),
            _vi('OutLess', BOOL, [2, 2, 4]),
        ],
    )
    return _model(graph, opset=23, ir_version=11, producer_name='comparison_broadcast_demo')


def make_ComplexTopK():
    """Ops: Constant, TopK"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['/Constant_output_0'],
            name='/Constant',
            value=_tensor('', INT64, [1], [3]),
        ),
        helper.make_node(
            'TopK',
            ['onnx::TopK_0', '/Constant_output_0'],
            ['4', '5'],
            name='/TopK',
            axis=1,
            largest=1,
            sorted=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'main_graph',
        inputs=[
            _vi('onnx::TopK_0', FLOAT, [2, 3, 9]),
        ],
        outputs=[
            _vi('4', FLOAT, [2, 3, 9]),
            _vi('5', INT64, [2, 3, 9]),
        ],
    )
    return _model(graph, opset=17, ir_version=8, producer_name='pytorch', producer_version='2.3.0')


def make_Concat_0D():
    """Ops: Concat"""
    nodes = [
        helper.make_node(
            'Concat',
            ['onnx::Concat_0', 'onnx::Concat_0'],
            ['1'],
            name='Concat_0',
            axis=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Concat_0', FLOAT, [2]),
        ],
        outputs=[
            _vi('1', FLOAT, [4]),
        ],
    )
    return _model(graph, opset=13, ir_version=7, producer_name='pytorch', producer_version='1.12.1')


def make_Constant():
    """Ops: Constant, Add"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['constant_output'],
            value=_tensor('constant_tensor', FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0]),
        ),
        helper.make_node('Add', ['constant_output', 'constant_output'], ['add_output']),
    ]
    graph = helper.make_graph(
        nodes,
        'constant_addition_graph',
        inputs=[
        ],
        outputs=[
            _vi('add_output', FLOAT, [2, 2]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='onnx_constant_model')


def make_ConvAddRelu():
    """Ops: Conv, Add, Relu"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'w'],
            ['conv_out'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
        helper.make_node('Add', ['conv_out', 'b'], ['add_out']),
        helper.make_node('Relu', ['add_out'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvAddRelu',
        inputs=[
            _vi('x', FLOAT, [1, 1, 4, 4]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 2, 2]),
        ],
        initializer=[
            _tensor('w', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            _tensor('b', FLOAT, [1], [0.5]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_ConvTranspose1d():
    """Ops: ConvTranspose"""
    nodes = [
        helper.make_node('ConvTranspose', ['X', 'W'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvTranspose1d',
        inputs=[
            _vi('X', FLOAT, [1, 1, 3]),
            _vi('W', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 2, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 2, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ConvTranspose2d():
    """Ops: ConvTranspose"""
    nodes = [
        helper.make_node('ConvTranspose', ['X', 'W'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvTranspose2d',
        inputs=[
            _vi('X', FLOAT, [1, 1, 3, 3]),
            _vi('W', FLOAT, [1, 2, 3, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 2, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 2, 3, 3], [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ConvTransposeBias2d():
    """Ops: ConvTranspose"""
    nodes = [
        helper.make_node('ConvTranspose', ['X', 'W', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvTranspose2d',
        inputs=[
            _vi('X', FLOAT, [1, 1, 3, 3]),
            _vi('W', FLOAT, [1, 2, 3, 3]),
            _vi('B', FLOAT, [2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 2, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 2, 3, 3], [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ]),
            _tensor('B', FLOAT, [2], [1.0, 2.0]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ConvTransposeBias2dBatched():
    """Ops: ConvTranspose"""
    nodes = [
        helper.make_node('ConvTranspose', ['X', 'W', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvTranspose2d',
        inputs=[
            _vi('X', FLOAT, [2, 1, 3, 3]),
            _vi('W', FLOAT, [1, 2, 3, 3]),
            _vi('B', FLOAT, [2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 2, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 2, 3, 3], [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0,
            ]),
            _tensor('B', FLOAT, [2], [1.0, 2.0]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ConvWithAsymmetricPadding():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'W'],
            ['y'],
            kernel_shape=[3, 3],
            pads=[1, 0, 1, 0],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithAsymmetricPadding',
        inputs=[
            _vi('x', FLOAT, [1, 1, 7, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_ConvWithAutopadSameLower():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'W'],
            ['y'],
            auto_pad='SAME_LOWER',
            kernel_shape=[3, 3],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithAutopadSameLower',
        inputs=[
            _vi('x', FLOAT, [1, 1, 5, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_ConvWithAutopadSameUpper():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'W'],
            ['y'],
            auto_pad='SAME_UPPER',
            kernel_shape=[3, 3],
            strides=[1, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvSameUpper',
        inputs=[
            _vi('x', FLOAT, [1, 1, 5, 5]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=11, ir_version=13)


def make_ConvWithDilation():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['input', 'W'],
            ['output'],
            dilations=[2, 2],
            group=1,
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithDilation',
        inputs=[
            _vi('input', FLOAT, [1, 1, 7, 7]),
        ],
        outputs=[
            _vi('output', FLOAT, [1, 1, 3, 3]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [
                0.10000000149011612, 0.20000000298023224, 0.30000001192092896, 0.4000000059604645,
                0.5, 0.6000000238418579, 0.699999988079071, 0.800000011920929, 0.9000000357627869,
            ]),
        ],
    )
    return _model(graph, opset=13, ir_version=8)


def make_ConvWithDynShapeStride():
    """Ops: Conv"""
    nodes = [
        helper.make_node('Conv', ['X', 'W'], ['Y'], kernel_shape=[3], pads=[0, 0], strides=[2]),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithDynShapeStride',
        inputs=[
            _vi('X', FLOAT, [1, 1, 'W']),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 'out_W']),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3], [1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_ConvWithPadding():
    """Ops: Conv"""
    nodes = [
        helper.make_node('Conv', ['x', 'W'], ['y'], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithPadding',
        inputs=[
            _vi('x', FLOAT, [1, 1, 5, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_ConvWithStridesNoPadding():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'W'],
            ['y'],
            kernel_shape=[3, 3],
            pads=[0, 0, 0, 0],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithStridesNoPadding',
        inputs=[
            _vi('x', FLOAT, [1, 1, 7, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_ConvWithStridesPadding():
    """Ops: Conv"""
    nodes = [
        helper.make_node(
            'Conv',
            ['x', 'W'],
            ['y'],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithStridesPadding',
        inputs=[
            _vi('x', FLOAT, [1, 1, 7, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_ConvWithoutPadding():
    """Ops: Conv"""
    nodes = [
        helper.make_node('Conv', ['x', 'W'], ['y'], kernel_shape=[3, 3], pads=[0, 0, 0, 0]),
    ]
    graph = helper.make_graph(
        nodes,
        'ConvWithoutPadding',
        inputs=[
            _vi('x', FLOAT, [1, 1, 5, 5]),
            _vi('W', FLOAT, [1, 1, 3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [1, 1, 5, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 1, 3, 3], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='python_script')


def make_Cos():
    """Ops: Cos"""
    nodes = [
        helper.make_node('Cos', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'CosGraph',
        inputs=[
            _vi('input', FLOAT, [3, 4]),
        ],
        outputs=[
            _vi('output', FLOAT, [3, 4]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='cos_example')


def make_Div():
    """Ops: Div"""
    nodes = [
        helper.make_node('Div', ['onnx::Div_0', 'onnx::Div_1'], ['2'], name='Div_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Div_0', FLOAT, [2]),
            _vi('onnx::Div_1', FLOAT, [2]),
        ],
        outputs=[
            _vi('2', FLOAT, [2]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Einsum_3():
    """Ops: Einsum"""
    nodes = [
        helper.make_node('Einsum', ['inputA', 'inputB'], ['output'], equation='abc,abd->ad'),
    ]
    graph = helper.make_graph(
        nodes,
        'EinsumGraph',
        inputs=[
            _vi('inputA', FLOAT, [2, 2, 3]),
            _vi('inputB', FLOAT, [2, 2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Einsum_4():
    """Ops: Einsum"""
    nodes = [
        helper.make_node('Einsum', ['inputA', 'inputB'], ['output'], equation='abcd,abed->abce'),
    ]
    graph = helper.make_graph(
        nodes,
        'EinsumGraph',
        inputs=[
            _vi('inputA', FLOAT, [2, 1, 2, 3]),
            _vi('inputB', FLOAT, [2, 1, 3, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 1, 2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Einsum_dotprod():
    """Ops: Einsum"""
    nodes = [
        helper.make_node('Einsum', ['inputA', 'inputB'], ['output'], equation='i,i->'),
    ]
    graph = helper.make_graph(
        nodes,
        'EinsumGraph',
        inputs=[
            _vi('inputA', FLOAT, [3]),
            _vi('inputB', FLOAT, [3]),
        ],
        outputs=[
            _vi('output', FLOAT, [0]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Einsum_matmul():
    """Ops: Einsum"""
    nodes = [
        helper.make_node('Einsum', ['inputA', 'inputB'], ['output'], equation='ik,kj->ij'),
    ]
    graph = helper.make_graph(
        nodes,
        'EinsumGraph',
        inputs=[
            _vi('inputA', FLOAT, [2, 2]),
            _vi('inputB', FLOAT, [2, 2]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Elu():
    """Ops: Elu"""
    nodes = [
        helper.make_node('Elu', ['input'], ['output'], name='/elu/Elu', alpha=1.0),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('input', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='2.0.1')


def make_EluAlpha():
    """Ops: Elu"""
    nodes = [
        helper.make_node('Elu', ['input'], ['output'], name='/elu/Elu', alpha=0.5),
    ]
    graph = helper.make_graph(
        nodes,
        'EluAlpha',
        inputs=[
            _vi('input', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=11, ir_version=13)


def make_Equal():
    """Ops: Equal"""
    nodes = [
        helper.make_node('Equal', ['onnx::Equal_0', 'onnx::Equal_1'], ['2'], name='/Equal'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Equal_0', FLOAT, [3]),
            _vi('onnx::Equal_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', BOOL, [3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='1.13.1')


def make_Erf():
    """Ops: Erf"""
    nodes = [
        helper.make_node('Erf', ['onnx::Erf_0'], ['1'], name='/Erf'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Erf_0', FLOAT, [12]),
        ],
        outputs=[
            _vi('1', FLOAT, [12]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='1.13.1')


def make_Exp():
    """Ops: Exp"""
    nodes = [
        helper.make_node('Exp', ['X'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Exp',
        inputs=[
            _vi('X', FLOAT, [10]),
        ],
        outputs=[
            _vi('Y', FLOAT, [10]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ExpandDiffSize():
    """Ops: Expand"""
    nodes = [
        helper.make_node('Expand', ['X', 'Shape'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Expand',
        inputs=[
            _vi('X', FLOAT, [3, 1]),
            _vi('Shape', INT64, [4]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
        initializer=[
            _tensor('Shape', INT64, [4], [3, 2, 1, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ExpandSameSize():
    """Ops: Expand"""
    nodes = [
        helper.make_node('Expand', ['X', 'Shape'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Expand',
        inputs=[
            _vi('X', FLOAT, [3, 1]),
            _vi('Shape', INT64, [2]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
        initializer=[
            _tensor('Shape', INT64, [2], [3, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_EyeLike():
    """Ops: EyeLike"""
    nodes = [
        helper.make_node('EyeLike', ['x'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'eyelike_model',
        inputs=[
            _vi('x', FLOAT, [3, 3]),
        ],
        outputs=[
            _vi('y', FLOAT, [3, 3]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='EyeLikeModel')


def make_FMod_ConstantFolding():
    """Ops: Constant, Mod"""
    nodes = [
        helper.make_node('Constant', [], ['X'], value=_tensor('X', FLOAT, [3], [10.0, 7.0, 5.0])),
        helper.make_node('Constant', [], ['D'], value=_tensor('D', FLOAT, [3], [3.0, 3.0, 3.0])),
        helper.make_node('Mod', ['X', 'D'], ['Y'], fmod=1),
    ]
    graph = helper.make_graph(
        nodes,
        'FMod_ConstantFolding',
        inputs=[
        ],
        outputs=[
            _vi('Y', FLOAT, [3]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_GRUBatchwise():
    """Ops: GRU"""
    nodes = [
        helper.make_node(
            'GRU',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=6,
            layout=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'GRUBatchwise',
        inputs=[
            _vi('X', FLOAT, [3, 1, 2]),
            _vi('W', FLOAT, [1, 18, 2]),
            _vi('R', FLOAT, [1, 18, 6]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 6]),
            _vi('Y_h', FLOAT, [3, 1, 6]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 18, 2], [
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
            ]),
            _random_tensor('R', [1, 18, 6], seed=101),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_GRUBidirectional():
    """Ops: GRU"""
    nodes = [
        helper.make_node(
            'GRU',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh', 'Sigmoid', 'Tanh'],
            clip=0.0,
            direction='bidirectional',
            hidden_size=5,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'GRUBidirectional',
        inputs=[
            _vi('X', FLOAT, [1, 3, 2]),
            _vi('W', FLOAT, [2, 15, 2]),
            _vi('R', FLOAT, [2, 15, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 2, 3, 5]),
            _vi('Y_h', FLOAT, [2, 3, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [2, 15, 2], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582,
            ]),
            _random_tensor('R', [2, 15, 5], seed=102),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_GRUDefaults():
    """Ops: GRU"""
    nodes = [
        helper.make_node(
            'GRU',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=5,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'GRUDefaults',
        inputs=[
            _vi('X', FLOAT, [1, 3, 2]),
            _vi('W', FLOAT, [1, 15, 2]),
            _vi('R', FLOAT, [1, 15, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 3, 5]),
            _vi('Y_h', FLOAT, [1, 3, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 15, 2], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612,
            ]),
            _random_tensor('R', [1, 15, 5], seed=103),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_GRUInitialBias():
    """Ops: GRU"""
    nodes = [
        helper.make_node(
            'GRU',
            ['X', 'W', 'R', 'B'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=3,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'GRUInitialBias',
        inputs=[
            _vi('X', FLOAT, [1, 3, 3]),
            _vi('W', FLOAT, [1, 9, 3]),
            _vi('R', FLOAT, [1, 9, 3]),
            _vi('B', FLOAT, [1, 18]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 3, 3]),
            _vi('Y_h', FLOAT, [1, 3, 3]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 9, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('R', FLOAT, [1, 9, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('B', FLOAT, [1, 18], [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_GRUSeqLength():
    """Ops: GRU"""
    nodes = [
        helper.make_node(
            'GRU',
            ['X', 'W', 'R', 'B'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=5,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'GRUSeqLength',
        inputs=[
            _vi('X', FLOAT, [2, 3, 3]),
            _vi('W', FLOAT, [1, 15, 3]),
            _vi('R', FLOAT, [1, 15, 5]),
            _vi('B', FLOAT, [1, 30]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 1, 3, 5]),
            _vi('Y_h', FLOAT, [1, 3, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 15, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612,
            ]),
            _random_tensor('R', [1, 15, 5], seed=104),
            _tensor('B', FLOAT, [1, 30], [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_Gather2d():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [3, 3]),
            _vi('I', INT64, [3, 2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 2, 3]),
        ],
        initializer=[
            _tensor('I', INT64, [3, 2], [0, 2, 0, 1, 2, 2]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_GatherAxis0():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [5, 4, 3, 2]),
            _vi('I', INT64, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 4, 3, 2]),
        ],
        initializer=[
            _tensor('I', INT64, [3], [0, 1, 3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_GatherAxis1():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [5, 4, 3, 2]),
            _vi('I', INT64, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [5, 3, 3, 2]),
        ],
        initializer=[
            _tensor('I', INT64, [3], [0, 1, 3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_GatherAxis2():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=2),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [5, 4, 3, 2]),
            _vi('I', INT64, [2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [5, 4, 2, 2]),
        ],
        initializer=[
            _tensor('I', INT64, [2], [1, 2]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_GatherAxis3():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=3),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [5, 4, 3, 2]),
            _vi('I', INT64, [1]),
        ],
        outputs=[
            _vi('Y', FLOAT, [5, 4, 3, 1]),
        ],
        initializer=[
            _tensor('I', INT64, [1], [1]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_GatherND_1():
    """Ops: GatherND"""
    nodes = [
        helper.make_node('GatherND', ['data', 'indices'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [2, 3, 3]),
            _vi('indices', INT64, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_GatherND_2():
    """Ops: GatherND"""
    nodes = [
        helper.make_node('GatherND', ['data', 'indices'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [2, 3, 3]),
            _vi('indices', INT64, [2, 2]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_GatherND_3():
    """Ops: GatherND"""
    nodes = [
        helper.make_node('GatherND', ['data', 'indices'], ['output'], batch_dims=1),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [2, 3, 2, 2]),
            _vi('indices', INT64, [2, 2, 1]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 2, 2, 2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_GatherNegativeIndices():
    """Ops: Gather"""
    nodes = [
        helper.make_node('Gather', ['X', 'I'], ['Y'], axis=0),
    ]
    graph = helper.make_graph(
        nodes,
        'Gather',
        inputs=[
            _vi('X', FLOAT, [10]),
            _vi('I', INT64, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3]),
        ],
        initializer=[
            _tensor('I', INT64, [3], [0, -9, -10]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Gelu():
    """Ops: Gelu"""
    nodes = [
        helper.make_node('Gelu', ['x'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Gelu',
        inputs=[
            _vi('x', FLOAT, [6]),
        ],
        outputs=[
            _vi('y', FLOAT, [6]),
        ],
    )
    return _model(graph, opset=20, ir_version=13)


def make_Greater():
    """Ops: Greater"""
    nodes = [
        helper.make_node(
            'Greater',
            ['onnx::Greater_0', 'onnx::Greater_1'],
            ['2'],
            name='/Greater',
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Greater_0', FLOAT, [3]),
            _vi('onnx::Greater_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', BOOL, [3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='2.0.1')


def make_GreaterOrEqual():
    """Ops: GreaterOrEqual"""
    nodes = [
        helper.make_node(
            'GreaterOrEqual',
            ['onnx::GreaterOrEqual_0', 'onnx::GreaterOrEqual_1'],
            ['2'],
            name='/GreaterOrEqual',
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::GreaterOrEqual_0', FLOAT, [3]),
            _vi('onnx::GreaterOrEqual_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', BOOL, [3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='2.0.1')


def make_HardSigmoid():
    """Ops: HardSigmoid"""
    nodes = [
        helper.make_node(
            'HardSigmoid',
            ['input'],
            ['output'],
            alpha=0.20000000298023224,
            beta=0.5,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'HardSigmoidGraph',
        inputs=[
            _vi('input', FLOAT, [6]),
        ],
        outputs=[
            _vi('output', FLOAT, [6]),
        ],
    )
    return _model(graph, opset=6, ir_version=13)


def make_HardSwish():
    """Ops: HardSwish"""
    nodes = [
        helper.make_node('HardSwish', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'HardSwishGraph',
        inputs=[
            _vi('input', FLOAT, [6]),
        ],
        outputs=[
            _vi('output', FLOAT, [6]),
        ],
    )
    return _model(graph, opset=14, ir_version=13)


def _instance_normalization(shape, scale, bias, **kwargs):
    """InstanceNormalization graph over an input of the given shape. scale and
    B are per-channel initializers of length shape[1]."""
    channels = shape[1]
    nodes = [
        helper.make_node('InstanceNormalization', ['X', 'scale', 'B'], ['Y'], **kwargs),
    ]
    graph = helper.make_graph(
        nodes,
        'InstanceNormalization',
        inputs=[
            _vi('X', FLOAT, shape),
            _vi('scale', FLOAT, [channels]),
            _vi('B', FLOAT, [channels]),
        ],
        outputs=[
            _vi('Y', FLOAT, shape),
        ],
        initializer=[
            _tensor('scale', FLOAT, [channels], scale),
            _tensor('B', FLOAT, [channels], bias),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_InstanceNormalization():
    """Ops: InstanceNormalization"""
    # Batch and channel size both > 1, so that a wrong per-instance offset or a
    # scale/bias indexed by the wrong axis does not go unnoticed.
    return _instance_normalization([2, 3, 4, 5],
                                   scale=[0.5, 1.0, -2.0],
                                   bias=[0.0, 0.25, -1.5])


def make_InstanceNormalization3d():
    """Ops: InstanceNormalization"""
    # Rank 3 (N, C, D): exercises a spatial size that is not a product of
    # several dimensions.
    return _instance_normalization([2, 2, 6],
                                   scale=[1.5, -0.5],
                                   bias=[1.0, 2.0])


def make_InstanceNormalizationEpsilon():
    """Ops: InstanceNormalization"""
    # Non-default epsilon, combined with the small-variance input in
    # TEST_INPUTS, so that the attribute dominates the result and a parser that
    # ignored it would be caught.
    return _instance_normalization([1, 2, 3, 3],
                                   scale=[1.0, 1.0],
                                   bias=[0.0, 0.0],
                                   epsilon=0.01)


def make_IsInf():
    """Ops: IsInf"""
    nodes = [
        helper.make_node('IsInf', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'Test',
        inputs=[
            _vi('input', FLOAT, [1, 'N']),
        ],
        outputs=[
            _vi('output', BOOL, [1, 'N']),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_LSTMBatchwise():
    """Ops: LSTM"""
    nodes = [
        helper.make_node(
            'LSTM',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=7,
            layout=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'LSTMBatchwise',
        inputs=[
            _vi('X', FLOAT, [3, 1, 2]),
            _vi('W', FLOAT, [1, 28, 2]),
            _vi('R', FLOAT, [1, 28, 7]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 7]),
            _vi('Y_h', FLOAT, [3, 1, 7]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 28, 2], [
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
                0.30000001192092896, 0.30000001192092896, 0.30000001192092896, 0.30000001192092896,
            ]),
            _random_tensor('R', [1, 28, 7], seed=105),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_LSTMBidirectional():
    """Ops: LSTM"""
    nodes = [
        helper.make_node(
            'LSTM',
            ['X', 'W', 'R'],
            ['Y', 'Y_h', 'Y_c'],
            activations=['Sigmoid', 'Tanh', 'Tanh', 'Sigmoid', 'Tanh', 'Tanh'],
            clip=0.0,
            direction='bidirectional',
            hidden_size=3,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'LSTMBidirectional',
        inputs=[
            _vi('X', FLOAT, [3, 1, 2]),
            _vi('W', FLOAT, [2, 12, 2]),
            _vi('R', FLOAT, [2, 12, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 2, 1, 3]),
            _vi('Y_h', FLOAT, [2, 1, 3]),
            _vi('Y_c', FLOAT, [2, 1, 3]),
        ],
        initializer=[
            _tensor('W', FLOAT, [2, 12, 2], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _random_tensor('R', [2, 12, 3], seed=106),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_LSTMDefaults():
    """Ops: LSTM"""
    nodes = [
        helper.make_node(
            'LSTM',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=3,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'LSTMDefaults',
        inputs=[
            _vi('X', FLOAT, [3, 1, 2]),
            _vi('W', FLOAT, [1, 12, 2]),
            _vi('R', FLOAT, [1, 12, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 3]),
            _vi('Y_h', FLOAT, [1, 1, 3]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 12, 2], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('R', FLOAT, [1, 12, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_LSTMInitialBias():
    """Ops: LSTM"""
    nodes = [
        helper.make_node(
            'LSTM',
            ['X', 'W', 'R', 'B'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=4,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'LSTMInitialBias',
        inputs=[
            _vi('X', FLOAT, [3, 1, 3]),
            _vi('W', FLOAT, [1, 16, 3]),
            _vi('R', FLOAT, [1, 16, 4]),
            _vi('B', FLOAT, [1, 32]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 4]),
            _vi('Y_h', FLOAT, [1, 1, 4]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 16, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('R', FLOAT, [1, 16, 4], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('B', FLOAT, [1, 32], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_LSTMPeepholes():
    """Ops: LSTM"""
    nodes = [
        helper.make_node(
            'LSTM',
            ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
            ['Y', 'Y_h'],
            activations=['Sigmoid', 'Tanh', 'Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=3,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'LSTMPeepholes',
        inputs=[
            _vi('X', FLOAT, [1, 2, 4]),
            _vi('W', FLOAT, [1, 12, 4]),
            _vi('R', FLOAT, [1, 12, 3]),
            _vi('B', FLOAT, [1, 24]),
            _vi('sequence_lens', FLOAT, [2]),
            _vi('initial_h', FLOAT, [1, 2, 3]),
            _vi('initial_c', FLOAT, [1, 2, 3]),
            _vi('P', FLOAT, [1, 9]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 2, 3]),
            _vi('Y_h', FLOAT, [1, 2, 3]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 12, 4], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('R', FLOAT, [1, 12, 3], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('B', FLOAT, [1, 24], [
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
            _tensor('sequence_lens', FLOAT, [2], [1.0, 1.0]),
            _tensor('initial_h', FLOAT, [1, 2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _tensor('initial_c', FLOAT, [1, 2, 3], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _tensor('P', FLOAT, [1, 9], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_LayerNormalization2d():
    """Ops: LayerNormalization"""
    nodes = [
        helper.make_node('LayerNormalization', ['X', 'Scale', 'B'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'LayerNormalization',
        inputs=[
            _vi('X', FLOAT, [3, 4]),
            _vi('Scale', FLOAT, [4]),
            _vi('B', FLOAT, [4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 4]),
        ],
        initializer=[
            _tensor('Scale', FLOAT, [4], [0.5, -0.20000000298023224, 0.30000001192092896, 1.0]),
            _tensor('B', FLOAT, [4], [0.20000000298023224, -0.10000000149011612, 0.10000000149011612, 0.0]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_LayerNormalization4d():
    """Ops: LayerNormalization"""
    nodes = [
        helper.make_node('LayerNormalization', ['X', 'Scale', 'B'], ['Y'], axis=2),
    ]
    graph = helper.make_graph(
        nodes,
        'LayerNormalization',
        inputs=[
            _vi('X', FLOAT, [2, 3, 4, 5]),
            _vi('Scale', FLOAT, [4, 5]),
            _vi('B', FLOAT, [4, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4, 5]),
        ],
        initializer=[
            _tensor('Scale', FLOAT, [4, 5], [
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
                0.10000000149011612, 0.10000000149011612, 0.10000000149011612, 0.10000000149011612,
            ]),
            _tensor('B', FLOAT, [4, 5], [
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
                0.20000000298023224, 0.20000000298023224, 0.20000000298023224, 0.20000000298023224,
            ]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Less():
    """Ops: Less"""
    nodes = [
        helper.make_node('Less', ['onnx::Less_0', 'onnx::Less_1'], ['2'], name='/Less'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Less_0', FLOAT, [3]),
            _vi('onnx::Less_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', BOOL, [3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='2.0.1')


def make_LessOrEqual():
    """Ops: LessOrEqual"""
    nodes = [
        helper.make_node(
            'LessOrEqual',
            ['onnx::LessOrEqual_0', 'onnx::LessOrEqual_1'],
            ['2'],
            name='/LessOrEqual',
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::LessOrEqual_0', FLOAT, [3]),
            _vi('onnx::LessOrEqual_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', BOOL, [3]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='2.0.1')


def make_LinearWithLeakyRelu():
    """Ops: LeakyRelu"""
    nodes = [
        helper.make_node(
            'LeakyRelu',
            ['input'],
            ['1'],
            name='LeakyRelu_0',
            alpha=0.10000000149011612,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input', FLOAT, [24]),
        ],
        outputs=[
            _vi('1', FLOAT, [24]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_LinearWithSelu():
    """Ops: Gemm, Selu, Gemm, Selu"""
    nodes = [
        helper.make_node(
            'Gemm',
            ['input.1', '0.weight', '0.bias'],
            ['5'],
            name='Gemm_0',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Selu', ['5'], ['6'], name='Selu_1'),
        helper.make_node(
            'Gemm',
            ['6', '2.weight', '2.bias'],
            ['7'],
            name='Gemm_2',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Selu', ['7'], ['8'], name='Selu_3'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input.1', FLOAT, [2, 24]),
        ],
        outputs=[
            _vi('8', FLOAT, [2, 12]),
        ],
        initializer=[
            _random_tensor('0.weight', [8, 24], seed=107),
            _tensor('0.bias', FLOAT, [8], [
                -0.02950236387550831, 0.1598697304725647, 0.0748923048377037, -0.07833899557590485,
                -0.1760983020067215, 0.230862095952034, 0.1015596091747284, -0.049189355224370956,
            ]),
            _random_tensor('2.weight', [12, 8], seed=108),
            _tensor('2.bias', FLOAT, [12], [
                -0.17583288252353668, 0.30308830738067627, -0.027203908190131187,
                0.037800826132297516, 0.08170158416032791, 0.3773317337036133,
                -0.17529195547103882, -0.37161365151405334, -0.1841122955083847,
                0.22103063762187958, -0.10950803756713867, -0.10128439217805862,
            ]),
        ],
    )
    return _model(graph, opset=9, ir_version=6, producer_name='pytorch', producer_version='1.9')


def make_LinearWithSigmoid():
    """Ops: Gemm, Sigmoid, Gemm, Sigmoid"""
    nodes = [
        helper.make_node(
            'Gemm',
            ['input.1', '0.weight', '0.bias'],
            ['5'],
            name='Gemm_0',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Sigmoid', ['5'], ['6'], name='Sigmoid_1'),
        helper.make_node(
            'Gemm',
            ['6', '2.weight', '2.bias'],
            ['7'],
            name='Gemm_2',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Sigmoid', ['7'], ['8'], name='Sigmoid_3'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input.1', FLOAT, [2, 24]),
        ],
        outputs=[
            _vi('8', FLOAT, [2, 12]),
        ],
        initializer=[
            _random_tensor('0.weight', [8, 24], seed=109),
            _tensor('0.bias', FLOAT, [8], [
                0.1170196384191513, -0.05893150717020035, 0.11833080649375916, 0.1336972713470459,
                -0.08772247284650803, 0.16479122638702393, 0.05615559592843056, 0.059780675917863846,
            ]),
            _random_tensor('2.weight', [12, 8], seed=110),
            _tensor('2.bias', FLOAT, [12], [
                0.06870245188474655, -0.015492145903408527, 0.46931853890419006,
                -0.4815196692943573, -0.23028719425201416, -0.24661526083946228,
                -0.22366689145565033, -0.6485911011695862, -0.011641554534435272,
                -0.8092096447944641, -0.737714409828186, -0.17296408116817474,
            ]),
        ],
    )
    return _model(graph, opset=9, ir_version=6, producer_name='pytorch', producer_version='1.9')


def make_Linear_16():
    """Ops: Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm"""
    nodes = [
        helper.make_node(
            'Gemm',
            ['input.1', '0.weight', '0.bias'],
            ['21'],
            name='Gemm_0',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['21'], ['22'], name='Relu_1'),
        helper.make_node(
            'Gemm',
            ['22', '2.weight', '2.bias'],
            ['23'],
            name='Gemm_2',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['23'], ['24'], name='Relu_3'),
        helper.make_node(
            'Gemm',
            ['24', '4.weight', '4.bias'],
            ['25'],
            name='Gemm_4',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['25'], ['26'], name='Relu_5'),
        helper.make_node(
            'Gemm',
            ['26', '6.weight', '6.bias'],
            ['27'],
            name='Gemm_6',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['27'], ['28'], name='Relu_7'),
        helper.make_node(
            'Gemm',
            ['28', '8.weight', '8.bias'],
            ['29'],
            name='Gemm_8',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['29'], ['30'], name='Relu_9'),
        helper.make_node(
            'Gemm',
            ['30', '10.weight', '10.bias'],
            ['31'],
            name='Gemm_10',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['31'], ['32'], name='Relu_11'),
        helper.make_node(
            'Gemm',
            ['32', '12.weight', '12.bias'],
            ['33'],
            name='Gemm_12',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['33'], ['34'], name='Relu_13'),
        helper.make_node(
            'Gemm',
            ['34', '14.weight', '14.bias'],
            ['35'],
            name='Gemm_14',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['35'], ['36'], name='Relu_15'),
        helper.make_node(
            'Gemm',
            ['36', '16.weight', '16.bias'],
            ['37'],
            name='Gemm_16',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['37'], ['38'], name='Relu_17'),
        helper.make_node(
            'Gemm',
            ['38', '18.weight', '18.bias'],
            ['39'],
            name='Gemm_18',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input.1', FLOAT, [16, 100]),
        ],
        outputs=[
            _vi('39', FLOAT, [16, 10]),
        ],
        initializer=[
            _tensor('0.bias', FLOAT, [50], [
                0.06874362379312515, 0.1215260922908783, -0.03796323388814926,
                -0.047220371663570404, 0.0851314440369606, 0.09796275943517685,
                0.12071841955184937, 0.07664817571640015, 0.11198078840970993, 0.02310258150100708,
                0.07579555362462997, 0.05929337441921234, -0.036450356245040894,
                0.11803308129310608, -0.011961907148361206, -0.08527068793773651,
                -0.057033807039260864, 0.1044885590672493, -0.018882740288972855,
                -0.008054572157561779, 0.10694648325443268, -0.022059820592403412,
                0.09017779678106308, 0.1540471315383911, 0.1271747350692749, 0.06436201930046082,
                0.1194877177476883, -0.010833785869181156, 0.1089724600315094,
                -0.044143423438072205, 0.06858711689710617, -0.03810128942131996,
                0.0594230554997921, 0.011302107945084572, 0.16360539197921753,
                -0.03886178508400917, 0.06342087686061859, 0.10477621853351593,
                0.07790201157331467, 0.025975681841373444, 0.15242689847946167,
                -0.07979436218738556, -0.015697987750172615, 0.16126343607902527,
                0.058438144624233246, -0.007473993580788374, 0.09990260750055313,
                0.06640422344207764, -0.02770175412297249, 0.049512993544340134,
            ]),
            _random_tensor('0.weight', [50, 100], seed=111),
            _tensor('10.bias', FLOAT, [50], [
                0.12787356972694397, 0.01754315197467804, 0.12297597527503967, 0.07300411909818649,
                0.05101786553859711, -0.009935596957802773, 0.13993382453918457,
                0.15092433989048004, 0.06841301918029785, -0.03337057679891586,
                -0.18426062166690826, -0.13440611958503723, 0.10937852412462234,
                0.11137652397155762, -0.10483825951814651, -0.02507081814110279,
                0.12054929882287979, 0.0411001481115818, 0.1838451772928238, 0.13574835658073425,
                -0.0077139283530414104, -0.12025056034326553, 0.0854426920413971,
                -0.05131257325410843, 0.13684552907943726, -0.014523052610456944,
                -0.08954862505197525, -0.025241060182452202, -0.008962735533714294,
                0.09331826120615005, -0.10867604613304138, -0.10423946380615234,
                0.17008665204048157, -0.03412630781531334, 0.07280059158802032,
                -0.04532545059919357, -0.10004503279924393, -0.11012918502092361,
                -0.0077126519754529, -0.1191520020365715, 0.12147060036659241, 0.10113030672073364,
                0.03328618407249451, 0.014212618581950665, -0.010599344968795776,
                0.10923430323600769, -0.01827055774629116, 0.17716272175312042,
                0.06910598278045654, -0.07394197583198547,
            ]),
            _random_tensor('10.weight', [50, 50], seed=112),
            _tensor('12.bias', FLOAT, [50], [
                -0.06509873270988464, 0.05613470822572708, -0.05249607563018799,
                -0.060684677213430405, 0.055331166833639145, 0.08404038101434708,
                0.0655064731836319, 0.13225528597831726, 0.035152286291122437, -0.0857200175523758,
                0.04633798822760582, -0.13850943744182587, -0.030993010848760605,
                0.07260533422231674, -0.0611225962638855, 0.04004671797156334, 0.03332715854048729,
                -0.13936835527420044, -0.11538780480623245, 0.03552905097603798,
                -0.07537106424570084, -0.10834012180566788, -0.16588839888572693,
                0.058801423758268356, 0.0744016021490097, 0.07377104461193085,
                -0.16663652658462524, 0.13944970071315765, -0.10723331570625305,
                0.16675545275211334, 0.11190473288297653, 0.1424584835767746, -0.10559768974781036,
                0.17358238995075226, 0.024868786334991455, -0.008324883878231049,
                -0.009020783007144928, 0.09669970721006393, 0.1663464903831482,
                0.051099903881549835, -0.11830130964517593, -0.13791216909885406,
                -0.054981157183647156, -0.14046736061573029, 0.024868272244930267,
                -0.0492456778883934, 0.13240450620651245, -0.1366450935602188,
                -0.006306866183876991, -0.06659865379333496,
            ]),
            _random_tensor('12.weight', [50, 50], seed=113),
            _tensor('14.bias', FLOAT, [50], [
                0.016014426946640015, 0.06593047082424164, -0.1345161348581314,
                -0.1251203864812851, -0.12696857750415802, 0.011852066963911057,
                0.1119963675737381, -0.0366256982088089, -0.07817808538675308,
                -0.0018910560756921768, -0.07488702237606049, 0.11818061023950577,
                -0.04405388981103897, -0.014389574527740479, 0.072415791451931,
                -0.040516626089811325, -0.06337642669677734, -0.038087353110313416,
                0.06708531081676483, 0.060243379324674606, 0.09579991549253464,
                -0.0834713950753212, -0.04309255629777908, -0.039707157760858536,
                -0.02101474069058895, -0.004626616835594177, 0.09738843142986298,
                -0.15382537245750427, -0.14784333109855652, 0.012172728776931763,
                0.18078944087028503, 0.018331220373511314, -0.1306842863559723,
                -0.1078730896115303, -0.049283646047115326, -0.04442322626709938,
                -0.05975477397441864, -0.03484858572483063, -0.1593368649482727,
                0.04525914043188095, -0.028948737308382988, 0.09824682772159576,
                -0.017328474670648575, -0.10201127827167511, 0.021711774170398712,
                0.02649231068789959, 0.1379029005765915, 0.0019947874825447798,
                -0.09130772948265076, 0.07110419124364853,
            ]),
            _random_tensor('14.weight', [50, 50], seed=114),
            _tensor('16.bias', FLOAT, [50], [
                -0.14252740144729614, 0.16887430846691132, -0.0887828916311264,
                -0.06314415484666824, -0.06602327525615692, 0.05441824719309807,
                0.06415509432554245, 0.06069942191243172, -0.0223076269030571, 0.10297013819217682,
                0.025865202769637108, -0.08093931525945663, -0.027676187455654144,
                0.05468316376209259, 0.1288861781358719, -0.0795307531952858,
                -0.018913164734840393, -0.1207500547170639, 0.17368493974208832,
                -0.049284402281045914, -0.05787952244281769, 0.06717755645513535,
                0.012359170243144035, 0.13264226913452148, -0.05257987976074219,
                0.017382705584168434, 0.06598390638828278, -0.09585361182689667,
                0.07884091138839722, 0.010707235895097256, 0.04929834231734276,
                -0.025524809956550598, 0.051943808794021606, 0.13757683336734772,
                -0.11596449464559555, -0.07238765060901642, 0.11116628348827362,
                -0.11908264458179474, -0.08664168417453766, 0.09629549086093903,
                0.11060114204883575, -0.013693519867956638, -0.1386561542749405,
                -0.06237578019499779, 0.08550456911325455, -0.12340494990348816,
                0.06833907216787338, -0.017610615119338036, -0.04134988784790039,
                0.02336009591817856,
            ]),
            _random_tensor('16.weight', [50, 50], seed=115),
            _tensor('18.bias', FLOAT, [10], [
                -0.028683319687843323, 0.031511370092630386, -0.015858041122555733,
                0.04559389129281044, 0.09545835852622986, -0.10511715710163116,
                -0.07386839389801025, -0.11918522417545319, -0.06869250535964966,
                0.09922939538955688,
            ]),
            _random_tensor('18.weight', [10, 50], seed=116),
            _tensor('2.bias', FLOAT, [50], [
                -0.044733885675668716, 0.053787779062986374, 0.07859575748443604,
                -0.06343381106853485, 0.15348155796527863, 0.14867684245109558,
                0.02656984142959118, -0.026198450475931168, -0.07519230246543884,
                -0.03524557128548622, 0.09328898042440414, 0.11387166380882263,
                -0.019346164539456367, 0.17526762187480927, -0.07706878334283829,
                0.15751178562641144, 0.019623270258307457, -0.07372663915157318,
                0.08727440983057022, 0.11638835817575455, 0.16839821636676788, 0.04258020967245102,
                -0.10223003476858139, 0.06937894970178604, -0.0855393335223198,
                0.12638899683952332, 0.020591460168361664, 0.1405806839466095,
                -0.0023452509194612503, -0.029579175636172295, 0.019782187417149544,
                0.06618925929069519, 0.16647274792194366, 0.14933745563030243,
                0.051312513649463654, 0.0006887729396112263, -0.07575076073408127,
                -0.054050710052251816, 0.13494345545768738, 0.025651181116700172,
                -0.09433789551258087, -0.02612384594976902, 0.030958404764533043,
                0.11118845641613007, 0.16908417642116547, 0.1360965222120285, 0.0985386073589325,
                0.048001762479543686, -0.047142088413238525, 0.12221584469079971,
            ]),
            _random_tensor('2.weight', [50, 50], seed=117),
            _tensor('4.bias', FLOAT, [50], [
                0.04200629144906998, -0.05310118943452835, -0.04059197008609772,
                0.1476421356201172, -0.044893037527799606, -0.0946018248796463,
                0.03687572851777077, 0.08952753245830536, -0.0013579304795712233,
                -0.04650532454252243, 0.10455886274576187, 0.04649180546402931,
                -0.09281352907419205, 0.14577698707580566, -0.04373973235487938,
                0.07441886514425278, -0.09758659452199936, 0.07919350266456604,
                -0.07836516946554184, 0.03809545934200287, -0.06411395221948624,
                0.0319918617606163, 0.051943857222795486, 0.00847010500729084, 0.12449851632118225,
                0.18247577548027039, -0.05370906740427017, 0.058310382068157196,
                -0.04016480967402458, 0.008250949904322624, -0.06189260259270668,
                -0.12295297533273697, 0.07729160040616989, 0.014789585024118423,
                0.10187598317861557, 0.0958903431892395, 0.06446435302495956, 0.012280937284231186,
                0.14996418356895447, -0.1411341279745102, -0.08492119610309601,
                -0.011174597777426243, -0.06453771144151688, -0.0344211682677269,
                0.0628582313656807, 0.0434207059442997, -0.04334687814116478,
                -0.029960226267576218, 0.15525946021080017, -0.04480167105793953,
            ]),
            _random_tensor('4.weight', [50, 50], seed=118),
            _tensor('6.bias', FLOAT, [50], [
                -0.1301494687795639, -0.016671590507030487, 0.09305505454540253,
                -0.0024569956585764885, -0.10665174573659897, 0.049031224101781845,
                -0.022929606959223747, 0.02805551514029503, -0.1490677148103714,
                0.1025087982416153, 0.00938428845256567, 0.15098121762275696, -0.11440007388591766,
                -0.06450272351503372, 0.016750779002904892, -0.08418712019920349,
                -0.14083871245384216, 0.03546612709760666, -0.12778249382972717,
                -0.10786302387714386, 0.0691528245806694, 0.04630193114280701, 0.09610986709594727,
                0.06807758659124374, -0.11870553344488144, -0.07684986293315887,
                0.17632094025611877, 0.11957243084907532, -0.018469832837581635,
                0.061927877366542816, 0.09733913093805313, 0.06544090062379837,
                0.08407261222600937, -0.09821699559688568, -0.02714831940829754,
                0.11982957273721695, -0.05582385137677193, 0.08686035871505737, 0.1096935048699379,
                -0.12632803618907928, 0.16949345171451569, -0.1535651534795761,
                -0.07482590526342392, 0.013653061352670193, 0.007351913955062628,
                0.12195851653814316, 0.002472013235092163, -0.03045388124883175,
                -0.06886417418718338, 0.053352996706962585,
            ]),
            _random_tensor('6.weight', [50, 50], seed=119),
            _tensor('8.bias', FLOAT, [50], [
                0.0448136180639267, -0.029453275725245476, 0.005919584538787603,
                -0.011282878927886486, 0.05477000027894974, 0.10227928310632706,
                0.005549189634621143, 0.09336987882852554, 0.1386832445859909, 0.15307164192199707,
                -0.02468901313841343, -0.0662059560418129, 0.0102847283706069,
                -0.02171068638563156, -0.11153922975063324, -0.08330245316028595,
                0.06905094534158707, 0.057425979524850845, 0.03267614543437958,
                0.04805871099233627, 0.09321744740009308, 0.1732863485813141, 0.043798334896564484,
                0.06929294764995575, -0.14251939952373505, 0.016439231112599373,
                -0.052573300898075104, -0.09261982887983322, 0.015587260015308857,
                0.12414858490228653, 0.15976372361183167, -0.1122899278998375, 0.12213458120822906,
                -0.03298468515276909, 0.12397517263889313, 0.008843302726745605,
                -0.12524719536304474, -0.10820302367210388, -0.09638859331607819,
                0.12722527980804443, 0.10527792572975159, -0.08983974158763885,
                0.10839671641588211, 0.13300462067127228, 0.11159244924783707,
                -0.054800763726234436, 0.1124715581536293, 0.09525484591722488,
                -0.04181470349431038, 0.049590643495321274,
            ]),
            _random_tensor('8.weight', [50, 50], seed=120),
        ],
    )
    return _model(graph, opset=9, ir_version=6, producer_name='pytorch', producer_version='1.5')


def make_Linear_32():
    """Ops: Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm"""
    nodes = [
        helper.make_node(
            'Gemm',
            ['input.1', '0.weight', '0.bias'],
            ['21'],
            name='Gemm_0',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['21'], ['22'], name='Relu_1'),
        helper.make_node(
            'Gemm',
            ['22', '2.weight', '2.bias'],
            ['23'],
            name='Gemm_2',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['23'], ['24'], name='Relu_3'),
        helper.make_node(
            'Gemm',
            ['24', '4.weight', '4.bias'],
            ['25'],
            name='Gemm_4',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['25'], ['26'], name='Relu_5'),
        helper.make_node(
            'Gemm',
            ['26', '6.weight', '6.bias'],
            ['27'],
            name='Gemm_6',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['27'], ['28'], name='Relu_7'),
        helper.make_node(
            'Gemm',
            ['28', '8.weight', '8.bias'],
            ['29'],
            name='Gemm_8',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['29'], ['30'], name='Relu_9'),
        helper.make_node(
            'Gemm',
            ['30', '10.weight', '10.bias'],
            ['31'],
            name='Gemm_10',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['31'], ['32'], name='Relu_11'),
        helper.make_node(
            'Gemm',
            ['32', '12.weight', '12.bias'],
            ['33'],
            name='Gemm_12',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['33'], ['34'], name='Relu_13'),
        helper.make_node(
            'Gemm',
            ['34', '14.weight', '14.bias'],
            ['35'],
            name='Gemm_14',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['35'], ['36'], name='Relu_15'),
        helper.make_node(
            'Gemm',
            ['36', '16.weight', '16.bias'],
            ['37'],
            name='Gemm_16',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['37'], ['38'], name='Relu_17'),
        helper.make_node(
            'Gemm',
            ['38', '18.weight', '18.bias'],
            ['39'],
            name='Gemm_18',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input.1', FLOAT, [32, 100]),
        ],
        outputs=[
            _vi('39', FLOAT, [32, 10]),
        ],
        initializer=[
            _tensor('0.bias', FLOAT, [50], [
                0.11124264448881149, 0.09824328124523163, 0.008467835374176502,
                -0.06339164823293686, -0.04812309890985489, 0.07819465547800064,
                0.04600067436695099, 0.12569192051887512, -0.006026973016560078,
                0.042687151581048965, 0.054539717733860016, 0.03256354108452797,
                0.1459706425666809, 0.011990390717983246, 0.11928670108318329,
                -0.035920701920986176, -0.01965402625501156, 0.01519996952265501,
                -0.020567703992128372, 0.006464967038482428, -0.040367837995290756,
                0.1230701357126236, 0.10814201086759567, 0.032152947038412094, 0.0349125899374485,
                0.09016934037208557, -0.0026081986725330353, 0.08004175126552582,
                0.08260804414749146, -0.030275214463472366, -0.017403658479452133,
                0.06508535891771317, 0.07616975903511047, 0.12453043460845947, 0.05175986513495445,
                -0.034277405589818954, -0.001854078029282391, 0.07586805522441864,
                -0.03652631491422653, -0.033068954944610596, -0.04171779379248619,
                0.01669793576002121, 0.11060179024934769, 0.07502856105566025,
                -0.009946250356733799, 0.11157600581645966, 0.11138223856687546,
                0.0743454173207283, 0.05090726912021637, -0.05379515886306763,
            ]),
            _random_tensor('0.weight', [50, 100], seed=121),
            _tensor('10.bias', FLOAT, [50], [
                0.023677507415413857, 0.01546807587146759, 0.12556889653205872,
                -0.025954080745577812, 0.13634377717971802, 0.022078335285186768,
                0.044483307749032974, 0.06803391128778458, -0.1336556077003479,
                -0.02744845300912857, -0.07780878245830536, 0.060131218284368515,
                0.08147525042295456, -0.1075383722782135, 0.038635917007923126,
                0.04753212630748749, 0.007815081626176834, -0.091900534927845,
                -0.040418289601802826, 0.12894690036773682, 0.04103463888168335,
                0.05978541821241379, -0.01061064749956131, -0.11792825162410736,
                -0.03278864547610283, 0.11842191964387894, -0.0180479995906353,
                -0.10795165598392487, 0.08416201174259186, 0.13871043920516968,
                -0.09923344850540161, 0.0004032762080896646, -0.01704275608062744,
                -0.09447328001260757, 0.02773180603981018, 0.03783107176423073,
                0.11792910844087601, -0.10673742741346359, 0.0416945181787014, 0.14280104637145996,
                0.08878400176763535, -0.08478271961212158, 0.14648577570915222, 0.1269282102584839,
                0.07769495993852615, -0.09769672155380249, -0.12921839952468872,
                -0.08971879631280899, 0.13251297175884247, -0.05491340905427933,
            ]),
            _random_tensor('10.weight', [50, 50], seed=122),
            _tensor('12.bias', FLOAT, [50], [
                -0.06826256215572357, 0.0388849675655365, 0.07320192456245422, 0.08387485146522522,
                -0.07019667327404022, 0.12242702394723892, -0.09621168673038483,
                0.11204088479280472, -0.13937069475650787, -0.0626523569226265,
                0.056398920714855194, 0.1534673422574997, -0.001662513823248446,
                -0.10726587474346161, 0.15035594999790192, -0.014129229821264744,
                0.11882692575454712, 0.015099582262337208, 0.1413463056087494,
                0.0007984754047356546, 0.08997610211372375, -0.07151174545288086,
                0.11034654080867767, -0.07181119918823242, 0.09321177005767822,
                -0.042380593717098236, -0.10582658648490906, 0.10396461933851242,
                0.045979660004377365, 0.12390460819005966, -0.01176830567419529,
                0.1260722577571869, -0.004162704572081566, 0.032158028334379196,
                0.05012675002217293, 0.010814670473337173, 0.061898116022348404,
                0.02869694121181965, 0.06419070810079575, -0.05789436399936676,
                -0.08914010971784592, 0.07718385756015778, -0.11866763234138489,
                0.046556998044252396, 0.010457875207066536, -0.11218304932117462,
                0.03364101052284241, -0.050707247108221054, -0.04438620060682297, 0.1124512255191803,
            ]),
            _random_tensor('12.weight', [50, 50], seed=123),
            _tensor('14.bias', FLOAT, [50], [
                -0.042912423610687256, -0.08314505219459534, 0.010369965806603432,
                0.10941207408905029, -0.06298284977674484, 0.0006218135822564363,
                -0.046892426908016205, 0.14291897416114807, -0.1065610870718956,
                0.12685929238796234, -0.06573846936225891, -0.08174490928649902,
                -0.12673887610435486, 0.11660975217819214, -0.13191364705562592,
                0.1266399770975113, 0.10539578646421432, -0.03997437655925751, 0.08274240791797638,
                -0.05659307911992073, -0.016876066103577614, -0.0644751563668251,
                0.09650067239999771, -0.1165977269411087, 0.02729981020092964, 0.1455359309911728,
                0.03180919587612152, -0.1552070677280426, 0.12558980286121368,
                0.018251288682222366, 0.03671560436487198, 0.08113941550254822, 0.0933469608426094,
                0.12075929343700409, -0.1378369927406311, 0.03614622354507446, 0.11320675164461136,
                -0.008757159113883972, -0.11563608795404434, 0.08717223256826401,
                -0.1610676348209381, 0.058534447103738785, 0.13570138812065125,
                0.09963928163051605, 0.015034488402307034, 0.05861431732773781, 0.1055833026766777,
                -0.051912613213062286, -0.013903460465371609, 0.11571002751588821,
            ]),
            _random_tensor('14.weight', [50, 50], seed=124),
            _tensor('16.bias', FLOAT, [50], [
                0.13645648956298828, 0.10871178656816483, -0.04361168295145035,
                0.06587550044059753, 0.0516870953142643, -0.012960155494511127,
                0.10853584110736847, -0.010951125994324684, -0.1266385167837143,
                -0.1094653308391571, 0.12544943392276764, -0.08655308932065964,
                0.024057991802692413, 0.0033439581748098135, 0.10357099026441574,
                -0.042763128876686096, 0.13817517459392548, -0.06586167216300964,
                -0.14774498343467712, -0.10796114802360535, -0.02931831032037735,
                -0.09633642435073853, 0.09680622816085815, -0.05977587401866913,
                0.06083926931023598, -0.04657533019781113, 0.06506074219942093,
                -0.15826131403446198, -0.0730171725153923, -0.006507332436740398,
                0.12966391444206238, 0.031853556632995605, 0.05254780873656273,
                0.08588340133428574, 0.12415903806686401, 0.11529742181301117,
                -0.10537046194076538, -0.13778118789196014, -0.10065561532974243,
                -0.02065938338637352, 0.0062689147889614105, -0.17919579148292542,
                -0.1153501570224762, 0.10259155184030533, 0.02723391354084015, 0.08433964848518372,
                0.007203978020697832, 0.07054678350687027, 0.07516984641551971, -0.13919509947299957,
            ]),
            _random_tensor('16.weight', [50, 50], seed=125),
            _tensor('18.bias', FLOAT, [10], [
                -0.0845300629734993, 0.06066350266337395, -0.0744476318359375,
                -0.11339960992336273, 0.07592877745628357, -0.07158249616622925,
                0.030116315931081772, -0.13125865161418915, 0.02029414474964142,
                -0.02809321880340576,
            ]),
            _random_tensor('18.weight', [10, 50], seed=126),
            _tensor('2.bias', FLOAT, [50], [
                0.09465888142585754, -0.04027128964662552, 0.08556117117404938,
                9.81339908321388e-05, 0.10296808928251266, 0.031810734421014786,
                0.12633393704891205, -0.04990290105342865, -0.051970064640045166,
                0.08027298748493195, 0.10882756114006042, 0.062105171382427216,
                0.008045201189815998, 0.08600206673145294, 0.06580343842506409,
                0.15130916237831116, -0.08689317107200623, -0.07181258499622345,
                0.11189711093902588, -0.017252318561077118, -0.11066558957099915,
                0.07146942615509033, -0.005338592920452356, 0.09884057939052582,
                0.08768560737371445, 0.06407410651445389, -0.03988203778862953,
                -0.06665781140327454, 0.039171550422906876, 0.054918356239795685,
                -0.04876664653420448, -0.020347272977232933, 0.0013799254084005952,
                0.09394200146198273, -0.03622853010892868, -0.0942491963505745,
                0.05046489089727402, 0.02834177017211914, -0.04078937694430351,
                -0.01767958141863346, 0.12003850191831589, -0.10315573215484619,
                -0.042500488460063934, 0.15254095196723938, 0.09979313611984253,
                0.03916972875595093, -0.007253315299749374, -0.0852055475115776,
                0.0820893943309784, 0.1186923235654831,
            ]),
            _random_tensor('2.weight', [50, 50], seed=127),
            _tensor('4.bias', FLOAT, [50], [
                0.024598661810159683, 0.08616624027490616, 0.038595184683799744,
                0.08363842964172363, 0.09489761292934418, -0.04017442837357521,
                0.06529752165079117, 0.10095158964395523, 0.14577096700668335, 0.06753067672252655,
                -0.0175277441740036, 0.030973171815276146, 0.03054368868470192,
                -0.08081741631031036, 0.08830731362104416, 0.0858900398015976, 0.06582188606262207,
                -0.08109358698129654, 0.11148745566606522, 0.07020239531993866, 0.146789088845253,
                0.14938125014305115, 0.006371157709509134, -0.08334558457136154,
                0.14391906559467316, 0.03840988501906395, 0.11901327967643738, 0.10867060720920563,
                0.034775227308273315, 0.11030173301696777, 0.024303283542394638,
                -0.11603455990552902, -0.03790628910064697, 0.13132940232753754,
                -0.13394464552402496, -0.04722129553556442, 0.03803771734237671,
                0.03392547369003296, 0.06807977706193924, -0.042708538472652435,
                0.1414441466331482, -0.06422796100378036, -0.05185883864760399,
                -0.050145845860242844, -0.06538667529821396, -0.1171017587184906,
                0.09188596159219742, 0.11895687878131866, -0.09153135120868683, 0.09006752818822861,
            ]),
            _random_tensor('4.weight', [50, 50], seed=128),
            _tensor('6.bias', FLOAT, [50], [
                -0.1229783222079277, 0.11139104515314102, -0.10524418950080872,
                0.0016380425076931715, 0.07938187569379807, 0.003908644896000624,
                -0.028028862550854683, -0.03216879814863205, -0.031408797949552536,
                -0.04045102000236511, 0.09041076898574829, 0.015676679089665413,
                -0.002135281218215823, -0.11499056965112686, -0.007753490470349789,
                0.025194739922881126, 0.13036774098873138, 0.11130959540605545,
                -0.09819938987493515, 0.040943779051303864, 0.03980429843068123,
                0.08250012993812561, 0.14428070187568665, 0.041119564324617386,
                -0.02864460088312626, -0.01911313831806183, -0.03500247001647949,
                -0.0062780012376606464, -0.11006780713796616, -0.011378481984138489,
                0.09277238696813583, 0.017135074362158775, -0.1253904551267624,
                -0.02176308073103428, -0.10329070687294006, 0.151449516415596, 0.133637934923172,
                0.056119631975889206, 0.005917373578995466, -0.07999040931463242,
                -0.04148922860622406, 0.009720489382743835, 0.031680550426244736,
                0.13133378326892853, 0.0069076018407940865, 0.15768522024154663,
                -0.011727947741746902, 0.032221000641584396, -0.03057415783405304,
                0.08015977591276169,
            ]),
            _random_tensor('6.weight', [50, 50], seed=129),
            _tensor('8.bias', FLOAT, [50], [
                -0.08394850790500641, -0.08946047723293304, 0.0023817981127649546,
                0.08956918865442276, 0.11502696573734283, 0.1312992125749588, -0.10089538991451263,
                0.06887184828519821, -0.05314618721604347, -0.11070530861616135,
                -0.10399400442838669, -0.09465840458869934, -0.07996507734060287,
                -0.055972978472709656, -0.0011011596070602536, -0.11151980608701706,
                -0.01033539418131113, 0.08838698267936707, -0.013264666311442852,
                0.1253567337989807, 0.028946250677108765, 0.08221150189638138, 0.08310119807720184,
                0.11621533334255219, -0.008178315125405788, -0.08422097563743591,
                0.014766180887818336, 0.09799371659755707, 0.05157942697405815,
                0.08685771375894547, -0.1275632679462433, 0.11424566060304642,
                -0.04376813769340515, -0.08646500110626221, 0.07470041513442993,
                -0.001382552902214229, 0.06231479346752167, 0.03777238354086876,
                -0.03722120448946953, 0.06540307402610779, 0.023184368386864662,
                -0.05114809796214104, -0.04856577143073082, -0.059934988617897034,
                0.1061999574303627, 0.1266958862543106, 0.009567365981638432, -0.03149254247546196,
                0.10635311156511307, -0.1022690087556839,
            ]),
            _random_tensor('8.weight', [50, 50], seed=130),
        ],
    )
    return _model(graph, opset=9, ir_version=6, producer_name='pytorch', producer_version='1.5')


def make_Linear_64():
    """Ops: Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm, Relu, Gemm"""
    nodes = [
        helper.make_node(
            'Gemm',
            ['input.1', '0.weight', '0.bias'],
            ['21'],
            name='Gemm_0',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['21'], ['22'], name='Relu_1'),
        helper.make_node(
            'Gemm',
            ['22', '2.weight', '2.bias'],
            ['23'],
            name='Gemm_2',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['23'], ['24'], name='Relu_3'),
        helper.make_node(
            'Gemm',
            ['24', '4.weight', '4.bias'],
            ['25'],
            name='Gemm_4',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['25'], ['26'], name='Relu_5'),
        helper.make_node(
            'Gemm',
            ['26', '6.weight', '6.bias'],
            ['27'],
            name='Gemm_6',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['27'], ['28'], name='Relu_7'),
        helper.make_node(
            'Gemm',
            ['28', '8.weight', '8.bias'],
            ['29'],
            name='Gemm_8',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['29'], ['30'], name='Relu_9'),
        helper.make_node(
            'Gemm',
            ['30', '10.weight', '10.bias'],
            ['31'],
            name='Gemm_10',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['31'], ['32'], name='Relu_11'),
        helper.make_node(
            'Gemm',
            ['32', '12.weight', '12.bias'],
            ['33'],
            name='Gemm_12',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['33'], ['34'], name='Relu_13'),
        helper.make_node(
            'Gemm',
            ['34', '14.weight', '14.bias'],
            ['35'],
            name='Gemm_14',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['35'], ['36'], name='Relu_15'),
        helper.make_node(
            'Gemm',
            ['36', '16.weight', '16.bias'],
            ['37'],
            name='Gemm_16',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
        helper.make_node('Relu', ['37'], ['38'], name='Relu_17'),
        helper.make_node(
            'Gemm',
            ['38', '18.weight', '18.bias'],
            ['39'],
            name='Gemm_18',
            alpha=1.0,
            beta=1.0,
            transB=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('input.1', FLOAT, [64, 100]),
        ],
        outputs=[
            _vi('39', FLOAT, [64, 10]),
        ],
        initializer=[
            _tensor('0.bias', FLOAT, [50], [
                0.09159639477729797, -0.006669722031801939, 0.13838760554790497,
                0.007312551606446505, 0.050098199397325516, 0.15664000809192657,
                0.1113118976354599, -0.01220855861902237, 0.04115782678127289, 0.09007082134485245,
                0.04722576215863228, 0.050395555794239044, 0.04292930290102959,
                -0.0252480898052454, 0.038219235837459564, 0.15084995329380035, 0.1120382621884346,
                0.04777619615197182, 0.04694032296538353, 0.03356737643480301, 0.14677441120147705,
                0.005037274677306414, -0.013039053417742252, 0.005265332292765379,
                -0.036055635660886765, -0.03629482164978981, 0.08636198192834854,
                0.06236021965742111, -0.06167418509721756, -0.05445173755288124,
                0.06576776504516602, -0.004990452900528908, 0.051186129450798035,
                0.003741896478459239, -0.06545235961675644, -0.05202485993504524,
                0.07194308191537857, 0.13038986921310425, 0.016981661319732666,
                -0.04673358425498009, 0.11374427378177643, -0.025251995772123337,
                0.014630576595664024, 0.0005040550604462624, 0.10728251188993454,
                0.008620365522801876, 0.0760183185338974, 0.07534903287887573,
                0.018748648464679718, -0.09183405339717865,
            ]),
            _random_tensor('0.weight', [50, 100], seed=131),
            _tensor('10.bias', FLOAT, [50], [
                0.023598425090312958, -0.13623277842998505, 0.07419407367706299,
                -0.05104606971144676, -0.011362170800566673, 0.06712517142295837,
                0.08822090178728104, -0.0020764917135238647, 0.11936502158641815,
                -0.06489405781030655, -0.10444420576095581, 0.037246569991111755,
                -0.08549445867538452, 0.13663238286972046, -0.13414929807186127,
                0.0032362418714910746, -0.12991856038570404, 0.005575540475547314,
                0.08955343067646027, 0.08483952283859253, 0.1431363821029663, 0.048573967069387436,
                -0.1422257423400879, 0.04993279650807381, 0.11434306204319, -0.0134720578789711,
                0.004054172430187464, -0.14051389694213867, 0.04162381589412689,
                -0.15304715931415558, 0.0083013866096735, -0.03706217184662819,
                0.010971440933644772, -0.06586533039808273, 0.018230563029646873,
                0.030368328094482422, 0.13547532260417938, -0.07951928675174713,
                0.13427899777889252, 0.02883337251842022, -0.06782680749893188,
                0.07169642299413681, -0.007853972725570202, -0.0510685108602047,
                -0.11296464502811432, -0.027279041707515717, 0.13714829087257385,
                0.07250010967254639, -0.0007291536312550306, -0.06502445042133331,
            ]),
            _random_tensor('10.weight', [50, 50], seed=132),
            _tensor('12.bias', FLOAT, [50], [
                -0.0801251232624054, -0.05511970445513725, 0.03490331768989563,
                0.02558317594230175, 0.05685405805706978, -0.0002709181571844965,
                0.020529454573988914, 0.16004309058189392, -0.0037057101726531982,
                0.021892093122005463, 0.0422116257250309, 0.13861492276191711, 0.04279313609004021,
                -0.06523241847753525, 0.025549277663230896, -0.11060528457164764,
                -0.11546116322278976, 0.05192999541759491, -0.12274068593978882,
                -0.047401707619428635, -0.09217683970928192, -0.05534840375185013,
                0.12988485395908356, 0.06574156880378723, -0.0901506319642067,
                0.049082305282354355, 0.021567750722169876, -0.03045421838760376,
                -0.04347901791334152, -0.02055964060127735, -0.13669030368328094,
                0.08841653913259506, 0.06565003842115402, 0.029235024005174637,
                0.04059187322854996, -0.06516950577497482, 0.07286635041236877,
                0.027074089273810387, 0.05025995522737503, 0.046343252062797546,
                0.13113434612751007, -0.040170587599277496, 0.0747847706079483,
                0.14416146278381348, -0.07301327586174011, -0.13885334134101868,
                0.061743613332509995, 0.015627363696694374, -0.14226868748664856,
                -0.040349800139665604,
            ]),
            _random_tensor('12.weight', [50, 50], seed=133),
            _tensor('14.bias', FLOAT, [50], [
                0.06647215783596039, 0.09063264727592468, -0.13074101507663727,
                -0.13949799537658691, 0.004678502678871155, -0.016874082386493683,
                -0.0011929124593734741, 0.03477223217487335, -0.14273054897785187,
                -0.12513236701488495, -0.1137811541557312, -0.10309846699237823,
                0.08320702612400055, -0.17264628410339355, -0.1673404425382614,
                -0.02677275612950325, 0.13770265877246857, -0.12795111536979675,
                0.11456170678138733, -0.01058092713356018, -0.04381580278277397,
                -0.14236316084861755, 0.04663165286183357, 0.09976256638765335,
                -0.016420619562268257, -0.03107152134180069, -0.11892712116241455,
                -0.14145000278949738, -0.09588667750358582, -0.10125806927680969,
                -0.07667424529790878, 0.08648142218589783, 0.13760849833488464,
                0.020085202530026436, -0.1349070966243744, -0.05883036181330681,
                0.06423906236886978, 0.021977975964546204, 0.004053518176078796,
                -0.11900798976421356, 0.09760946035385132, -0.07316134124994278,
                -0.12477194517850876, 0.05623914301395416, -0.1079934686422348,
                0.15659739077091217, 0.06688092648983002, -0.020453844219446182,
                -0.06655491888523102, 0.042102597653865814,
            ]),
            _random_tensor('14.weight', [50, 50], seed=134),
            _tensor('16.bias', FLOAT, [50], [
                0.12710162997245789, -0.11911947280168533, -0.07807320356369019,
                -0.035091958940029144, -0.1217832863330841, 0.07629904896020889,
                0.03150901570916176, 0.050709955394268036, 0.0009314765920862556,
                0.01097114384174347, 0.08148525655269623, 0.12900851666927338, 0.0837954506278038,
                -0.10252774506807327, 0.026486115530133247, 0.12476948648691177,
                0.11806709319353104, -0.06699508428573608, -0.06001771613955498,
                0.025374993681907654, -0.16904965043067932, 0.07055971026420593,
                -0.087677963078022, -0.06734836846590042, 0.09914802014827728,
                -0.02004489302635193, 0.021534450352191925, -0.1340482234954834,
                0.10110945254564285, 0.11758943647146225, -0.1362207531929016, -0.0481615886092186,
                -0.012928187847137451, -0.0925549864768982, -0.06118923798203468,
                -0.1054878681898117, -0.11407053470611572, -0.07652084529399872,
                -0.015235151164233685, 0.030054721981287003, -0.13776032626628876,
                -0.07530069351196289, -0.06638159602880478, -0.14768138527870178,
                -0.015214302577078342, -0.125348299741745, 0.0817929357290268, 0.12177981436252594,
                -0.09903938323259354, 0.004565980285406113,
            ]),
            _random_tensor('16.weight', [50, 50], seed=135),
            _tensor('18.bias', FLOAT, [10], [
                0.09701453149318695, 0.03215126693248749, -0.0013841761974617839,
                -0.03331436961889267, 0.059485986828804016, -0.09943024069070816,
                0.06851242482662201, 0.022922510281205177, 0.06729692965745926,
                -0.051858361810445786,
            ]),
            _random_tensor('18.weight', [10, 50], seed=136),
            _tensor('2.bias', FLOAT, [50], [
                -0.057317283004522324, 0.05470332130789757, -0.00017579866107553244,
                0.025761187076568604, 0.004599941428750753, -0.1244237944483757,
                0.08218210935592651, 0.09337401390075684, 0.13487280905246735,
                0.007681787014007568, 0.17570124566555023, 0.14757677912712097, 0.1500885784626007,
                -0.0012366429436951876, -0.08663220703601837, 0.02858916111290455,
                -0.06410738825798035, 0.032137297093868256, -0.1023068055510521,
                -0.01723470538854599, 0.0846070945262909, 0.014609461650252342,
                -0.05716143921017647, 0.10196615755558014, 0.15495315194129944,
                0.012524719350039959, -0.051979098469018936, -0.001390553079545498,
                -0.014226754195988178, 0.05362191051244736, 0.12925197184085846,
                -0.02208387665450573, 0.17744328081607819, -0.046732302755117416,
                -0.018103472888469696, 0.0403139665722847, 0.09309444576501846,
                0.07387915998697281, 0.0640542134642601, 0.06046606972813606, -0.12276843190193176,
                0.06488797813653946, 0.1577167510986328, 0.07812289893627167, 0.11168011277914047,
                0.0024907258339226246, -0.07085540145635605, -0.09919169545173645,
                0.1299211084842682, 0.14291509985923767,
            ]),
            _random_tensor('2.weight', [50, 50], seed=137),
            _tensor('4.bias', FLOAT, [50], [
                -0.1099703386425972, 0.07785989344120026, -0.05281399190425873,
                -0.03463250771164894, 0.029007652774453163, -0.054168153554201126,
                0.022841189056634903, 0.1798921525478363, 0.03891337290406227, 0.04181481525301933,
                -0.011574423871934414, -0.07251907885074615, 0.1228223443031311,
                0.10927148163318634, -0.007268114946782589, -0.10726579278707504,
                -0.019819920882582664, 0.13857461512088776, 0.11483220756053925,
                0.12228760123252869, -0.07547355443239212, 0.14420607686042786,
                -0.033190470188856125, -0.019701037555933, -0.01583164744079113,
                0.12164537608623505, 0.05813533440232277, 0.0008929441100917757,
                0.14075720310211182, 0.019338877871632576, -0.016825949773192406,
                0.14181609451770782, 0.01773025095462799, 0.06352683901786804,
                -0.07688445597887039, 0.15308715403079987, -0.03269578516483307,
                0.019046491011977196, -0.10575303435325623, 0.10171069949865341,
                0.11624108999967575, 0.07952452450990677, 0.07676685601472855,
                -0.09268729388713837, -0.09515844285488129, 0.08966265618801117,
                0.12256971001625061, 0.09974833577871323, -0.0020852594170719385,
                0.07005017250776291,
            ]),
            _random_tensor('4.weight', [50, 50], seed=138),
            _tensor('6.bias', FLOAT, [50], [
                -0.038818925619125366, 0.06811299920082092, 0.1118309274315834,
                0.07874610275030136, 0.05133644491434097, 0.01601332612335682,
                -0.08843040466308594, -0.018260041251778603, 0.021909251809120178,
                0.03044956363737583, 0.09717729687690735, 0.14612819254398346,
                0.033583976328372955, 0.07662538439035416, 0.07066786289215088, 0.1253795325756073,
                0.006381378974765539, 0.02676522172987461, 0.11174061894416809,
                -0.015764767304062843, -0.11681172251701355, -0.050747837871313095,
                0.0873001366853714, 0.019174877554178238, -0.12318209558725357,
                -0.011137604713439941, -0.10481218248605728, 0.01928110420703888,
                0.16160382330417633, 0.13084180653095245, 0.05665783956646919,
                -0.12876908481121063, -0.02381623350083828, 0.05391921475529671,
                0.12290484458208084, 0.14049169421195984, 0.06409084051847458,
                -0.09369882196187973, 0.10229893773794174, 0.026894323527812958,
                0.09336742758750916, 0.10093056410551071, -0.0749121680855751, 0.080393485724926,
                -0.07547304034233093, 0.10684267431497574, 0.04369564354419708,
                -0.07769715040922165, -0.07865709811449051, -0.02007412724196911,
            ]),
            _random_tensor('6.weight', [50, 50], seed=139),
            _tensor('8.bias', FLOAT, [50], [
                0.03757968917489052, 0.044984981417655945, -0.10111311078071594,
                0.058455660939216614, 0.01773645728826523, -0.07982795685529709,
                0.0912586972117424, -0.09884101152420044, -0.039294760674238205,
                0.04224107041954994, 0.06647494435310364, -0.10234533250331879,
                -0.019759902730584145, -0.07916851341724396, 0.11428356915712357,
                0.007246797904372215, 0.08650056272745132, -0.11538772284984589,
                -0.03382350504398346, 0.04845535755157471, -0.011813861317932606,
                -0.06618679314851761, 0.03560318797826767, -0.060869812965393066,
                -0.1252366602420807, 0.10331248492002487, -0.12572361528873444,
                0.14386624097824097, 0.10191508382558823, 0.14575746655464172,
                -0.07625168561935425, 0.13770657777786255, 0.05900927633047104,
                -0.0907086580991745, 0.02626039646565914, 0.12620759010314941, 0.03937942162156105,
                0.12115412205457687, 0.10908836126327515, -0.13738510012626648,
                -0.10638768970966339, -0.03824464604258537, -0.11428388208150864,
                0.023384658619761467, -0.07685241103172302, 0.025741903111338615,
                -0.07088501751422882, 0.07981190085411072, 0.10838404297828674, 0.05189177393913269,
            ]),
            _random_tensor('8.weight', [50, 50], seed=140),
        ],
    )
    return _model(graph, opset=9, ir_version=6, producer_name='pytorch', producer_version='1.5')


def make_Log():
    """Ops: Log"""
    nodes = [
        helper.make_node('Log', ['onnx::Log_0'], ['1'], name='/Log'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::Log_0', FLOAT, [4]),
        ],
        outputs=[
            _vi('1', FLOAT, [4]),
        ],
    )
    return _model(graph, opset=14, ir_version=7, producer_name='pytorch', producer_version='1.13.1')


def make_MatMul_Stacked():
    """Ops: MatMul"""
    nodes = [
        helper.make_node('MatMul', ['input1', 'input2'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'AddGraph',
        inputs=[
            _vi('input1', FLOAT, ['N', 2, 2]),
            _vi('input2', FLOAT, [2, 1]),
        ],
        outputs=[
            _vi('output', FLOAT, ['N', 2, 1]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_MatMul_Stacked2():
    """Ops: MatMul"""
    nodes = [
        helper.make_node('MatMul', ['input1', 'input2'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'AddGraph',
        inputs=[
            _vi('input1', FLOAT, ['N', 2, 2]),
            _vi('input2', FLOAT, ['N', 2, 1]),
        ],
        outputs=[
            _vi('output', FLOAT, ['N', 2, 1]),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_Max():
    """Ops: Max"""
    nodes = [
        helper.make_node('Max', ['X1', 'X2'], ['Y'], name='Max'),
    ]
    graph = helper.make_graph(
        nodes,
        'test-model',
        inputs=[
            _vi('X1', FLOAT, [1, 3]),
            _vi('X2', FLOAT, [1, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 3]),
        ],
    )
    return _model(graph, opset=13, ir_version=8, producer_name='onnx-example')


def make_MaxMultidirectionalBroadcast():
    """Ops: Max"""
    nodes = [
        helper.make_node('Max', ['A', 'B', 'C'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Max',
        inputs=[
            _vi('A', FLOAT, [3, 1]),
            _vi('B', FLOAT, [2, 3, 1]),
            _vi('C', FLOAT, [1, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_MaxPool1d():
    """Ops: MaxPool"""
    nodes = [
        helper.make_node(
            'MaxPool',
            ['onnx::MaxPool_0'],
            ['1'],
            name='MaxPool_0',
            kernel_shape=[3],
            pads=[0, 0],
            strides=[1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::MaxPool_0', FLOAT, [1, 6, 10]),
        ],
        outputs=[
            _vi('1', FLOAT, [1, 6, 8]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_MaxPool2d():
    """Ops: MaxPool"""
    nodes = [
        helper.make_node(
            'MaxPool',
            ['onnx::MaxPool_0'],
            ['1'],
            name='MaxPool_0',
            kernel_shape=[3, 2],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::MaxPool_0', FLOAT, [1, 1, 5, 10]),
        ],
        outputs=[
            _vi('1', FLOAT, [1, 1, 3, 9]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_MaxPool2d_AsymPad():
    """Ops: MaxPool"""
    nodes = [
        helper.make_node(
            'MaxPool',
            ['X'],
            ['Y'],
            kernel_shape=[2, 2],
            pads=[0, 1, 0, 1],
            strides=[1, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'MaxPool2d_AsymPad',
        inputs=[
            _vi('X', FLOAT, [1, 1, 4, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 3, 5]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_MaxPool2d_CeilMode():
    """Ops: MaxPool"""
    nodes = [
        helper.make_node(
            'MaxPool',
            ['X'],
            ['Y'],
            ceil_mode=1,
            kernel_shape=[2, 2],
            strides=[2, 2],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'maxpool_ceil',
        inputs=[
            _vi('X', FLOAT, [1, 1, 5, 5]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 1, 3, 3]),
        ],
    )
    return _model(graph, opset=11, ir_version=6)


def make_MaxPool3d():
    """Ops: MaxPool"""
    nodes = [
        helper.make_node(
            'MaxPool',
            ['onnx::MaxPool_0'],
            ['1'],
            name='MaxPool_0',
            kernel_shape=[3, 2, 2],
            pads=[0, 0, 0, 0, 0, 0],
            strides=[1, 1, 1],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::MaxPool_0', FLOAT, [1, 1, 3, 3, 3]),
        ],
        outputs=[
            _vi('1', FLOAT, [1, 1, 1, 2, 2]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_MeanMultidirectionalBroadcast():
    """Ops: Mean"""
    nodes = [
        helper.make_node('Mean', ['A', 'B', 'C'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Mean',
        inputs=[
            _vi('A', FLOAT, [3, 1]),
            _vi('B', FLOAT, [2, 3, 1]),
            _vi('C', FLOAT, [1, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_MinMultidirectionalBroadcast():
    """Ops: Min"""
    nodes = [
        helper.make_node('Min', ['A', 'B', 'C'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Min',
        inputs=[
            _vi('A', FLOAT, [3, 1]),
            _vi('B', FLOAT, [2, 3, 1]),
            _vi('C', FLOAT, [1, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Mod_ConstantFolding():
    """Ops: Constant, Mod"""
    nodes = [
        helper.make_node('Constant', [], ['X'], value=_tensor('X', INT64, [3], [10, 7, 5])),
        helper.make_node('Constant', [], ['D'], value=_tensor('D', INT64, [3], [3, 3, 3])),
        helper.make_node('Mod', ['X', 'D'], ['Y'], fmod=0),
    ]
    graph = helper.make_graph(
        nodes,
        'Mod_ConstantFolding',
        inputs=[
        ],
        outputs=[
            _vi('Y', INT64, [3]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_Mul():
    """Ops: Mul"""
    nodes = [
        helper.make_node('Mul', ['onnx::Mul_0', 'onnx::Mul_1'], ['2'], name='Mul_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Mul_0', FLOAT, [2]),
            _vi('onnx::Mul_1', FLOAT, [2]),
        ],
        outputs=[
            _vi('2', FLOAT, [2]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Neg():
    """Ops: Neg"""
    nodes = [
        helper.make_node('Neg', ['onnx::Neg_0'], ['1'], name='Neg_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Neg_0', FLOAT, [12]),
        ],
        outputs=[
            _vi('1', FLOAT, [12]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_NonZero():
    """Ops: NonZero"""
    nodes = [
        helper.make_node('NonZero', ['data'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', UINT8, [2, 2, 3]),
        ],
        outputs=[
            _vi('output', INT64, [3, 'N']),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_NonZero_Constant():
    """Ops: Constant, NonZero"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['constant_data'],
            value=_tensor('const_tensor', BOOL, [2, 2, 3], [False, True, False, True, True, False, False, False, True, False, True, True]),
        ),
        helper.make_node('NonZero', ['constant_data'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
        ],
        outputs=[
            _vi('output', INT64, [3, 6]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_NotIsNaN():
    """Ops: IsNaN, Not"""
    nodes = [
        helper.make_node('IsNaN', ['input'], ['temp_result']),
        helper.make_node('Not', ['temp_result'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'Test',
        inputs=[
            _vi('input', FLOAT, [1, 'N']),
        ],
        outputs=[
            _vi('output', BOOL, [1, 'N']),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_Pad():
    """Ops: Constant, Pad"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['pad_values'],
            value=_tensor('const_tensor', INT64, [6], [1, 0, 1, 0, 1, 2]),
        ),
        helper.make_node('Pad', ['input', 'pad_values'], ['output'], mode='constant'),
    ]
    graph = helper.make_graph(
        nodes,
        'PadGraph',
        inputs=[
            _vi('input', FLOAT, [1, 2, 2]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3, 5]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Pow():
    """Ops: Pow"""
    nodes = [
        helper.make_node('Pow', ['onnx::Pow_0', 'onnx::Pow_1'], ['2'], name='Pow_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Pow_0', FLOAT, [3]),
            _vi('onnx::Pow_1', FLOAT, [3]),
        ],
        outputs=[
            _vi('2', FLOAT, [3]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Pow_broadcast():
    """Ops: Pow"""
    nodes = [
        helper.make_node('Pow', ['onnx::Pow_0', 'onnx::Pow_1'], ['2'], name='Pow_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Pow_0', FLOAT, [1, 2, 3]),
            _vi('onnx::Pow_1', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('2', FLOAT, [1, 2, 3]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_RNNBatchwise():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R'],
            ['Y', 'Y_h'],
            activations=['Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=4,
            layout=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNBatchwise',
        inputs=[
            _vi('X', FLOAT, [3, 1, 2]),
            _vi('W', FLOAT, [1, 4, 2]),
            _vi('R', FLOAT, [1, 4, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 4]),
            _vi('Y_h', FLOAT, [3, 1, 4]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 4, 2], [
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
            ]),
            _tensor('R', FLOAT, [1, 4, 4], [
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
                0.05000000074505806, 0.05000000074505806, 0.05000000074505806, 0.05000000074505806,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNBidirectional():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
            ['Y', 'Y_h'],
            activations=['Tanh', 'Tanh'],
            clip=0.0,
            direction='bidirectional',
            hidden_size=4,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNBidirectional',
        inputs=[
            _vi('X', FLOAT, [3, 3, 2]),
            _vi('W', FLOAT, [2, 4, 2]),
            _vi('R', FLOAT, [2, 4, 4]),
            _vi('B', FLOAT, [2, 8]),
            _vi('sequence_lens', FLOAT, [3]),
            _vi('initial_h', FLOAT, [2, 3, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 2, 3, 4]),
            _vi('Y_h', FLOAT, [2, 3, 4]),
        ],
        initializer=[
            _tensor('W', FLOAT, [2, 4, 2], [
                1.1630799770355225, 2.212209939956665, 0.48380500078201294, 0.7740039825439453,
                0.2995629906654358, 1.0434399843215942, 0.15302500128746033, 1.1839300394058228,
                -1.1688100099563599, 1.8917100429534912, 1.5580699443817139, -1.2347400188446045,
                -0.5459449887275696, -1.7710299491882324, -2.3556299209594727, -0.4513840079307556,
            ]),
            _tensor('R', FLOAT, [2, 4, 4], [
                -0.264847993850708, -1.3031100034713745, 0.07120870053768158, 0.641979992389679,
                -2.7653799057006836, -0.6520739793777466, -0.7842749953269958, -1.767490029335022,
                -0.45067301392555237, -0.9179289937019348, -0.9666540026664734, 0.6508560180664062,
                0.285537987947464, -0.9098479747772217, -1.9045900106430054, -0.14092600345611572,
                -1.3713099956512451, 0.7806439995765686, 0.4410090148448944, 1.158560037612915,
                0.31329798698425293, 1.9676599502563477, -1.1199100017547607,
                -0.004409589804708958, 0.40762200951576233, 2.6056900024414062,
                -0.8409860134124756, 0.5856580138206482, 0.8232920169830322, -0.6968179941177368,
                1.1511499881744385, 0.15026900172233582,
            ]),
            _tensor('B', FLOAT, [2, 8], [
                -0.16102899610996246, -2.5899100303649902, 0.3397209942340851, -0.3166399896144867,
                0.049052998423576355, -1.8979500532150269, -0.32712098956108093,
                -0.1596280038356781, -0.18305400013923645, -0.9774590134620667,
                -1.0830899477005005, -0.01658809930086136, 1.9934899806976318, 1.3551299571990967,
                -0.6979780197143555, -0.7086179852485657,
            ]),
            _tensor('sequence_lens', FLOAT, [3], [3.0, 3.0, 3.0]),
            _tensor('initial_h', FLOAT, [2, 3, 4], [
                -0.37107500433921814, 0.2525329887866974, -1.4219499826431274, 0.39302998781204224,
                -0.4631119966506958, -1.0243799686431885, -0.5383989810943604, -2.2150800228118896,
                -1.4220999479293823, -0.14936499297618866, 1.2587000131607056, 1.3829400539398193,
                -0.0841611996293068, 1.456969976425171, 0.06793870031833649, 2.1154799461364746,
                -1.510509967803955, 1.5094799995422363, 0.20635099709033966, -0.9814450144767761,
                -0.22147700190544128, -0.2304839938879013, 0.4533129930496216, 0.7954760193824768,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNBidirectionalBatchwise():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'],
            ['Y', 'Y_h'],
            activations=['Tanh', 'Tanh'],
            clip=0.0,
            direction='bidirectional',
            hidden_size=4,
            layout=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNBidirectionalBatchwise',
        inputs=[
            _vi('X', FLOAT, [3, 3, 2]),
            _vi('W', FLOAT, [2, 4, 2]),
            _vi('R', FLOAT, [2, 4, 4]),
            _vi('B', FLOAT, [2, 8]),
            _vi('sequence_lens', FLOAT, [3]),
            _vi('initial_h', FLOAT, [3, 2, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 3, 2, 4]),
            _vi('Y_h', FLOAT, [3, 2, 4]),
        ],
        initializer=[
            _tensor('W', FLOAT, [2, 4, 2], [
                1.1630799770355225, 2.212209939956665, 0.48380500078201294, 0.7740039825439453,
                0.2995629906654358, 1.0434399843215942, 0.15302500128746033, 1.1839300394058228,
                -1.1688100099563599, 1.8917100429534912, 1.5580699443817139, -1.2347400188446045,
                -0.5459449887275696, -1.7710299491882324, -2.3556299209594727, -0.4513840079307556,
            ]),
            _tensor('R', FLOAT, [2, 4, 4], [
                -0.264847993850708, -1.3031100034713745, 0.07120870053768158, 0.641979992389679,
                -2.7653799057006836, -0.6520739793777466, -0.7842749953269958, -1.767490029335022,
                -0.45067301392555237, -0.9179289937019348, -0.9666540026664734, 0.6508560180664062,
                0.285537987947464, -0.9098479747772217, -1.9045900106430054, -0.14092600345611572,
                -1.3713099956512451, 0.7806439995765686, 0.4410090148448944, 1.158560037612915,
                0.31329798698425293, 1.9676599502563477, -1.1199100017547607,
                -0.004409589804708958, 0.40762200951576233, 2.6056900024414062,
                -0.8409860134124756, 0.5856580138206482, 0.8232920169830322, -0.6968179941177368,
                1.1511499881744385, 0.15026900172233582,
            ]),
            _tensor('B', FLOAT, [2, 8], [
                -0.16102899610996246, -2.5899100303649902, 0.3397209942340851, -0.3166399896144867,
                0.049052998423576355, -1.8979500532150269, -0.32712098956108093,
                -0.1596280038356781, -0.18305400013923645, -0.9774590134620667,
                -1.0830899477005005, -0.01658809930086136, 1.9934899806976318, 1.3551299571990967,
                -0.6979780197143555, -0.7086179852485657,
            ]),
            _tensor('sequence_lens', FLOAT, [3], [3.0, 3.0, 3.0]),
            _tensor('initial_h', FLOAT, [2, 3, 4], [
                -0.37107500433921814, 0.2525329887866974, -1.4219499826431274, 0.39302998781204224,
                -0.0841611996293068, 1.456969976425171, 0.06793870031833649, 2.1154799461364746,
                -0.4631119966506958, -1.0243799686431885, -0.5383989810943604, -2.2150800228118896,
                -1.510509967803955, 1.5094799995422363, 0.20635099709033966, -0.9814450144767761,
                -1.4220999479293823, -0.14936499297618866, 1.2587000131607056, 1.3829400539398193,
                -0.22147700190544128, -0.2304839938879013, 0.4533129930496216, 0.7954760193824768,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNDefaults():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B'],
            ['Y', 'Y_h'],
            activations=['Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=5,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNDefaults',
        inputs=[
            _vi('X', FLOAT, [3, 1, 3]),
            _vi('W', FLOAT, [1, 5, 3]),
            _vi('R', FLOAT, [1, 5, 5]),
            _vi('B', FLOAT, [1, 10]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 1, 5]),
            _vi('Y_h', FLOAT, [1, 1, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 5, 3], [
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
            ]),
            _tensor('R', FLOAT, [1, 5, 5], [
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582,
            ]),
            _tensor('B', FLOAT, [1, 10], [
                0.009999999776482582, 0.009999999776482582, 0.009999999776482582,
                0.009999999776482582, 0.009999999776482582, 0.0, 0.0, 0.0, 0.0, 0.0,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNSeqLength():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B'],
            ['Y', 'Y_h'],
            activations=['Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=5,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNSeqLength',
        inputs=[
            _vi('X', FLOAT, [2, 3, 3]),
            _vi('W', FLOAT, [1, 5, 3]),
            _vi('R', FLOAT, [1, 5, 5]),
            _vi('B', FLOAT, [1, 10]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 1, 3, 5]),
            _vi('Y_h', FLOAT, [1, 3, 5]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 5, 3], [
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
            ]),
            _tensor('R', FLOAT, [1, 5, 5], [
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164, 0.019999999552965164, 0.019999999552965164,
                0.019999999552965164,
            ]),
            _tensor('B', FLOAT, [1, 10], [
                0.03099999949336052, 0.03099999949336052, 0.03099999949336052, 0.03099999949336052,
                0.03099999949336052, 0.020999999716877937, 0.020999999716877937,
                0.020999999716877937, 0.020999999716877937, 0.020999999716877937,
            ]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNSequence():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B', 'sequence_lens'],
            ['Y', 'Y_h'],
            activations=['Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=6,
            layout=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNSequence',
        inputs=[
            _vi('X', FLOAT, [3, 3, 5]),
            _vi('W', FLOAT, [1, 6, 5]),
            _vi('R', FLOAT, [1, 6, 6]),
            _vi('B', FLOAT, [1, 12]),
            _vi('sequence_lens', FLOAT, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 1, 3, 6]),
            _vi('Y_h', FLOAT, [1, 3, 6]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 6, 5], [
                0.23690000176429749, 0.13459999859333038, 0.33169999718666077, -0.4821999967098236,
                -0.1362999975681305, 0.9419999718666077, -0.45019999146461487, -2.8173999786376953,
                0.2888999879360199, 1.6714999675750732, 0.29670000076293945, 1.679900050163269,
                -0.8342999815940857, 0.44929999113082886, 0.03700000047683716, -0.5325999855995178,
                1.1545000076293945, -1.6477999687194824, 0.777899980545044, -0.9257000088691711,
                -1.482200026512146, -0.8716999888420105, -0.017400000244379044, 2.06850004196167,
                -0.7620000243186951, 0.010499999858438969, -2.9377999305725098, 0.888700008392334,
                -0.9477999806404114, -1.5724999904632568,
            ]),
            _tensor('R', FLOAT, [1, 6, 6], [
                1.0134999752044678, -0.2632000148296356, -0.678600013256073, -1.017899990081787,
                -2.1319000720977783, -0.003599999938160181, 1.9585000276565552, 1.1375000476837158,
                2.121000051498413, 0.6409000158309937, -2.050299882888794, -2.4921000003814697,
                0.5932000279426575, 1.5161000490188599, -0.7768999934196472, 0.2849000096321106,
                0.20720000565052032, -0.3086000084877014, -0.965499997138977, 0.9178000092506409,
                -0.4291999936103821, -1.5053999423980713, -0.7396000027656555, 0.8928999900817871,
                -0.18359999358654022, -1.6291999816894531, 1.0712000131607056, 0.37700000405311584,
                0.17790000140666962, -1.1167000532150269, -0.6861000061035156, 1.2390999794006348,
                -0.5447999835014343, -0.3880999982357025, -0.5164999961853027, 0.012799999676644802,
            ]),
            _tensor('B', FLOAT, [1, 12], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _tensor('sequence_lens', FLOAT, [3], [3.0, 2.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RNNSequenceBatchwise():
    """Ops: RNN"""
    nodes = [
        helper.make_node(
            'RNN',
            ['X', 'W', 'R', 'B', 'sequence_lens'],
            ['Y', 'Y_h'],
            activations=['Tanh'],
            clip=0.0,
            direction='forward',
            hidden_size=6,
            layout=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RNNSequenceBatchwise',
        inputs=[
            _vi('X', FLOAT, [3, 3, 5]),
            _vi('W', FLOAT, [1, 6, 5]),
            _vi('R', FLOAT, [1, 6, 6]),
            _vi('B', FLOAT, [1, 12]),
            _vi('sequence_lens', FLOAT, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3, 3, 1, 6]),
            _vi('Y_h', FLOAT, [3, 1, 6]),
        ],
        initializer=[
            _tensor('W', FLOAT, [1, 6, 5], [
                0.23690000176429749, 0.13459999859333038, 0.33169999718666077, -0.4821999967098236,
                -0.1362999975681305, 0.9419999718666077, -0.45019999146461487, -2.8173999786376953,
                0.2888999879360199, 1.6714999675750732, 0.29670000076293945, 1.679900050163269,
                -0.8342999815940857, 0.44929999113082886, 0.03700000047683716, -0.5325999855995178,
                1.1545000076293945, -1.6477999687194824, 0.777899980545044, -0.9257000088691711,
                -1.482200026512146, -0.8716999888420105, -0.017400000244379044, 2.06850004196167,
                -0.7620000243186951, 0.010499999858438969, -2.9377999305725098, 0.888700008392334,
                -0.9477999806404114, -1.5724999904632568,
            ]),
            _tensor('R', FLOAT, [1, 6, 6], [
                1.0134999752044678, -0.2632000148296356, -0.678600013256073, -1.017899990081787,
                -2.1319000720977783, -0.003599999938160181, 1.9585000276565552, 1.1375000476837158,
                2.121000051498413, 0.6409000158309937, -2.050299882888794, -2.4921000003814697,
                0.5932000279426575, 1.5161000490188599, -0.7768999934196472, 0.2849000096321106,
                0.20720000565052032, -0.3086000084877014, -0.965499997138977, 0.9178000092506409,
                -0.4291999936103821, -1.5053999423980713, -0.7396000027656555, 0.8928999900817871,
                -0.18359999358654022, -1.6291999816894531, 1.0712000131607056, 0.37700000405311584,
                0.17790000140666962, -1.1167000532150269, -0.6861000061035156, 1.2390999794006348,
                -0.5447999835014343, -0.3880999982357025, -0.5164999961853027, 0.012799999676644802,
            ]),
            _tensor('B', FLOAT, [1, 12], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            _tensor('sequence_lens', FLOAT, [3], [3.0, 2.0, 1.0]),
        ],
    )
    return _model(graph, opset=14, ir_version=7)


def make_RandomNormal():
    """Ops: RandomNormal"""
    nodes = [
        helper.make_node(
            'RandomNormal',
            [],
            ['output'],
            mean=1.0,
            scale=3.0,
            seed=111,
            shape=[2, 3],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RandomNormal',
        inputs=[
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_RandomUniform():
    """Ops: RandomUniform"""
    nodes = [
        helper.make_node(
            'RandomUniform',
            [],
            ['output'],
            high=20.0,
            low=10.0,
            seed=111,
            shape=[2, 3],
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'RandomUniform',
        inputs=[
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_RangeFloat():
    """Ops: Range"""
    nodes = [
        helper.make_node('Range', ['start', 'limit', 'delta'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Range',
        inputs=[
            _vi('start', FLOAT, [1]),
            _vi('limit', FLOAT, [1]),
            _vi('delta', FLOAT, [1]),
        ],
        outputs=[
            _vi('Y', FLOAT, ['output_size']),
        ],
    )
    return _model(graph, opset=19, ir_version=9)


def make_RangeInt():
    """Ops: Range"""
    nodes = [
        helper.make_node('Range', ['start', 'limit', 'delta'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Range',
        inputs=[
            _vi('start', INT64, [1]),
            _vi('limit', INT64, [1]),
            _vi('delta', INT64, [1]),
        ],
        outputs=[
            _vi('Y', INT64, ['output_size']),
        ],
    )
    return _model(graph, opset=19, ir_version=9)


def make_Reciprocal():
    """Ops: Reciprocal"""
    nodes = [
        helper.make_node('Reciprocal', ['X'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Reciprocal',
        inputs=[
            _vi('X', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_ReduceMean():
    """Ops: ReduceMean"""
    nodes = [
        helper.make_node(
            'ReduceMean',
            ['onnx::ReduceMean_0'],
            ['1'],
            name='ReduceMean_0',
            axes=[1],
            keepdims=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::ReduceMean_0', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('1', FLOAT, [1, 3]),
        ],
    )
    return _model(graph, opset=13, ir_version=7, producer_name='pytorch', producer_version='1.12.1')


def make_ReduceMean_kFirst():
    """Ops: ReduceMean"""
    nodes = [
        helper.make_node('ReduceMean', ['X'], ['Y'], axes=[0], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        'ReduceMean_kFirst',
        inputs=[
            _vi('X', FLOAT, [3, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [4]),
        ],
    )
    return _model(graph, opset=13, ir_version=13)


def make_ReduceProd():
    """Ops: ReduceProd"""
    nodes = [
        helper.make_node(
            'ReduceProd',
            ['onnx::ReduceProd_0'],
            ['1'],
            name='ReduceProd_0',
            axes=[1],
            keepdims=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'torch_jit',
        inputs=[
            _vi('onnx::ReduceProd_0', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('1', FLOAT, [1, 3]),
        ],
    )
    return _model(graph, opset=13, ir_version=7, producer_name='pytorch', producer_version='1.12.1')


def make_ReduceSum():
    """Ops: ReduceSum"""
    nodes = [
        helper.make_node('ReduceSum', ['input'], ['output'], axes=[0, 1, 2], keepdims=1),
    ]
    graph = helper.make_graph(
        nodes,
        'ReduceSumGraph',
        inputs=[
            _vi('input', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [1, 1, 1]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_ReduceSumSquare():
    """Ops: ReduceSumSquare"""
    nodes = [
        helper.make_node('ReduceSumSquare', ['input'], ['output'], axes=[2], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        'ReduceSumSquareGraph',
        inputs=[
            _vi('input', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [1, 2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_ScatterElements():
    """Ops: ScatterElements"""
    nodes = [
        helper.make_node('ScatterElements', ['data', 'indices', 'updates'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [3, 3]),
            _vi('indices', INT64, [2, 3]),
            _vi('updates', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [3, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_ScatterND_1():
    """Ops: ScatterND"""
    nodes = [
        helper.make_node('ScatterND', ['data', 'indices', 'updates'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [5]),
            _vi('indices', INT64, [3, 1]),
            _vi('updates', FLOAT, [3]),
        ],
        outputs=[
            _vi('output', FLOAT, [5]),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_ScatterND_2():
    """Ops: ScatterND"""
    nodes = [
        helper.make_node('ScatterND', ['data', 'indices', 'updates'], ['output'], reduction='add'),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [3, 2]),
            _vi('indices', INT64, [2, 1]),
            _vi('updates', FLOAT, [2, 2]),
        ],
        outputs=[
            _vi('output', FLOAT, [3, 2]),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_ScatterND_3():
    """Ops: ScatterND"""
    nodes = [
        helper.make_node('ScatterND', ['data', 'indices', 'updates'], ['output'], reduction='mul'),
    ]
    graph = helper.make_graph(
        nodes,
        'TestGraph',
        inputs=[
            _vi('data', FLOAT, [2, 2]),
            _vi('indices', INT64, [2, 2]),
            _vi('updates', FLOAT, [2]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 2]),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_Shape():
    """Ops: Shape, ConstantOfShape, Mul"""
    nodes = [
        helper.make_node('Shape', ['input'], ['out_shape']),
        helper.make_node(
            'ConstantOfShape',
            ['out_shape'],
            ['scale_values'],
            value=_tensor('value', FLOAT, [1], [2.0]),
        ),
        helper.make_node('Mul', ['input', 'scale_values'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'ShapeGraph',
        inputs=[
            _vi('input', FLOAT, [1, 2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [1, 2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Sin():
    """Ops: Sin"""
    nodes = [
        helper.make_node('Sin', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'sin_test',
        inputs=[
            _vi('input', FLOAT, [3, 4]),
        ],
        outputs=[
            _vi('output', FLOAT, [3, 4]),
        ],
    )
    return _model(graph, opset=7, ir_version=10, producer_name='onnx-example')


def make_Slice():
    """Ops: Slice"""
    nodes = [
        helper.make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Slice',
        inputs=[
            _vi('x', FLOAT, [20, 10, 5]),
        ],
        outputs=[
            _vi('y', FLOAT, [3, 10, 5]),
        ],
        initializer=[
            _tensor('starts', INT64, [2], [0, 0]),
            _tensor('ends', INT64, [2], [3, 10]),
            _tensor('axes', INT64, [2], [0, 1]),
            _tensor('steps', INT64, [2], [1, 1]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='Slice')


def make_Slice_Default_Axis():
    """Ops: Slice"""
    nodes = [
        helper.make_node('Slice', ['x', 'starts', 'ends'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Slice',
        inputs=[
            _vi('x', FLOAT, [20, 10, 5]),
        ],
        outputs=[
            _vi('y', FLOAT, [20, 10, 1]),
        ],
        initializer=[
            _tensor('starts', INT64, [3], [0, 0, 3]),
            _tensor('ends', INT64, [3], [20, 10, 4]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='Slice_Default_Axis')


def make_Slice_Default_Steps():
    """Ops: Slice"""
    nodes = [
        helper.make_node('Slice', ['x', 'starts', 'ends', 'axes'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Slice',
        inputs=[
            _vi('x', FLOAT, [20, 10, 5]),
        ],
        outputs=[
            _vi('y', FLOAT, [20, 10, 1]),
        ],
        initializer=[
            _tensor('starts', INT64, [3], [0, 0, 3]),
            _tensor('ends', INT64, [3], [20, 10, 4]),
            _tensor('axes', INT64, [3], [0, 1, 2]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='Slice_Default_Steps')


def make_Slice_Neg():
    """Ops: Slice"""
    nodes = [
        helper.make_node('Slice', ['x', 'starts', 'ends', 'axes', 'steps'], ['y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Slice',
        inputs=[
            _vi('x', FLOAT, [20, 10, 5]),
        ],
        outputs=[
            _vi('y', FLOAT, [20, 9, 5]),
        ],
        initializer=[
            _tensor('starts', INT64, [1], [0]),
            _tensor('ends', INT64, [1], [-1]),
            _tensor('axes', INT64, [1], [1]),
            _tensor('steps', INT64, [1], [1]),
        ],
    )
    return _model(graph, opset=19, ir_version=9, producer_name='Slice_Neg')


def make_Softmax1d():
    """Ops: Softmax"""
    nodes = [
        helper.make_node('Softmax', ['X'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Softmax',
        inputs=[
            _vi('X', FLOAT, [3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Softmax2d():
    """Ops: Softmax"""
    nodes = [
        helper.make_node('Softmax', ['X'], ['Y'], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        'Softmax',
        inputs=[
            _vi('X', FLOAT, [1, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [1, 3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Softmax3d():
    """Ops: Softmax"""
    nodes = [
        helper.make_node('Softmax', ['X'], ['Y'], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        'Softmax',
        inputs=[
            _vi('X', FLOAT, [2, 3, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Softmax4d():
    """Ops: Softmax"""
    nodes = [
        helper.make_node('Softmax', ['X'], ['Y'], axis=-2),
    ]
    graph = helper.make_graph(
        nodes,
        'Softmax',
        inputs=[
            _vi('X', FLOAT, [2, 3, 4, 2]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3, 4, 2]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Softplus():
    """Ops: Softplus"""
    nodes = [
        helper.make_node('Softplus', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'Abs',
        inputs=[
            _vi('input', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=25, ir_version=13, producer_name='onnx-example')


def make_Split_0():
    """Ops: Constant, Split"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['split_values'],
            value=_tensor('split', INT64, [2], [1, 1]),
        ),
        helper.make_node('Split', ['input', 'split_values'], ['output1', 'output2']),
    ]
    graph = helper.make_graph(
        nodes,
        'SplitGraph',
        inputs=[
            _vi('input', FLOAT, [2, 2, 3]),
        ],
        outputs=[
            _vi('output1', FLOAT, [1, 2, 3]),
            _vi('output2', FLOAT, [1, 2, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Split_1():
    """Ops: Constant, Split"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['split_values'],
            value=_tensor('split', INT64, [2], [1, 1]),
        ),
        helper.make_node('Split', ['input', 'split_values'], ['output1', 'output2'], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        'SplitGraph',
        inputs=[
            _vi('input', FLOAT, [2, 2, 3]),
        ],
        outputs=[
            _vi('output1', FLOAT, [2, 1, 3]),
            _vi('output2', FLOAT, [2, 1, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Split_2():
    """Ops: Split"""
    nodes = [
        helper.make_node('Split', ['input'], ['output1', 'output2'], axis=2, num_outputs=2),
    ]
    graph = helper.make_graph(
        nodes,
        'SplitGraph',
        inputs=[
            _vi('input', FLOAT, [2, 2, 3]),
        ],
        outputs=[
            _vi('output1', FLOAT, [2, 2, 2]),
            _vi('output2', FLOAT, [2, 2, 1]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


def make_Sqrt():
    """Ops: Sqrt"""
    nodes = [
        helper.make_node('Sqrt', ['X'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Sqrt',
        inputs=[
            _vi('X', FLOAT, [2, 3]),
        ],
        outputs=[
            _vi('Y', FLOAT, [2, 3]),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Sub():
    """Ops: Sub"""
    nodes = [
        helper.make_node('Sub', ['onnx::Sub_0', 'onnx::Sub_1'], ['2'], name='Sub_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Sub_0', FLOAT, [2]),
            _vi('onnx::Sub_1', FLOAT, [2]),
        ],
        outputs=[
            _vi('2', FLOAT, [2]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_SumMultidirectionalBroadcast():
    """Ops: Sum"""
    nodes = [
        helper.make_node('Sum', ['A', 'B', 'C'], ['Y']),
    ]
    graph = helper.make_graph(
        nodes,
        'Sum',
        inputs=[
            _vi('A', FLOAT, [3, 1]),
            _vi('B', FLOAT, [2, 3, 1]),
            _vi('C', FLOAT, [1, 4]),
        ],
        outputs=[
            _vi('Y', FLOAT, []),
        ],
    )
    return _model(graph, opset=17, ir_version=8)


def make_Swish():
    """Ops: Swish"""
    nodes = [
        helper.make_node('Swish', ['input'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'Swish',
        inputs=[
            _vi('input', FLOAT, [6]),
        ],
        outputs=[
            _vi('output', FLOAT, [6]),
        ],
    )
    return _model(graph, opset=24, ir_version=13)


def make_Tanh():
    """Ops: Tanh"""
    nodes = [
        helper.make_node('Tanh', ['onnx::Tanh_0'], ['1'], name='Tanh_0'),
    ]
    graph = helper.make_graph(
        nodes,
        'torch-jit-export',
        inputs=[
            _vi('onnx::Tanh_0', FLOAT, [24]),
        ],
        outputs=[
            _vi('1', FLOAT, [24]),
        ],
    )
    return _model(graph, opset=9, ir_version=4, producer_name='pytorch', producer_version='1.11.0')


def make_Tile5D():
    """Ops: Tile"""
    nodes = [
        helper.make_node('Tile', ['input', 'repeats'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'TileGraph',
        inputs=[
            _vi('input', FLOAT, [2, 2, 2, 3, 3]),
        ],
        outputs=[
            _vi('output', FLOAT, [4, 2, 4, 3, 9]),
        ],
        initializer=[
            _tensor('repeats', INT64, [5], [2, 1, 2, 1, 3]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='tile-model')


def make_TopK():
    """Ops: Constant, TopK"""
    nodes = [
        helper.make_node(
            'Constant',
            [],
            ['/Constant_output_0'],
            name='/Constant',
            value=_tensor('', INT64, [1], [5]),
        ),
        helper.make_node(
            'TopK',
            ['onnx::TopK_0', '/Constant_output_0'],
            ['4', '5'],
            name='/TopK',
            axis=-1,
            largest=1,
            sorted=1,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        'main_graph',
        inputs=[
            _vi('onnx::TopK_0', FLOAT, [9]),
        ],
        outputs=[
            _vi('4', FLOAT, [5]),
            _vi('5', INT64, [5]),
        ],
    )
    return _model(graph, opset=17, ir_version=8, producer_name='pytorch', producer_version='2.3.0')


def make_Where():
    """Ops: Where"""
    nodes = [
        helper.make_node('Where', ['cond', 'inputA', 'inputB'], ['output']),
    ]
    graph = helper.make_graph(
        nodes,
        'PadGraph',
        inputs=[
            _vi('inputA', FLOAT, [1, 2]),
            _vi('inputB', FLOAT, [3, 2]),
            _vi('cond', BOOL, [3, 1]),
        ],
        outputs=[
            _vi('output', FLOAT, [3, 2]),
        ],
    )
    return _model(graph, opset=21, ir_version=10, producer_name='onnx-example')


MODELS = {
    'Abs': make_Abs,
    'Add': make_Add,
    'AddBroadcast1': make_AddBroadcast1,
    'AddBroadcast2': make_AddBroadcast2,
    'AddBroadcast3': make_AddBroadcast3,
    'AddBroadcast4': make_AddBroadcast4,
    'AddBroadcast5': make_AddBroadcast5,
    'AddBroadcast6': make_AddBroadcast6,
    'AddBroadcast7': make_AddBroadcast7,
    'AvgPool': make_AvgPool,
    'Cast': make_Cast,
    'Clip': make_Clip,
    'Comparison_broadcast': make_Comparison_broadcast,
    'Comparison_broadcast_3d': make_Comparison_broadcast_3d,
    'ComplexTopK': make_ComplexTopK,
    'Concat_0D': make_Concat_0D,
    'Constant': make_Constant,
    'ConvAddRelu': make_ConvAddRelu,
    'ConvTranspose1d': make_ConvTranspose1d,
    'ConvTranspose2d': make_ConvTranspose2d,
    'ConvTransposeBias2d': make_ConvTransposeBias2d,
    'ConvTransposeBias2dBatched': make_ConvTransposeBias2dBatched,
    'ConvWithAsymmetricPadding': make_ConvWithAsymmetricPadding,
    'ConvWithAutopadSameLower': make_ConvWithAutopadSameLower,
    'ConvWithAutopadSameUpper': make_ConvWithAutopadSameUpper,
    'ConvWithDilation': make_ConvWithDilation,
    'ConvWithDynShapeStride': make_ConvWithDynShapeStride,
    'ConvWithPadding': make_ConvWithPadding,
    'ConvWithStridesNoPadding': make_ConvWithStridesNoPadding,
    'ConvWithStridesPadding': make_ConvWithStridesPadding,
    'ConvWithoutPadding': make_ConvWithoutPadding,
    'Cos': make_Cos,
    'Div': make_Div,
    'Einsum_3': make_Einsum_3,
    'Einsum_4': make_Einsum_4,
    'Einsum_dotprod': make_Einsum_dotprod,
    'Einsum_matmul': make_Einsum_matmul,
    'Elu': make_Elu,
    'EluAlpha': make_EluAlpha,
    'Equal': make_Equal,
    'Erf': make_Erf,
    'Exp': make_Exp,
    'ExpandDiffSize': make_ExpandDiffSize,
    'ExpandSameSize': make_ExpandSameSize,
    'EyeLike': make_EyeLike,
    'FMod_ConstantFolding': make_FMod_ConstantFolding,
    'GRUBatchwise': make_GRUBatchwise,
    'GRUBidirectional': make_GRUBidirectional,
    'GRUDefaults': make_GRUDefaults,
    'GRUInitialBias': make_GRUInitialBias,
    'GRUSeqLength': make_GRUSeqLength,
    'Gather2d': make_Gather2d,
    'GatherAxis0': make_GatherAxis0,
    'GatherAxis1': make_GatherAxis1,
    'GatherAxis2': make_GatherAxis2,
    'GatherAxis3': make_GatherAxis3,
    'GatherND_1': make_GatherND_1,
    'GatherND_2': make_GatherND_2,
    'GatherND_3': make_GatherND_3,
    'GatherNegativeIndices': make_GatherNegativeIndices,
    'Gelu': make_Gelu,
    'Greater': make_Greater,
    'GreaterOrEqual': make_GreaterOrEqual,
    'HardSigmoid': make_HardSigmoid,
    'HardSwish': make_HardSwish,
    'InstanceNormalization': make_InstanceNormalization,
    'InstanceNormalization3d': make_InstanceNormalization3d,
    'InstanceNormalizationEpsilon': make_InstanceNormalizationEpsilon,
    'IsInf': make_IsInf,
    'LSTMBatchwise': make_LSTMBatchwise,
    'LSTMBidirectional': make_LSTMBidirectional,
    'LSTMDefaults': make_LSTMDefaults,
    'LSTMInitialBias': make_LSTMInitialBias,
    'LSTMPeepholes': make_LSTMPeepholes,
    'LayerNormalization2d': make_LayerNormalization2d,
    'LayerNormalization4d': make_LayerNormalization4d,
    'Less': make_Less,
    'LessOrEqual': make_LessOrEqual,
    'LinearWithLeakyRelu': make_LinearWithLeakyRelu,
    'LinearWithSelu': make_LinearWithSelu,
    'LinearWithSigmoid': make_LinearWithSigmoid,
    'Linear_16': make_Linear_16,
    'Linear_32': make_Linear_32,
    'Linear_64': make_Linear_64,
    'Log': make_Log,
    'MatMul_Stacked': make_MatMul_Stacked,
    'MatMul_Stacked2': make_MatMul_Stacked2,
    'Max': make_Max,
    'MaxMultidirectionalBroadcast': make_MaxMultidirectionalBroadcast,
    'MaxPool1d': make_MaxPool1d,
    'MaxPool2d': make_MaxPool2d,
    'MaxPool2d_AsymPad': make_MaxPool2d_AsymPad,
    'MaxPool2d_CeilMode': make_MaxPool2d_CeilMode,
    'MaxPool3d': make_MaxPool3d,
    'MeanMultidirectionalBroadcast': make_MeanMultidirectionalBroadcast,
    'MinMultidirectionalBroadcast': make_MinMultidirectionalBroadcast,
    'Mod_ConstantFolding': make_Mod_ConstantFolding,
    'Mul': make_Mul,
    'Neg': make_Neg,
    'NonZero': make_NonZero,
    'NonZero_Constant': make_NonZero_Constant,
    'NotIsNaN': make_NotIsNaN,
    'Pad': make_Pad,
    'Pow': make_Pow,
    'Pow_broadcast': make_Pow_broadcast,
    'RNNBatchwise': make_RNNBatchwise,
    'RNNBidirectional': make_RNNBidirectional,
    'RNNBidirectionalBatchwise': make_RNNBidirectionalBatchwise,
    'RNNDefaults': make_RNNDefaults,
    'RNNSeqLength': make_RNNSeqLength,
    'RNNSequence': make_RNNSequence,
    'RNNSequenceBatchwise': make_RNNSequenceBatchwise,
    'RandomNormal': make_RandomNormal,
    'RandomUniform': make_RandomUniform,
    'RangeFloat': make_RangeFloat,
    'RangeInt': make_RangeInt,
    'Reciprocal': make_Reciprocal,
    'ReduceMean': make_ReduceMean,
    'ReduceMean_kFirst': make_ReduceMean_kFirst,
    'ReduceProd': make_ReduceProd,
    'ReduceSum': make_ReduceSum,
    'ReduceSumSquare': make_ReduceSumSquare,
    'ScatterElements': make_ScatterElements,
    'ScatterND_1': make_ScatterND_1,
    'ScatterND_2': make_ScatterND_2,
    'ScatterND_3': make_ScatterND_3,
    'Shape': make_Shape,
    'Sin': make_Sin,
    'Slice': make_Slice,
    'Slice_Default_Axis': make_Slice_Default_Axis,
    'Slice_Default_Steps': make_Slice_Default_Steps,
    'Slice_Neg': make_Slice_Neg,
    'Softmax1d': make_Softmax1d,
    'Softmax2d': make_Softmax2d,
    'Softmax3d': make_Softmax3d,
    'Softmax4d': make_Softmax4d,
    'Softplus': make_Softplus,
    'Split_0': make_Split_0,
    'Split_1': make_Split_1,
    'Split_2': make_Split_2,
    'Sqrt': make_Sqrt,
    'Sub': make_Sub,
    'SumMultidirectionalBroadcast': make_SumMultidirectionalBroadcast,
    'Swish': make_Swish,
    'Tanh': make_Tanh,
    'Tile5D': make_Tile5D,
    'TopK': make_TopK,
    'Where': make_Where,
}


# ===========================================================================
# Reference data for the value-based unit tests
# ===========================================================================
#
# TEST_INPUTS defines the inputs that TestCustomModelsFromONNX.cxx (and
# TestCladAutodiff.cxx) feed to each model, one numpy array per graph input.
# The expected outputs are computed here with onnx's ReferenceEvaluator (or
# with the numpy fallbacks in EXPECTED_OVERRIDES below) and written next to
# the generated models as references/<Name>.ref, from where the tests read
# both the inputs and the expected outputs at runtime. Models without an
# entry have their expectations hardcoded in the test source instead.


def f32(vals, shape=None):
    a = np.asarray(vals, np.float32)
    return a.reshape(shape) if shape is not None else a


def i64(vals, shape=None):
    a = np.asarray(vals, np.int64)
    return a.reshape(shape) if shape is not None else a


def rand_f32(seed, shape):
    """Deterministic standard-normal random tensor."""
    return np.random.RandomState(seed).randn(*shape).astype(np.float32)


TEST_INPUTS = {
    'Add': [
        f32([1.0, 2.0], (2,)),
        f32([0.0, 1.0], (2,)),
    ],
    'AddBroadcast1': [
        f32([-0.7802330255508423, -1.3402948379516602, -3.014829397201538, 0.5364136099815369, -1.2259478569030762], (5,)),
        f32([1.0626695156097412, 0.43842875957489014, 1.2247647047042847, 0.7976327538490295, 0.9868820905685425, 0.2526761293411255, 0.4487488269805908, 0.31516772508621216, -0.7877119779586792, 0.6456566452980042, 0.5045059323310852, -0.41265228390693665, -0.22474539279937744, -0.22362373769283295, 0.005096740089356899, 0.16927210986614227, 1.0675697326660156, -0.8163477182388306, 0.8846774697303772, 0.7890205979347229], (4, 5)),
    ],
    'AddBroadcast2': [
        f32([0.6008180379867554, 0.565757691860199, -0.5840851068496704, -1.5082775354385376, 1.2396254539489746], (5,)),
        f32([-1.2251673936843872, -2.503737449645996, -0.614517331123352, 0.44316595792770386, 0.004092322196811438, 1.4352006912231445, -0.8375269174575806, 1.1876263618469238, -1.42122220993042, 0.3771233558654785, -0.616450846195221, 1.966413140296936, -2.035682201385498, -0.53670334815979, -2.2214934825897217, -1.5829707384109497, -1.2514921426773071, 0.6506291031837463, 2.06339693069458, 0.6022816300392151, -0.5390340089797974, -1.2628082036972046, 0.7877674698829651, 0.10825152695178986, 2.3282978534698486, -1.5089000463485718, -0.5955929160118103, -0.0920059084892273, 1.6322861909866333, 1.946860671043396, 0.7456556558609009, 0.3869551122188568, -1.832051157951355, -1.1573481559753418, 0.03800858184695244, -0.21694916486740112, -0.23516549170017242, 0.2181714028120041, 0.061358895152807236, -0.8570862412452698, -2.0186426639556885, -1.6137357950210571, -2.0205025672912598, -0.32505208253860474, -0.10711464285850525, 0.46847009658813477, 0.19955800473690033, -1.9463766813278198, 0.24790054559707642, 0.7761988043785095, -0.19873686134815216, -2.00885009765625, 1.4684786796569824, 0.9610288143157959, -0.008149653673171997, 0.4633333384990692, -0.1113162413239479, 1.8204692602157593, -0.10051906853914261, 2.405775308609009, 2.5781426429748535, -1.5141286849975586, -0.06480903923511505, 0.9229392409324646, -1.314860463142395, 0.36738714575767517, -0.002170204883441329, -0.47474405169487, -0.6289427280426025, -1.317047357559204, -0.6206338405609131, -0.49025020003318787, -0.21248511970043182, -0.023678667843341827, 0.028880996629595757, -0.7447777986526489, 0.013009180314838886, -1.6810555458068848, 0.08222470432519913, -1.1493949890136719, -1.575654149055481, -0.7993866801261902, -0.4064111113548279, 1.0935839414596558, 1.5832337141036987, -0.08151749521493912, -0.0909925028681755, 2.3559670448303223, -0.06853648275136948, 0.4128839373588562, 0.500495433807373, -1.484426498413086, -0.5193490386009216, 0.3810258209705353, -0.10618859529495239, 0.2839215397834778, 1.1321500539779663, 1.2155804634094238, -1.0466749668121338, -0.9411510825157166, -0.04043630510568619, 1.455543041229248, 0.16402567923069, -0.33469337224960327, 1.2770131826400757, 0.8647446036338806, 1.0962142944335938, -1.0656343698501587, -1.5563756227493286, 2.143430471420288, 0.4696103632450104, 0.9091355800628662, -0.6206033825874329, -1.0423543453216553, -1.329746961593628, -0.13596804440021515, 0.9624383449554443, 1.134135127067566, -0.9246122241020203, -2.2613234519958496], (2, 3, 4, 5)),
    ],
    'AddBroadcast3': [
        f32([0.13225243985652924, -0.4780140519142151, -1.470346212387085, 0.8778636455535889, -0.5138850212097168, 0.7701201438903809, 0.994074821472168, -0.4101419746875763, 1.7650624513626099, 1.241428017616272], (2, 1, 1, 5)),
        f32([-0.7990003824234009, 1.2677446603775024, 0.10287351161241531, -0.007047129794955254, 0.19927170872688293, 1.7712593078613281, 0.2339390069246292, -0.751605749130249, -0.4098702073097229, 0.02957325056195259, 2.487703800201416, 2.724266767501831, 0.16116267442703247, 0.13580884039402008, -1.3455098867416382, 1.0834174156188965, -0.5723267793655396, -0.27434247732162476, 2.2975919246673584, 0.7250648140907288, -0.3598426282405853, -1.4755396842956543, 0.46544721722602844, 0.4530450701713562, 0.393509179353714, 0.2533503770828247, -2.154552698135376, 0.5859283208847046, 0.09075859934091568, 1.328303575515747, 2.1687653064727783, -1.315091609954834, -0.7790181636810303, 1.7297074794769287, 0.8941051959991455, 1.1889108419418335, 0.5837250351905823, -0.6117035150527954, -0.8382923007011414, 0.6391794681549072, 0.6662607789039612, -1.0766762495040894, 0.014115190133452415, -0.6708264946937561, -0.04556865990161896, -0.049491479992866516, -1.8707592487335205, 0.255876362323761, 0.1471511423587799, -0.7458451390266418, -1.1937352418899536, -1.5214205980300903, -0.9252294301986694, -0.9812653064727783, -0.07535745948553085, -1.4692507982254028, -0.08861242234706879, 0.6495186686515808, -0.1691899448633194, 0.8701536059379578, 0.5768899321556091, 1.3629382848739624, 1.282568335533142, 0.3924553692340851, 0.43308472633361816, 0.8452982902526855, -0.5668654441833496, -0.8479184508323669, -0.11286944150924683, 0.6085797548294067, -0.7951951026916504, -0.2049192488193512, -1.529517412185669, -0.3903006315231323, -2.7616076469421387, 0.09055905789136887, -0.991420328617096, 0.3348078429698944, -1.0999988317489624, 1.3614935874938965, 0.18557575345039368, 0.554069995880127, 1.2316406965255737, -0.23469014465808868, -1.3727471828460693, 1.80717933177948, 1.429667592048645, 0.7207739353179932, -0.09774938970804214, 1.1206538677215576, -0.5151561498641968, -0.9527944922447205, 0.8764696717262268, -0.5944010019302368, -0.12440208345651627, -0.710966944694519, -0.630127489566803, 0.5172616839408875, 1.2372664213180542, 1.5625547170639038, -0.9446976184844971, -0.381147563457489, -0.4202176034450531, -0.5892148613929749, -0.7143963575363159, 0.04793575033545494, -2.042145252227783, -0.45765405893325806, -1.1230720281600952, 0.9072713851928711, 0.9627283215522766, 0.5430320501327515, -0.8497303128242493, 0.2878032922744751, 0.17027853429317474, -0.11893711239099503, -1.2241463661193848, -1.6274759769439697, 0.5326449871063232, 0.5348359942436218], (2, 3, 4, 5)),
    ],
    'AddBroadcast4': [
        f32([1.9430140256881714, 0.40606817603111267], (2, 1)),
        f32([0.5089889168739319, -0.2782992124557495, -0.6876162886619568, 0.3318638205528259, 0.5791553258895874, 0.40685799717903137, 1.420383334159851, 0.19857093691825867], (2, 4)),
    ],
    'AddBroadcast5': [
        f32([-0.45616137981414795, -0.05853134021162987, 1.0956422090530396, 0.9588031768798828, 0.9499531984329224, -0.3586410582065582, 1.0857089757919312, 0.6028053164482117], (2, 1, 4)),
        f32([1.6978745460510254, 1.1064167022705078, 2.197551727294922, 0.06709206104278564, 0.0457230806350708, -2.1450436115264893, -0.4773070216178894, 0.15205423533916473, -0.2515922486782074, -0.07529807090759277, 0.517436683177948, 0.08267594873905182, 0.3401562571525574, 0.0946023091673851, -1.166089653968811, -0.2346605807542801, -0.5520268082618713, -0.13844847679138184, 0.5305575728416443, 0.1706864833831787, -0.4949127733707428, -1.4246270656585693, -0.9997391104698181, -0.257132887840271], (2, 3, 4)),
    ],
    'AddBroadcast6': [
        f32([1.0549867153167725, -1.6431103944778442, 0.11925146728754044, -1.597557783126831, -0.014453129842877388, -0.6944054365158081, -0.12011280655860901, 0.005393229890614748, -0.16923530399799347, 2.3453359603881836, 1.302680492401123, 0.4569944441318512], (2, 1, 3, 1, 2)),
        f32([0.03162163123488426, 1.363404393196106, -0.347364604473114, -0.7185632586479187, 0.40669968724250793, -0.3759573996067047, 0.22234952449798584, 1.6956379413604736, 0.9145916700363159, -0.020812150090932846, -1.6489421129226685, -0.011892610229551792, 0.5803133845329285, -0.11880190670490265, 0.7009931802749634, -0.3742424249649048, -0.23980526626110077, -0.031784068793058395, -0.27969110012054443, 0.018956879153847694, 1.3211175203323364, 0.021139059215784073, 0.514503002166748, -1.4176076650619507, -0.1922055333852768, 0.23529522120952606, 0.9519990682601929, -1.3897144794464111, -0.7583696246147156, -0.9095695614814758, -0.13006828725337982, -0.6439045667648315, -0.0808228999376297, 0.7913475632667542, 1.006848692893982, -1.438180923461914, -0.14550620317459106, -0.3363551199436188, -0.6185612082481384, -0.4928140640258789, -1.1294726133346558, 1.6181882619857788, -0.05826431140303612, -1.4780218601226807, 0.2563738226890564, -0.15478579699993134, 2.507887840270996, 0.3089805841445923], (2, 2, 3, 2, 2)),
    ],
    'AddBroadcast7': [
        f32([-0.4216483533382416, -0.6176707744598389, -0.6877889633178711, -1.1417591571807861, 0.6320437788963318, -0.6063031554222107], (2, 1, 3, 1)),
        f32([1.4051986932754517, -0.2876608669757843, 0.0749375969171524, 1.2207484245300293, -0.48621267080307007, -0.688210129737854, -0.6774346828460693, 0.3670888841152191, 0.0008057440281845629, -0.2080310881137848, 0.9697791337966919, 0.7583738565444946], (1, 1, 3, 4)),
    ],
    'AvgPool': [f32([0.4763999879360199, -0.19760000705718994, 1.6505999565124512, -0.24210000038146973, 0.6412000060081482, 1.9984999895095825, 0.3937999904155731, 0.1347000002861023, 0.22040000557899475, -0.7502999901771545, 0.21389999985694885, 0.7285000085830688, -0.020999999716877937, -0.4584999978542328, -1.5333000421524048, -0.4772000014781952, 0.5559999942779541, 0.6323000192642212, -2.5371999740600586, 1.4905999898910522, -1.1061999797821045, -0.970300018787384, 0.23659999668598175, -0.91839998960495, 0.30140000581741333, 0.7985000014305115, -0.6840999722480774, -2.285399913787842, -2.7727999687194824, -1.2805999517440796, -1.0946999788284302, -0.5989999771118164, -0.30329999327659607, -1.9041999578475952, -0.5403000116348267, 0.23319999873638153, 0.921500027179718, -0.15489999949932098, 0.05570000037550926, -0.5566999912261963, -1.4970999956130981, 0.5386000275611877, -0.2921999990940094, 0.4860000014305115, -0.39730000495910645, -0.46239998936653137, 0.4514000117778778, 0.23849999904632568, 0.3783000111579895, -1.0499999523162842], (1, 1, 5, 10))],
    'Cast': [i64([1, 2, 3, 4, 5, 6], (2, 3))],
    'ComplexTopK': [f32([9.0, 8.0, 4.5, 1.7000000476837158, 2.9000000953674316, 3.200000047683716, 4.0, 2.5999999046325684, 7.400000095367432, 3.5, 5.599999904632568, 7.099999904632568, 9.800000190734863, 1.100000023841858, 3.299999952316284, 6.199999809265137, 8.399999618530273, 0.699999988079071, 2.200000047683716, 3.299999952316284, 4.400000095367432, 5.5, 6.599999904632568, 7.699999809265137, 8.800000190734863, 9.899999618530273, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 7.0, 8.0, 9.0], (2, 3, 9))],
    'Constant': [
    ],
    'ConvAddRelu': [f32(np.arange(-7.0, 9.0), (1, 1, 4, 4))],
    'ConvTranspose1d': [f32(np.arange(0.0, 3.0), (1, 1, 3))],
    'ConvTranspose2d': [f32(np.arange(0.0, 9.0), (1, 1, 3, 3))],
    'ConvTransposeBias2d': [f32(np.arange(0.0, 9.0), (1, 1, 3, 3))],
    'ConvTransposeBias2dBatched': [f32(np.arange(0.0, 18.0), (2, 1, 3, 3))],
    'ConvWithAsymmetricPadding': [f32(np.arange(0.0, 35.0), (1, 1, 7, 5))],
    'ConvWithAutopadSameLower': [f32(np.arange(0.0, 25.0), (1, 1, 5, 5))],
    'ConvWithAutopadSameUpper': [f32(np.arange(0.0, 25.0), (1, 1, 5, 5))],
    'ConvWithDilation': [f32(np.arange(0.0, 49.0), (1, 1, 7, 7))],
    'ConvWithPadding': [f32(np.arange(0.0, 25.0), (1, 1, 5, 5))],
    'ConvWithStridesNoPadding': [f32(np.arange(0.0, 35.0), (1, 1, 7, 5))],
    'ConvWithStridesPadding': [f32(np.arange(0.0, 35.0), (1, 1, 7, 5))],
    'ConvWithoutPadding': [f32(np.arange(0.0, 25.0), (1, 1, 5, 5))],
    'Div': [
        f32([4.0, 2.0], (2,)),
        f32([2.0, 2.0], (2,)),
    ],
    'Elu': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (2, 3))],
    'EluAlpha': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (2, 3))],
    'Equal': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 2.0, 6.0], (3,)),
    ],
    'Erf': [f32([-1.041200041770935, 0.19179999828338623, 0.9984999895095825, -0.5958999991416931, 0.6841999888420105, -2.4718000888824463, 0.18039999902248383, 0.6851000189781189, 1.5645999908447266, -1.4981000423431396, 0.42480000853538513, -0.8503999710083008], (12,))],
    'Exp': [f32([1.4656645059585571, 0.6333451271057129, 2.4048163890838623, 0.5446845293045044, -1.4127167463302612, -0.18609187006950378, 0.275448203086853, 1.106152057647705, 0.884743869304657, 0.47531232237815857], (10,))],
    'ExpandDiffSize': [f32([0.0, 1.0, 2.0], (3, 1))],
    'ExpandSameSize': [f32([0.0, 1.0, 2.0], (3, 1))],
    'EyeLike': [f32([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 3))],
    'GRUBatchwise': [f32(np.arange(1.0, 7.0), (3, 1, 2))],
    'GRUBidirectional': [f32(np.arange(1.0, 7.0), (1, 3, 2))],
    'GRUDefaults': [f32(np.arange(1.0, 7.0), (1, 3, 2))],
    'GRUInitialBias': [f32(np.arange(1.0, 10.0), (1, 3, 3))],
    'GRUSeqLength': [f32(np.arange(1.0, 19.0), (2, 3, 3))],
    'Gather2d': [f32(np.arange(0.0, 9.0), (3, 3))],
    'GatherAxis0': [f32(np.arange(0.0, 120.0), (5, 4, 3, 2))],
    'GatherAxis1': [f32(np.arange(0.0, 120.0), (5, 4, 3, 2))],
    'GatherAxis2': [f32(np.arange(0.0, 120.0), (5, 4, 3, 2))],
    'GatherAxis3': [f32(np.arange(0.0, 120.0), (5, 4, 3, 2))],
    'GatherNegativeIndices': [f32(np.arange(0.0, 10.0), (10,))],
    'Gelu': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (6,))],
    'Greater': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 2.0, 6.0], (3,)),
    ],
    'GreaterOrEqual': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 2.0, 6.0], (3,)),
    ],
    'HardSigmoid': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (6,))],
    'HardSwish': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (6,))],
    # Per (n, c) slice a different mean and variance, so that a normalization
    # that mixed up instances or channels would not cancel out.
    'InstanceNormalization': [
        f32(np.arange(120.0) * 0.1 - 6.0 + (np.arange(120.0) % 7), (2, 3, 4, 5)),
    ],
    'InstanceNormalization3d': [
        f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
             -1.0, -3.0, 0.0, 2.0, -2.0, 4.0,
             10.0, 10.0, 10.0, 10.0, 10.0, 11.0,
             0.5, -0.5, 1.5, -1.5, 2.5, -2.5], (2, 2, 6)),
    ],
    # Small variance (~2e-3), comparable to the model's epsilon of 0.01.
    'InstanceNormalizationEpsilon': [
        f32((np.arange(18.0) % 5 - 2.0) * 0.025, (1, 2, 3, 3)),
    ],
    'LSTMBatchwise': [f32(np.arange(1.0, 7.0), (3, 1, 2))],
    'LSTMBidirectional': [f32(np.arange(1.0, 7.0), (3, 1, 2))],
    'LSTMDefaults': [f32(np.arange(1.0, 7.0), (3, 1, 2))],
    'LSTMInitialBias': [f32(np.arange(1.0, 10.0), (3, 1, 3))],
    'LSTMPeepholes': [f32(np.arange(1.0, 9.0), (1, 2, 4))],
    'LayerNormalization2d': [f32(np.arange(0.0, 12.0), (3, 4))],
    'LayerNormalization4d': [f32(np.arange(0.0, 120.0), (2, 3, 4, 5))],
    'Less': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 2.0, 6.0], (3,)),
    ],
    'LessOrEqual': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 2.0, 6.0], (3,)),
    ],
    'LinearWithLeakyRelu': [f32([0.43689998984336853, -0.6881999969482422, 1.030900001525879, -1.0262999534606934, -0.15189999341964722, 1.2237000465393066, -0.7053999900817871, -0.1762000024318695, -0.6811000108718872, -2.259700059890747, 1.0388000011444092, -0.7993000149726868, 0.1467999964952469, 1.325700044631958, -0.4713999927043915, -0.0957999974489212, 0.7056999802589417, -0.3749000132083893, -0.3310000002384186, 0.09860000014305115, -0.13699999451637268, 0.08320000022649765, -1.6464999914169312, -0.2793000042438507], (24,))],
    'LinearWithSelu': [f32(np.full(48, 1.0), (2, 24))],
    'LinearWithSigmoid': [f32(np.full(48, 1.0), (2, 24))],
    'Linear_16': [f32(np.full(1600, 1.0), (16, 100))],
    'Linear_32': [f32(np.full(3200, 1.0), (32, 100))],
    'Linear_64': [f32(np.full(6400, 1.0), (64, 100))],
    'Log': [f32([1.0, 2.0, 3.0, 4.0], (4,))],
    'Max': [
        f32([1.0, 2.0, -1.0], (1, 3)),
        f32([3.0, 0.0, 4.0], (1, 3)),
    ],
    'MaxMultidirectionalBroadcast': [
        f32([0.35974153876304626, -2.2087337970733643, 0.957462728023529], (3, 1)),
        f32([0.7590198516845703, -0.4654446244239807, -0.34920576214790344, -0.14607539772987366, 0.08269050717353821, -0.700456976890564], (2, 3, 1)),
        f32([-0.4146898090839386, -0.46591925621032715, 0.5617253184318542, 0.056169308722019196], (1, 4)),
    ],
    'MaxPool1d': [f32([0.09070000052452087, 0.10289999842643738, 0.814300000667572, 1.4496999979019165, -0.7785000205039978, 0.3824999928474426, -0.3763999938964844, 1.5785000324249268, -0.08349999785423279, 0.16220000386238098, 1.5866999626159668, 0.9822999835014343, -0.882099986076355, 0.4438999891281128, -0.13779999315738678, -0.2273000031709671, -0.01979999989271164, -2.0230000019073486, 0.09049999713897705, 0.6674000024795532, -1.4290000200271606, -1.309999942779541, -0.9438999891281128, -0.08330000191926956, -0.19189999997615814, 0.6886000037193298, 0.9388999938964844, -1.2913999557495117, -1.3583999872207642, -2.03410005569458, -0.32690000534057617, 0.1703999936580658, 1.1776000261306763, 1.3971999883651733, -1.8874000310897827, -1.533400058746338, 1.154099941253662, 0.3010999858379364, 0.6568999886512756, -2.350399971008301, 0.4032999873161316, 0.11420000344514847, 2.284600019454956, -1.3947999477386475, -0.8572999835014343, 0.5756000280380249, -1.086400032043457, 0.22830000519752502, 0.8946999907493591, 1.7626999616622925, -0.1657000035047531, 0.0649000033736229, -1.606600046157837, 0.41620001196861267, -1.152500033378601, -0.8184000253677368, 1.1324000358581543, -1.1086000204086304, 0.10610000044107437, 1.007099986076355], (1, 6, 10))],
    'MaxPool2d': [f32([0.6266000270843506, 0.1656000018119812, 0.275299996137619, -0.45579999685287476, -1.4592000246047974, 0.9284999966621399, -1.340999960899353, 1.3222999572753906, -0.5935999751091003, -1.364799976348877, -0.2989000082015991, 0.5900999903678894, -0.8845000267028809, -0.043299999088048935, 0.8313999772071838, -1.71589994430542, -0.5764999985694885, 0.8677999973297119, 1.0256999731063843, 0.7846999764442444, -0.34209999442100525, -1.2364000082015991, -0.5805000066757202, 0.44209998846054077, 1.218400001525879, 0.5042999982833862, 1.6822999715805054, -1.04830002784729, -2.2797999382019043, -1.892699956893921, 0.7716000080108643, 0.04050000011920929, 0.31209999322891235, -0.3010999858379364, -0.32659998536109924, -1.965999960899353, 1.0836999416351318, 0.23170000314712524, 0.9083999991416931, -0.32850000262260437, -0.9398000240325928, -0.20649999380111694, -0.9498999714851379, -0.9739000201225281, -0.12880000472068787, -0.13750000298023224, -1.261199951171875, 0.8809999823570251, 0.850600004196167, 0.445499986410141], (1, 1, 5, 10))],
    'MaxPool2d_AsymPad': [f32([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0], (1, 1, 4, 4))],
    'MaxPool2d_CeilMode': [f32(np.arange(25), (1, 1, 5, 5))],
    'MaxPool3d': [f32([-2.649600028991699, 1.0476000308990479, -0.5152999758720398, 0.37709999084472656, 0.41290000081062317, -0.3077000081539154, -0.8716999888420105, -0.8040000200271606, -0.35249999165534973, -0.17649999260902405, -0.33640000224113464, 0.8737000226974487, -0.23810000717639923, -0.8296999931335449, 0.4666000008583069, 0.6984000205993652, -0.6759999990463257, 0.629800021648407, 1.3832999467849731, 0.11010000109672546, 0.20389999449253082, -0.5476999878883362, 0.23409999907016754, 0.9180999994277954, 0.38420000672340393, 0.24279999732971191, 1.7924000024795532], (1, 1, 3, 3, 3))],
    'MeanMultidirectionalBroadcast': [
        f32([0.35974154, -2.20873388, 0.95746274], (3, 1)),
        f32([0.75901985, -0.46544461, -0.34920575, -0.1460754, 0.08269051, -0.70045695], (2, 3, 1)),
        f32([-0.41468981, -0.46591926, 0.56172534, 0.05616931], (1, 4)),
    ],
    'MinMultidirectionalBroadcast': [
        f32([0.35974153876304626, -2.2087337970733643, 0.957462728023529], (3, 1)),
        f32([0.7590198516845703, -0.4654446244239807, -0.34920576214790344, -0.14607539772987366, 0.08269050717353821, -0.700456976890564], (2, 3, 1)),
        f32([-0.4146898090839386, -0.46591925621032715, 0.5617253184318542, 0.056169308722019196], (1, 4)),
    ],
    'Mul': [
        f32([1.0, 2.0], (2,)),
        f32([0.0, 1.0], (2,)),
    ],
    'Neg': [f32([-1.909999966621399, 1.881100058555603, -1.7268999814987183, -0.10939999669790268, -0.014499999582767487, 0.250900000333786, 0.5892999768257141, -2.2732999324798584, -0.7077000141143799, 1.0644999742507935, -0.8607000112533569, 0.2084999978542328], (12,))],
    'Pow': [
        f32([1.0, 2.0, 3.0], (3,)),
        f32([4.0, 5.0, 6.0], (3,)),
    ],
    'Pow_broadcast': [
        f32([1.0, 2.0, 3.0, 3.0, 4.0, 5.0], (1, 2, 3)),
        f32([2.0, 3.0, 4.0, 2.0, 3.0, 4.0], (2, 3)),
    ],
    'RNNBatchwise': [f32(np.arange(1.0, 7.0), (3, 1, 2))],
    'RNNBidirectional': [f32([0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17], (3, 3, 2))],
    'RNNBidirectionalBatchwise': [f32([0.0, 0.01, 0.06, 0.07, 0.12, 0.13, 0.02, 0.03, 0.08, 0.09, 0.14, 0.15, 0.04, 0.05, 0.1, 0.11, 0.16, 0.17], (3, 3, 2))],
    'RNNDefaults': [f32(np.arange(1.0, 10.0), (3, 1, 3))],
    'RNNSeqLength': [f32(np.arange(1.0, 19.0), (2, 3, 3))],
    'RNNSequence': [f32([0.01, -0.01, 0.08, 0.09, 0.001, 0.09, -0.7, -0.35, 0.0, 0.001, 0.16, -0.19, 0.003, 0.0, 0.0001, 0.05, -0.09, 0.013, 0.5, 0.005, 0.2, -0.05, 0.062, -0.04, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.087, 0.01, 0.3, -0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 3, 5))],
    'RNNSequenceBatchwise': [f32([0.01, -0.01, 0.08, 0.09, 0.001, 0.05, -0.09, 0.013, 0.5, 0.005, 0.06, 0.087, 0.01, 0.3, -0.001, 0.09, -0.7, -0.35, 0.0, 0.001, 0.2, -0.05, 0.062, -0.04, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16, -0.19, 0.003, 0.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], (3, 3, 5))],
    'RangeFloat': [
        f32([1.0], (1,)),
        f32([10.0], (1,)),
        f32([2.0], (1,)),
    ],
    'RangeInt': [
        i64([1], (1,)),
        i64([10], (1,)),
        i64([2], (1,)),
    ],
    'Reciprocal': [f32([1.2690999507904053, -1.215999960899353, 0.6392999887466431, -0.4438000023365021, 0.8065000176429749, 0.20110000669956207], (2, 3))],
    'ReduceMean': [f32([5.0, 2.0, 3.0, 5.0, 5.0, 4.0], (1, 2, 3))],
    'ReduceProd': [f32([5.0, 2.0, 3.0, 5.0, 5.0, 4.0], (1, 2, 3))],
    'Shape': [f32([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (1, 2, 3))],
    'Slice': [rand_f32(1, (20, 10, 5))],
    'Slice_Default_Axis': [rand_f32(2, (20, 10, 5))],
    'Slice_Default_Steps': [rand_f32(3, (20, 10, 5))],
    'Slice_Neg': [rand_f32(4, (20, 10, 5))],
    'Softmax1d': [f32([-1.0, 0.0, 1.0], (3,))],
    'Softmax2d': [f32([-1.0, 0.0, 1.0], (1, 3))],
    'Softmax3d': [f32([-0.8938999772071838, -0.36739999055862427, 0.17630000412464142, 1.580399990081787, -0.46869999170303345, 1.2252999544143677, -1.3487999439239502, -0.10000000149011612, -0.12620000541210175, 0.49619999527931213, 1.0870000123977661, 0.690500020980835, -0.3450999855995178, -1.698099970817566, -0.46880000829696655, 0.44679999351501465, -0.5479000210762024, 0.06499999761581421, 1.044600009918213, -1.624899983406067, -0.718999981880188, -1.7519999742507935, 3.7753000259399414, -1.493899941444397], (2, 3, 4))],
    'Softmax4d': [f32([-0.586899995803833, -1.4271999597549438, -0.15459999442100525, 0.009600000455975533, 0.17059999704360962, 0.03880000114440918, -0.3483999967575073, -0.7828999757766724, 1.113800048828125, -0.5644000172615051, -0.6263999938964844, -1.1890000104904175, 1.6741000413894653, -0.7129999995231628, 0.9592000246047974, 1.7476999759674072, -0.47749999165534973, 1.3407000303268433, -0.3882000148296356, -0.4560000002384186, 1.0384999513626099, -0.16689999401569366, 0.5540000200271606, -1.0789999961853027, -0.6152999997138977, -0.6273999810218811, -1.2303999662399292, -0.6757000088691711, 1.017799973487854, -0.2379000037908554, -0.7911999821662903, -0.016499999910593033, -0.5422999858856201, 0.14589999616146088, 1.3585000038146973, -0.5005000233650208, -0.21870000660419464, -1.8180999755859375, -0.6642000079154968, 0.028699999675154686, -1.9103000164031982, 0.7983999848365784, -0.7860000133514404, 1.5133999586105347, 1.3873000144958496, -0.6462000012397766, -0.6353999972343445, -0.13349999487400055], (2, 3, 4, 2))],
    'Sqrt': [f32([0.8343999981880188, 0.4715999960899353, 0.6226000189781189, 0.8447999954223633, 0.2483000010251999, 0.9466999769210815], (2, 3))],
    'Sub': [
        f32([1.0, 2.0], (2,)),
        f32([0.0, 1.0], (2,)),
    ],
    'SumMultidirectionalBroadcast': [
        f32([0.35974153876304626, -2.2087337970733643, 0.957462728023529], (3, 1)),
        f32([0.7590198516845703, -0.4654446244239807, -0.34920576214790344, -0.14607539772987366, 0.08269050717353821, -0.700456976890564], (2, 3, 1)),
        f32([-0.4146898090839386, -0.46591925621032715, 0.5617253184318542, 0.056169308722019196], (1, 4)),
    ],
    'Swish': [f32([1.0, -2.0, 3.0, 0.5, -1.0, 2.0], (6,))],
    'Tanh': [f32([-0.38960000872612, -0.3521000146865845, 0.03629999980330467, 1.0961999893188477, 0.5084999799728394, -0.8522999882698059, -0.6765999794006348, 0.24210000038146973, 1.597100019454956, 1.3873000144958496, -0.21119999885559082, -0.6894999742507935, -0.5069000124931335, -2.1394999027252197, -0.7087000012397766, 1.1657999753952026, 1.3493000268936157, 0.8131999969482422, 1.7156000137329102, -0.8636999726295471, -0.19709999859333038, 0.041099999099969864, -0.5662000179290771, -0.2515999972820282], (24,))],
    'Tile5D': [f32([0.2386120855808258, 0.5549510717391968, -1.8190287351608276, 0.5724563598632812, -0.6596977710723877, 0.17560836672782898, 0.7608169317245483, 0.08603227883577347, -0.049375515431165695, 0.2705111503601074, 1.42119562625885, 0.032626643776893616, -1.212586522102356, -0.5129594802856445, -0.43296414613723755, -0.1606937050819397, 1.1884371042251587, -0.662174642086029, -2.291109323501587, -0.6852569580078125, 2.325223922729492, -0.19389064610004425, -0.5784135460853577, -0.39328137040138245, 0.2831517457962036, 0.4496127665042877, -0.2029038816690445, 0.35477763414382935, 0.4266718924045563, 0.24683749675750732, 1.90426504611969, -0.4861580729484558, 0.9139055013656616, -0.5031066536903381, 0.9583520293235779, -0.23210509121418, 1.3183971643447876, 1.7042455673217773, -0.3201166093349457, -0.14444805681705475, -0.8829464912414551, 1.725736141204834, 0.45657631754875183, 0.4920198321342468, -1.088847041130066, 0.49437597393989563, -0.006085286382585764, 2.475630760192871, 0.12170185893774033, -0.8953945636749268, 1.1430096626281738, 1.3278610706329346, 0.3076854348182678, 0.036237504333257675, 0.05180325731635094, 0.2802475392818451, 0.5289335250854492, 0.9356630444526672, 0.7863689064979553, 0.4239695370197296, 0.8723016977310181, -0.2248474359512329, 0.3891502320766449, 0.5463842153549194, -0.7782878875732422, -0.8570080399513245, -2.593783378601074, -0.11392943561077118, 0.5637082457542419, 2.075004816055298, -1.0598397254943848, 1.0823975801467896], (2, 2, 2, 3, 3))],
    'TopK': [f32([9.0, 8.0, 4.5, 1.7000000476837158, 2.9000000953674316, 3.200000047683716, 4.0, 2.5999999046325684, 7.400000095367432], (9,))],
}


def _recurrent_reference(model, feeds):
    """Expected outputs for the RNN/LSTM/GRU models that onnx's
    ReferenceEvaluator cannot evaluate (bidirectional recurrence, batchwise
    layout or per-batch sequence lengths). Implements the ONNX operator
    definitions directly with numpy, reading the weights from the model."""
    node = model.graph.node[0]
    op = node.op_type
    init = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}

    def operand(idx):
        if idx < len(node.input) and node.input[idx]:
            name = node.input[idx]
            return feeds[name] if name in feeds else init[name]
        return None

    attrs = {a.name: a for a in node.attribute}
    hidden = attrs["hidden_size"].i
    layout = attrs["layout"].i if "layout" in attrs else 0
    direction = attrs["direction"].s.decode() if "direction" in attrs else "forward"
    linear_before_reset = (
        attrs["linear_before_reset"].i if "linear_before_reset" in attrs else 0
    )
    if "activations" in attrs:
        acts = [s.decode() for s in attrs["activations"].strings]
        defaults = {"RNN": ["Tanh"], "GRU": ["Sigmoid", "Tanh"], "LSTM": ["Sigmoid", "Tanh", "Tanh"]}[op]
        num_dir_acts = 2 if direction == "bidirectional" else 1
        if acts != defaults * num_dir_acts:
            raise RuntimeError(f"{op}: non-default activations {acts} not implemented")

    X, W, R = operand(0), operand(1), operand(2)
    if op == "LSTM":
        B, seq_lens, initial_h, initial_c, P = (operand(i) for i in range(3, 8))
    else:
        B, seq_lens, initial_h = (operand(i) for i in range(3, 6))
        initial_c = P = None

    if layout == 1:
        X = X.transpose(1, 0, 2)
    seq_length, batch, _ = X.shape
    num_dir = 2 if direction == "bidirectional" else 1
    ngates = {"RNN": 1, "GRU": 3, "LSTM": 4}[op]
    if layout == 1:
        # With layout=1, initial_h/initial_c hold [batch, num_dir, hidden].
        # The pytorch exporter kept the layout-0 dims on the initializer, so
        # reinterpret the raw buffer instead of transposing the array.
        if initial_h is not None:
            initial_h = initial_h.flatten().reshape(batch, num_dir, hidden).transpose(1, 0, 2)
        if initial_c is not None:
            initial_c = initial_c.flatten().reshape(batch, num_dir, hidden).transpose(1, 0, 2)
    if B is None:
        B = np.zeros((num_dir, 2 * ngates * hidden), np.float32)
    if seq_lens is None:
        seq_lens = np.full(batch, seq_length)
    seq_lens = np.asarray(seq_lens).astype(np.int64)
    if direction != "forward" and not np.all(seq_lens == seq_length):
        raise RuntimeError("per-batch sequence lengths only implemented for direction=forward")
    if initial_h is None:
        initial_h = np.zeros((num_dir, batch, hidden), np.float32)
    if op == "LSTM" and initial_c is None:
        initial_c = np.zeros((num_dir, batch, hidden), np.float32)

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    Y = np.zeros((seq_length, num_dir, batch, hidden), np.float32)
    Y_h = np.zeros((num_dir, batch, hidden), np.float32)
    Y_c = np.zeros((num_dir, batch, hidden), np.float32)
    for d in range(num_dir):
        reverse = direction == "reverse" or d == 1
        Wd, Rd = W[d], R[d]
        Wb, Rb = B[d][: ngates * hidden], B[d][ngates * hidden :]
        h = initial_h[d].astype(np.float32)
        c = initial_c[d].astype(np.float32) if op == "LSTM" else None
        for step in range(seq_length):
            t = seq_length - 1 - step if reverse else step
            x = X[t]
            pre = x @ Wd.T + h @ Rd.T + Wb + Rb
            if op == "RNN":
                h_new = np.tanh(pre)
            elif op == "GRU":
                z = sigmoid(pre[:, :hidden])
                r = sigmoid(pre[:, hidden : 2 * hidden])
                Wpre_h = x @ Wd[2 * hidden :].T + Wb[2 * hidden :]
                if linear_before_reset:
                    h_tilde = np.tanh(Wpre_h + r * (h @ Rd[2 * hidden :].T + Rb[2 * hidden :]))
                else:
                    h_tilde = np.tanh(Wpre_h + (r * h) @ Rd[2 * hidden :].T + Rb[2 * hidden :])
                h_new = (1 - z) * h_tilde + z * h
            else:  # LSTM: gate order in W/R/B is i, o, f, c
                pc = P[d] if P is not None else np.zeros(3 * hidden, np.float32)
                i_g = sigmoid(pre[:, :hidden] + pc[:hidden] * c)
                f_g = sigmoid(pre[:, 2 * hidden : 3 * hidden] + pc[2 * hidden :] * c)
                c_tilde = np.tanh(pre[:, 3 * hidden :])
                c_new = f_g * c + i_g * c_tilde
                o_g = sigmoid(pre[:, hidden : 2 * hidden] + pc[hidden : 2 * hidden] * c_new)
                h_new = o_g * np.tanh(c_new)
            valid = (seq_lens > t).reshape(batch, 1)
            if op == "LSTM":
                c = np.where(valid, c_new, c)
            h = np.where(valid, h_new, h)
            Y[t, d] = np.where(valid, h_new, 0)
        Y_h[d] = h
        if op == "LSTM":
            Y_c[d] = c

    if layout == 1:
        Y = Y.transpose(2, 0, 1, 3)
        Y_h = Y_h.transpose(1, 0, 2)
        Y_c = Y_c.transpose(1, 0, 2)
    outs = {"Y": Y, "Y_h": Y_h, "Y_c": Y_c}
    return [outs[name] for name in ("Y", "Y_h", "Y_c")[: len(node.output)] if name]


def _maxpool2d_reference(model, feeds):
    """MaxPool with asymmetric padding; the ReferenceEvaluator computes a
    wrong output shape for that case."""
    node = model.graph.node[0]
    attrs = {a.name: (list(a.ints) if a.type == onnx.AttributeProto.INTS else a.i)
             for a in node.attribute}
    kh, kw = attrs["kernel_shape"]
    sh, sw = attrs.get("strides", [1, 1])
    pt, pl, pb, pr = attrs.get("pads", [0, 0, 0, 0])
    if attrs.get("ceil_mode", 0) != 0:
        raise RuntimeError("ceil_mode not implemented")
    x = feeds[node.input[0]]
    n, c, h, w = x.shape
    xp = np.full((n, c, h + pt + pb, w + pl + pr), -np.inf, np.float32)
    xp[:, :, pt : pt + h, pl : pl + w] = x
    h_out = (h + pt + pb - kh) // sh + 1
    w_out = (w + pl + pr - kw) // sw + 1
    out = np.empty((n, c, h_out, w_out), np.float32)
    for i in range(h_out):
        for j in range(w_out):
            out[:, :, i, j] = xp[:, :, i * sh : i * sh + kh, j * sw : j * sw + kw].max(axis=(2, 3))
    return [out]


def _mean_reference(model, feeds):
    """Mean with multidirectional broadcasting; the ReferenceEvaluator fails
    to broadcast the inputs to a common shape."""
    return [np.mean(np.broadcast_arrays(*feeds.values()), axis=0, dtype=np.float32)]


# Models whose expected outputs the ReferenceEvaluator cannot compute.
EXPECTED_OVERRIDES = {
    "GRUBidirectional": _recurrent_reference,
    "LSTMBidirectional": _recurrent_reference,
    "RNNBidirectional": _recurrent_reference,
    "RNNBidirectionalBatchwise": _recurrent_reference,
    "RNNSequence": _recurrent_reference,
    "RNNSequenceBatchwise": _recurrent_reference,
    "MaxPool2d_AsymPad": _maxpool2d_reference,
    "MeanMultidirectionalBroadcast": _mean_reference,
}


def compute_reference(name):
    """Return (feeds, outputs) for the test inputs of the given model."""
    model = MODELS[name]()
    init_names = {t.name for t in model.graph.initializer}
    graph_inputs = [vi for vi in model.graph.input if vi.name not in init_names]
    arrays = TEST_INPUTS[name]
    if len(arrays) != len(graph_inputs):
        raise RuntimeError(
            f"{name}: {len(arrays)} test inputs for {len(graph_inputs)} graph inputs"
        )
    feeds = {vi.name: arr for vi, arr in zip(graph_inputs, arrays)}
    if name in EXPECTED_OVERRIDES:
        outputs = EXPECTED_OVERRIDES[name](model, feeds)
    else:
        outputs = ReferenceEvaluator(model).run(None, feeds)
    return feeds, [np.asarray(o) for o in outputs]


def _write_reference_file(path, feeds, outputs):
    """Text format, one entry per graph input/output:
    <key> <f32|f64|i64|u8> <count>\n<values...>\n"""

    def entry(f, key, arr):
        if arr.dtype == np.bool_:
            arr = arr.astype(np.uint8)
        code = {"float32": "f32", "float64": "f64", "int64": "i64",
                "int32": "i64", "uint8": "u8"}[arr.dtype.name]
        flat = arr.flatten()
        if code in ("f32", "f64"):
            vals = " ".join(repr(float(v)) for v in flat)
        else:
            vals = " ".join(str(int(v)) for v in flat)
        f.write(f"{key} {code} {arr.size}\n{vals}\n")

    with open(path, "w") as f:
        for k, (name, arr) in enumerate(feeds.items()):
            entry(f, f"input{k}", np.asarray(arr))
        for k, arr in enumerate(outputs):
            entry(f, f"output{k}", arr)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="*", metavar="MODEL",
                        help="models to generate (default: all)")
    parser.add_argument("--outdir", default=".", help="output directory")
    parser.add_argument("--list", action="store_true", dest="list_models",
                        help="only print the available model names")
    parser.add_argument("--no-references", action="store_true",
                        help="only generate the models, without the reference data files")
    args = parser.parse_args()

    if args.list_models:
        print("\n".join(MODELS))
        return 0

    if onnx is None:
        print("error: the onnx python package is required to generate the models", file=sys.stderr)
        return 1

    names = args.models or list(MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print(f"error: unknown model(s): {', '.join(unknown)}", file=sys.stderr)
        return 1

    os.makedirs(args.outdir, exist_ok=True)
    ref_dir = os.path.join(args.outdir, "references")
    if not args.no_references:
        os.makedirs(ref_dir, exist_ok=True)
    n_refs = 0
    for name in names:
        onnx.save(MODELS[name](), os.path.join(args.outdir, name + ".onnx"))
        if name in TEST_INPUTS and not args.no_references:
            feeds, outputs = compute_reference(name)
            _write_reference_file(os.path.join(ref_dir, name + ".ref"), feeds, outputs)
            n_refs += 1
    print(f"generated {len(names)} models and {n_refs} reference files in {args.outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
