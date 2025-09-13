if (tmva)
    list(APPEND PYROOT_EXTRA_PYTHON_SOURCES
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/__init__.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/generate_keras_functional.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/generate_keras_sequential.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/parser_test_function.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/__init__.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/batchnorm.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/binary.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/concat.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/conv.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/dense.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/elu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/flatten.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/identity.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/layernorm.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/leaky_relu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/permute.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/pooling.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/reshape.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/relu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/rnn.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/selu.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/sigmoid.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/softmax.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/swish.py
        ROOT/_pythonization/_tmva/_sofie/_parser/_keras/layers/tanh.py)
    set(PYROOT_EXTRA_PYTHON_SOURCES "${PYROOT_EXTRA_PYTHON_SOURCES}" PARENT_SCOPE)
endif()