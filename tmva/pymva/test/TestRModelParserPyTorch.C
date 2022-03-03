#include <numeric>

#include "PyTorchModuleModel.hxx"
#include "PyTorchSequentialModel.hxx"
#include "PyTorchConvolutionModel.hxx"

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "gtest/gtest.h"
#include <cmath>

constexpr float DEFAULT_TOLERANCE = 1e-6f;

TEST(RModelParser_PyTorch, SEQUENTIAL_MODEL)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   float inputSequential[]={-1.6207,  0.6133,
                             0.5058, -1.2560,
                            -0.7750, -1.6701,
                             0.8171, -0.2858};
    TMVA_SOFIE_PyTorchModelSequential::Session s("PyTorchSequentialModel.dat");
    std::vector<float> outputSequential = s.infer(inputSequential);

    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* fGlobalNS = PyModule_GetDict(main);
    PyObject* fLocalNS = PyDict_New();
    if (!fGlobalNS) {
        throw std::runtime_error("Can't init global namespace for Python");
        }
    if (!fLocalNS) {
        throw std::runtime_error("Can't init local namespace for Python");
        }
    PyRun_String("import torch",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=torch.jit.load('PyTorchModelSequential.pt')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=torch.reshape(torch.FloatTensor([-1.6207,  0.6133,"
                                                        " 0.5058, -1.2560,"
                                                        "-0.7750, -1.6701,"
                                                        " 0.8171, -0.2858]),(2,4))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).detach().numpy().reshape(2,6)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputSequentialSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputSequential.size(), pOutputSequentialSize);

    PyArrayObject* pSequentialValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputSequential=(float*)PyArray_DATA(pSequentialValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputSequential.size(); ++i) {
      EXPECT_LE(std::abs(outputSequential[i] - pOutputSequential[i]), TOLERANCE);
    }
}

TEST(RModelParser_PyTorch, MODULE_MODEL)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   float inputModule[]={0.5516,  0.3585,
                       -0.4854, -1.3884,
                        0.8057, -0.9449,
                        0.5626, -0.6466,
                       -1.8818,  0.4736,
                        1.1102,  1.8694};
    TMVA_SOFIE_PyTorchModelModule::Session s("PyTorchModuleModel.dat");
    std::vector<float> outputModule = s.infer(inputModule);

    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* fGlobalNS = PyModule_GetDict(main);
    PyObject* fLocalNS = PyDict_New();
    if (!fGlobalNS) {
        throw std::runtime_error("Can't init global namespace for Python");
        }
    if (!fLocalNS) {
        throw std::runtime_error("Can't init local namespace for Python");
        }
    PyRun_String("import torch",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=torch.jit.load('PyTorchModelModule.pt')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=torch.reshape(torch.FloatTensor([0.5516,  0.3585, "
                                                       "-0.4854, -1.3884,"
                                                       " 0.8057, -0.9449,"
                                                       " 0.5626, -0.6466,"
                                                       "-1.8818,  0.4736,"
                                                       " 1.1102,  1.8694]),(2,6))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).detach().numpy().reshape(2,12)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputModuleSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputModule.size(), pOutputModuleSize);

    PyArrayObject* pModuleValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputModule=(float*)PyArray_DATA(pModuleValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputModule.size(); ++i) {
      EXPECT_LE(std::abs(outputModule[i] - pOutputModule[i]), TOLERANCE);
    }
}

TEST(RModelParser_PyTorch, CONVOLUTION_MODEL)
{
    constexpr float TOLERANCE = 1.E-3;
    std::vector<float> inputConv(5*6*5*5);
    std::iota(inputConv.begin(), inputConv.end(), 1.0f);

    TMVA_SOFIE_PyTorchModelConvolution::Session s("PyTorchConvolutionModel.dat");
    std::vector<float> outputConv = s.infer(inputConv.data());

    Py_Initialize();
    PyObject* main = PyImport_AddModule("__main__");
    PyObject* fGlobalNS = PyModule_GetDict(main);
    PyObject* fLocalNS = PyDict_New();
    if (!fGlobalNS) {
        throw std::runtime_error("Can't init global namespace for Python");
        }
    if (!fLocalNS) {
        throw std::runtime_error("Can't init local namespace for Python");
        }
    PyRun_String("import torch",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=torch.jit.load('PyTorchModelConvolution.pt')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=torch.arange(1,751,dtype=torch.float).reshape(5,6,5,5)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).detach().numpy().reshape(5,5,2,2)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputConvSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputConv.size(), pOutputConvSize);

    PyArrayObject* pConvValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputConv=(float*)PyArray_DATA(pConvValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputConv.size(); ++i) {
      EXPECT_LE(std::abs(outputConv[i] - pOutputConv[i]), TOLERANCE);
    }
}
