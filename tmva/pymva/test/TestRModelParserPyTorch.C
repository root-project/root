#include "PyTorchModuleModel.hxx"
#include "PyTorchSequentialModel.hxx"

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-6f;

TEST(RModelParser_PyTorch, SEQUENTIAL)
{   
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   float inputSequential[]={-1.6207,  0.6133,
                             0.5058, -1.2560, 
                            -0.7750, -1.6701, 
                             0.8171, -0.2858};
    std::vector<float> outputSequential = TMVA_SOFIE_PyTorchModelSequential::infer(inputSequential);
    
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

TEST(RModelParser_PyTorch, MODULE)
{   
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   float inputModule[]={0.5516,  0.3585, 
                       -0.4854, -1.3884,  
                        0.8057, -0.9449, 
                        0.5626, -0.6466, 
                       -1.8818,  0.4736,  
                        1.1102,  1.8694};
    std::vector<float> outputModule = TMVA_SOFIE_PyTorchModelModule::infer(inputModule);
    
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
