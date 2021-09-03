#include "KerasFunctionalModel.hxx"
#include "KerasSequentialModel.hxx"

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "gtest/gtest.h"

constexpr float DEFAULT_TOLERANCE = 1e-6f;

TEST(RModelParser_Keras, SEQUENTIAL)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   float inputSequential[]={ 0.12107884, 0.89718615, 0.89123899, 0.32197549,
                             0.17891638, 0.83555135, 0.98680066, 0.14496809,
                             0.07255503, 0.55386989, 0.6628149 , 0.29843291,
                             0.71059786, 0.44043452, 0.13792047, 0.93007397,
                             0.16799397, 0.75473803, 0.43203355, 0.68360968,
                             0.83879351, 0.0558927 , 0.57500447, 0.49063431,
                             0.63637339, 0.94483464, 0.11032887, 0.22424818,
                             0.50972592, 0.04671024, 0.39230661, 0.80500943};

    std::vector<float> outputSequential = TMVA_SOFIE_KerasModelSequential::infer(inputSequential);

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
    PyRun_String("import os",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("from tensorflow.keras.models import load_model",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("import numpy",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=load_model('KerasModelSequential.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.array([0.12107884, 0.89718615, 0.89123899, 0.32197549,"
                                    "0.17891638, 0.83555135, 0.98680066, 0.14496809,"
                                    "0.07255503, 0.55386989, 0.6628149 , 0.29843291,"
                                    "0.71059786, 0.44043452, 0.13792047, 0.93007397,"
                                    "0.16799397, 0.75473803, 0.43203355, 0.68360968,"
                                    "0.83879351, 0.0558927 , 0.57500447, 0.49063431,"
                                    "0.63637339, 0.94483464, 0.11032887, 0.22424818,"
                                    "0.50972592, 0.04671024, 0.39230661, 0.80500943]).reshape(4,8)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
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

TEST(RModelParser_Keras, FUNCTIONAL)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    float inputFunctional[]={0.60828574, 0.50069386, 0.75186709, 0.14968806, 0.7692464 ,0.77027585, 0.75095316, 0.96651197,
                             0.38536308, 0.95565917, 0.62796356, 0.13818375, 0.65484891,0.89220363, 0.23879365, 0.00635323};
    std::vector<float> outputFunctional = TMVA_SOFIE_KerasModelFunctional::infer(inputFunctional);

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
    PyRun_String("import os",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("from tensorflow.keras.models import load_model",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("import numpy",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=load_model('KerasModelFunctional.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.array([0.60828574, 0.50069386, 0.75186709, 0.14968806, 0.7692464 ,0.77027585, 0.75095316, 0.96651197,"
                                    "0.38536308, 0.95565917, 0.62796356, 0.13818375, 0.65484891,0.89220363, 0.23879365, 0.00635323]).reshape(2,8)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputFunctionalSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputFunctional.size(), pOutputFunctionalSize);

    PyArrayObject* pFunctionalValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputFunctional=(float*)PyArray_DATA(pFunctionalValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputFunctional.size(); ++i) {
      EXPECT_LE(std::abs(outputFunctional[i] - pOutputFunctional[i]), TOLERANCE);
    }
}
