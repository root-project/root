#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "gtest/gtest.h"
#include <cmath>

#include "TSystem.h"
#include "TMVA/RSofieReader.hxx"
#include "TMVA/PyMethodBase.h"

constexpr float DEFAULT_TOLERANCE = 1e-6f;

void GenerateModels() {

   FILE* fKerasModels = fopen("generateKerasModels.py", "r");
   if (!fKerasModels) {
      std::string msg = "Can't find Python script to generate models in " + std::string(gSystem->pwd());
      throw std::runtime_error(msg);
   }
   PyRun_SimpleFile(fKerasModels, "generateKerasModels.py");
}

TEST(RModelParser_Keras, SEQUENTIAL)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   // input is 8 x batch size that is fixed to be 4
   std::vector<float> inputSequential = { 0.12107884, 0.89718615, 0.89123899, 0.32197549,
                             0.17891638, 0.83555135, 0.98680066, 0.14496809,
                             0.07255503, 0.55386989, 0.6628149 , 0.29843291,
                             0.71059786, 0.44043452, 0.13792047, 0.93007397,
                             0.16799397, 0.75473803, 0.43203355, 0.68360968,
                             0.83879351, 0.0558927 , 0.57500447, 0.49063431,
                             0.63637339, 0.94483464, 0.11032887, 0.22424818,
                             0.50972592, 0.04671024, 0.39230661, 0.80500943};

    Py_Initialize();

    if (gSystem->AccessPathName("KerasModelSequential.h5",kFileExists))
        GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelSequential.h5",{{4,8}});
    std::vector<float> outputSequential = r.Compute(inputSequential);


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
    std::vector<float> inputFunctional ={0.60828574, 0.50069386, 0.75186709, 0.14968806, 0.7692464 ,0.77027585, 0.75095316, 0.96651197,
                             0.38536308, 0.95565917, 0.62796356, 0.13818375, 0.65484891,0.89220363, 0.23879365, 0.00635323};


    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelFunctional.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelFunctional.h5");
    std::vector<float> outputFunctional = r.Compute(inputFunctional);

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

TEST(RModelParser_Keras, BATCH_NORM)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputBatchNorm = {0.22308163, 0.95274901, 0.44712538, 0.84640867,
                             0.69947928, 0.29743695, 0.81379782, 0.39650574};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelBatchNorm.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelBatchNorm.h5");
    std::vector<float> outputBatchNorm =  r.Compute(inputBatchNorm);

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
    PyRun_String("model=load_model('KerasModelBatchNorm.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.array([0.22308163, 0.95274901, 0.44712538, 0.84640867,"
                                    "0.69947928, 0.29743695, 0.81379782, 0.39650574]).reshape(2,4)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputBatchNormSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputBatchNorm.size(), pOutputBatchNormSize);

    PyArrayObject* pBatchNormValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputBatchNorm=(float*)PyArray_DATA(pBatchNormValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputBatchNorm.size(); ++i) {
      EXPECT_LE(std::abs(outputBatchNorm[i] - pOutputBatchNorm[i]), TOLERANCE);
    }
}

#if PY_MAJOR_VERSION >= 3
TEST(RModelParser_Keras, CONV_VALID)
#else
TEST(DISABLED_RModelParser_Keras, CONV_VALID)
#endif
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputConv2D_Valid =  {1,1,1,1,
                                1,1,1,1,
                                1,1,1,1,
                                1,1,1,1};
    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelConv2D_Valid.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelConv2D_Valid.h5");
    std::vector<float> outputConv2D_Valid =  r.Compute(inputConv2D_Valid);

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
    PyRun_String("model=load_model('KerasModelConv2D_Valid.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.ones((1,4,4,1))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputConv2DValidSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputConv2D_Valid.size(), pOutputConv2DValidSize);

    PyArrayObject* pConv2DValidValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputConv2DValid=(float*)PyArray_DATA(pConv2DValidValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputConv2D_Valid.size(); ++i) {
      EXPECT_LE(std::abs(outputConv2D_Valid[i] - pOutputConv2DValid[i]), TOLERANCE);
    }
}

#if PY_MAJOR_VERSION >= 3
TEST(RModelParser_Keras, CONV_SAME)
#else
TEST(DISABLED_RModelParser_Keras, CONV_SAME)
#endif
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputConv2D_Same =  {1,1,1,1,
                                1,1,1,1,
                                1,1,1,1,
                                1,1,1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelConv2D_Same.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelConv2D_Same.h5");
    std::vector<float> outputConv2D_Same =  r.Compute(inputConv2D_Same);

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
    PyRun_String("model=load_model('KerasModelConv2D_Same.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.ones((1,4,4,1))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputConv2DSameSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputConv2D_Same.size(), pOutputConv2DSameSize);

    PyArrayObject* pConv2DSameValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputConv2DSame=(float*)PyArray_DATA(pConv2DSameValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputConv2D_Same.size(); ++i) {
      EXPECT_LE(std::abs(outputConv2D_Same[i] - pOutputConv2DSame[i]), TOLERANCE);
    }
}

TEST(RModelParser_Keras, RESHAPE)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputReshape = {1,1,1,1,
                          1,1,1,1,
                          1,1,1,1,
                          1,1,1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelReshape.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelReshape.h5");
    std::vector<float> outputReshape =  r.Compute(inputReshape);

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
    PyRun_String("model=load_model('KerasModelReshape.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.ones((1,4,4,1))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputReshapeSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputReshape.size(), pOutputReshapeSize);

    PyArrayObject* pReshapeValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputReshape=(float*)PyArray_DATA(pReshapeValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputReshape.size(); ++i) {
      EXPECT_LE(std::abs(outputReshape[i] - pOutputReshape[i]), TOLERANCE);
    }
}

TEST(RModelParser_Keras, CONCATENATE)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputConcatenate_1 = {1,1};
    std::vector<float>inputConcatenate_2 = {1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelConcatenate.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelConcatenate.h5");
    std::vector<float> outputConcatenate =  r.Compute(inputConcatenate_1, inputConcatenate_2);

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
    PyRun_String("model=load_model('KerasModelConcatenate.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input_1=numpy.ones((1,2))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input_2=numpy.ones((1,2))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model([input_1,input_2]).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputConcatenateSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputConcatenate.size(), pOutputConcatenateSize);

    PyArrayObject* pConcatenateValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputConcatenate=(float*)PyArray_DATA(pConcatenateValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputConcatenate.size(); ++i) {
      EXPECT_LE(std::abs(outputConcatenate[i] - pOutputConcatenate[i]), TOLERANCE);
    }
}

TEST(RModelParser_Keras, BINARY_OP)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>input_BinaryOp_1 = {1,1};
    std::vector<float>input_BinaryOp_2 = {1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelBinaryOp.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelBinaryOp.h5");
    std::vector<float> outputBinaryOp =  r.Compute(input_BinaryOp_1,input_BinaryOp_2);

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
    PyRun_String("model=load_model('KerasModelBinaryOp.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input1=numpy.array([1,1])",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input2=numpy.array([1,1])",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model([input1,input2]).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputBinaryOpSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputBinaryOp.size(), pOutputBinaryOpSize);

    PyArrayObject* pBinaryOpValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputBinaryOp=(float*)PyArray_DATA(pBinaryOpValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputBinaryOp.size(); ++i) {
      EXPECT_LE(std::abs(outputBinaryOp[i] - pOutputBinaryOp[i]), TOLERANCE);
    }
}

TEST(RModelParser_Keras, ACTIVATIONS)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>inputActivations = {1,1,1,1,1,1,1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelActivations.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelActivations.h5");
    std::vector<float> outputActivations =  r.Compute(inputActivations);

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
    PyRun_String("model=load_model('KerasModelActivations.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.ones((1,8))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputActivationsSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputActivations.size(), pOutputActivationsSize);

    PyArrayObject* pActivationsValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputActivations=(float*)PyArray_DATA(pActivationsValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputActivations.size(); ++i) {
      EXPECT_LE(std::abs(outputActivations[i] - pOutputActivations[i]), TOLERANCE);
    }
}

TEST(RModelParser_Keras, SWISH)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>input = {1,1,1,1,1,1,1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelSwish.h5",kFileExists))
       GenerateModels();

    TMVA::Experimental:: RSofieReader r("KerasModelSwish.h5");
    std::vector<float> output =  r.Compute(input);

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
    PyRun_String("model=load_model('KerasModelSwish.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.ones((1,8))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(output.size(), pOutputSize);

    PyArrayObject* pValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutput=(float*)PyArray_DATA(pValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < output.size(); ++i) {
        EXPECT_LE(std::abs(output[i] - pOutput[i]), TOLERANCE);
    }
}

TEST(RModel, CUSTOM_OP)
{
    constexpr float TOLERANCE = DEFAULT_TOLERANCE;
    std::vector<float>input_custom ={1,1,1,1,1,1,1,1};

    Py_Initialize();
    if (gSystem->AccessPathName("KerasModelCustomOp.h5",kFileExists))
       GenerateModels();



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

    PyRun_String("from tensorflow.keras.layers import Lambda",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("import numpy",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model=load_model('KerasModelCustomOp.h5')",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("model.add(Lambda(lambda x: x * 2))",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("input=numpy.array([1,1,1,1,1,1,1,1]).reshape(1,8)",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("output=model(input).numpy()",Py_single_input,fGlobalNS,fLocalNS);
    PyRun_String("outputSize=output.size",Py_single_input,fGlobalNS,fLocalNS);
    std::size_t pOutputCustomOpSize=(std::size_t)PyLong_AsLong(PyDict_GetItemString(fLocalNS,"outputSize"));

    // get input name for custom (it is output of one before last)
    PyRun_String("outputName = model.get_layer(index=len(model.layers)-2).output.name",Py_single_input,fGlobalNS,fLocalNS);
    PyObject *pOutputName = PyDict_GetItemString(fLocalNS, "outputName");
    std::string outputName = TMVA::PyMethodBase::PyStringAsString(pOutputName);
    TMVA::Experimental:: RSofieReader r;
    r.AddCustomOperator(/*OpName*/ "Scale_by_2",
                        /*input tensor names where to insert custom op */std::string("{\"" + outputName + "\"}"),
                        /*output tensor names*/"{\"Scale2Output\"}",
                        /*output shapes*/"{{1,4}}",
                       /*header file name with the compute function*/ "scale_by_2_op.hxx");
    // need to load model afterwards
    r.Load("KerasModelCustomOp.h5",{}, false);
    std::vector<float> outputCustomOp =  r.Compute(input_custom);

    //Testing the actual and expected output tensor sizes
    EXPECT_EQ(outputCustomOp.size(), pOutputCustomOpSize);

    PyArrayObject* pCustomOpValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"output");
    float* pOutputCustomOp=(float*)PyArray_DATA(pCustomOpValues);

    //Testing the actual and expected output tensor values
    for (size_t i = 0; i < outputCustomOp.size(); ++i) {
        EXPECT_LE(std::abs(outputCustomOp[i] - pOutputCustomOp[i]), TOLERANCE);
    }
}