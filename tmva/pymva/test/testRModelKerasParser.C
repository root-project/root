#include "TMVA/RModelParser_Keras.h"
#include "KerasSequentialModel.hxx"
#include "KerasFunctionalModel.hxx"

using namespace TMVA::Experimental;

bool compare(float i, float j){
    return std::abs(i-j)<std::numeric_limits<float>::epsilon();
}

int testKerasParser() {
    std::setprecision(5);
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

    int errSequential=0,errFunctional=0;

    std::cout<<"Testing Keras Parser for Sequential API model...";
    float inputSequential[]={0.12107884, 0.89718615, 0.89123899, 0.32197549,
                             0.17891638,0.83555135, 0.98680066, 0.14496809, 
                             0.07255503, 0.55386989, 0.6628149 , 0.29843291, 
                             0.71059786,0.44043452, 0.13792047, 0.93007397, 
                             0.16799397, 0.75473803, 0.43203355, 0.68360968, 
                             0.83879351,0.0558927 , 0.57500447, 0.49063431, 
                             0.63637339, 0.94483464, 0.11032887, 0.22424818, 
                             0.50972592,0.04671024, 0.39230661, 0.80500943};

    std::vector<float> outputSequential = TMVA_SOFIE_KerasModelSequential::infer(inputSequential);
    SOFIE::PyRunString("import os",fGlobalNS,fLocalNS);    
    SOFIE::PyRunString("os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("from keras.models import load_model",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("import numpy",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("model=load_model('KerasModelSequential.h5')",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("ip=numpy.array([0.12107884, 0.89718615, 0.89123899, 0.32197549," 
                                       "0.17891638, 0.83555135, 0.98680066, 0.14496809," 
                                       "0.07255503, 0.55386989, 0.6628149 , 0.29843291," 
                                       "0.71059786,0.44043452, 0.13792047, 0.93007397," 
                                       "0.16799397, 0.75473803, 0.43203355, 0.68360968," 
                                       "0.83879351, 0.0558927 , 0.57500447, 0.49063431," 
                                       "0.63637339, 0.94483464, 0.11032887, 0.22424818," 
                                       "0.50972592,0.04671024, 0.39230661, 0.80500943]).reshape(4,8)",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("op=model(ip).numpy()",fGlobalNS,fLocalNS);
    PyArrayObject* pSequentialValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"op");
    float* pOutputSequential=(float*)PyArray_DATA(pSequentialValues);
    if(!std::equal(outputSequential.begin(),outputSequential.end(),pOutputSequential,compare)){
        std::cout<<"\n[ERROR] Outputs from parsed Keras Sequential API model doesn't matches with the actual model outputs\n";
        errSequential=1;
    }

    else{
        std::cout<<" OK\n";
    }


    std::cout<<"Testing Keras Parser for Functional API model...";
    float inputFunctional[]={0.60828574, 0.50069386, 0.75186709, 0.14968806, 0.7692464 ,0.77027585, 0.75095316, 0.96651197, 
                             0.38536308, 0.95565917, 0.62796356, 0.13818375, 0.65484891,0.89220363, 0.23879365, 0.00635323};
    std::vector<float> outputFunctional = TMVA_SOFIE_KerasModelFunctional::infer(inputFunctional);
    SOFIE::PyRunString("model=load_model('KerasModelFunctional.h5')",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("ip=numpy.array([0.60828574, 0.50069386, 0.75186709, 0.14968806, 0.7692464 ,0.77027585, 0.75095316, 0.96651197," 
                                       "0.38536308, 0.95565917, 0.62796356, 0.13818375, 0.65484891,0.89220363, 0.23879365, 0.00635323]).reshape(2,8)",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("op=model(ip).numpy()",fGlobalNS,fLocalNS);
    PyArrayObject* pFunctionalValues=(PyArrayObject*)PyDict_GetItemString(fLocalNS,"op");
    float* pOutputFunctional=(float*)PyArray_DATA(pFunctionalValues);
    if(!std::equal(outputFunctional.begin(),outputFunctional.end(),pOutputFunctional,compare)){
        std::cout<<"\n[ERROR] Outputs from parsed Keras Functional API model doesn't matches with the actual model outputs\n";
        errFunctional=1;
    }
    else{
        std::cout<<" OK\n";
    }

    return errSequential || errFunctional;
}


int main(){
    int err= testKerasParser();
    return err;
}
