#include "TMVA/RModelParser_PyTorch.h"
#include "PyTorchModuleModel.hxx"
#include "PyTorchSequentialModel.hxx"

using namespace TMVA::Experimental;

int testPyTorchParser() {

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

    int errSequential=0,errModule=0;

    std::cout<<"Testing PyTorch Parser for nn.Sequential model...";
    float inputSequential[]={-1.6207,  0.6133, 0.5058, -1.2560, -0.7750, -1.6701, 0.8171, -0.2858};
    std::vector<float> outputSequential = TMVA_SOFIE_PyTorchModelSequential::infer(inputSequential);
    SOFIE::PyRunString("import torch",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("model=torch.jit.load('PyTorchModelSequential.pt')",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("ip=torch.reshape(torch.FloatTensor([-1.6207,  0.6133, 0.5058, -1.2560, -0.7750, -1.6701, 0.8171, -0.2858]),(2,4))",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("op=model(ip).detach().numpy()",fGlobalNS,fLocalNS);
    RTensor<float> pSequentialValues=SOFIE::getArray(PyDict_GetItemString(fLocalNS,"op"));
    float* pSequentialValVec=(float*)pSequentialValues.GetData();
    std::vector<float>pOutputSequential{pSequentialValVec,pSequentialValVec+pSequentialValues.GetSize()};
    if(outputSequential!=pOutputSequential){
        std::cout<<"\nOutputs from parsed PyTorch nn.Sequential model doesn't matches with the actual model outputs\n";
        errSequential=1;
    }
    else{
        std::cout<<" OK\n";
    }

    std::cout<<"Testing PyTorch Parser for nn.Module model...";
    float inputModule[]={0.5516,  0.3585, -0.4854, -1.3884,  0.8057, -0.9449, 0.5626, -0.6466, -1.8818,  0.4736,  1.1102,  1.8694};
    std::vector<float> outputModule = TMVA_SOFIE_PyTorchModelModule::infer(inputModule);
    SOFIE::PyRunString("modelModule=torch.jit.load('PyTorchModelModule.pt')",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("ipModule=torch.reshape(torch.FloatTensor([0.5516,  0.3585, -0.4854, -1.3884,  0.8057, -0.9449, 0.5626, -0.6466, -1.8818,  0.4736,  1.1102,  1.8694]),(2,6))",fGlobalNS,fLocalNS);
    SOFIE::PyRunString("opModule=modelModule(ipModule).detach().numpy().reshape(2,12)",fGlobalNS,fLocalNS);
    RTensor<float> pModuleValues=SOFIE::getArray(PyDict_GetItemString(fLocalNS,"opModule"));
    float* pModuleValVec=(float*)pModuleValues.GetData();
    std::vector<float>pOutputModule{pModuleValVec,pModuleValVec+pModuleValues.GetSize()};
    if(outputModule!=pOutputModule){
        std::cout<<"\nOutputs from parsed PyTorch nn.Module model doesn't matches with the actual model outputs\n";
        errModule=1;
    }
    else{
        std::cout<<" OK\n";
    }

    return errSequential || errModule;
}


int main(){
    int err= testPyTorchParser();
    return err;
}
