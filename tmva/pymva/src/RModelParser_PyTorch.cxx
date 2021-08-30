#include "TMVA/RModelParser_PyTorch.h"


namespace TMVA{
namespace Experimental{
namespace SOFIE{

std::unordered_map<std::string, NodeType> NType =
    {
        {"'onnx::Gemm'", NodeType::GEMM},
        {"'onnx::Relu'", NodeType::RELU},
        {"'onnx::Transpose'", NodeType::TRANSPOSE}
    };


namespace PyTorch{

void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS){
   PyObject *fPyReturn = PyRun_String(code, Py_single_input, fGlobalNS, fLocalNS);
   if (!fPyReturn) {
      std::cout<<"Python error message:\n";
      PyErr_Print();
      throw std::runtime_error("Failed to run python code: "+code);
   }
 }

const char* PyStringAsString(PyObject* str){
   #if PY_MAJOR_VERSION < 3   // for Python2
      const char *stra_name = PyBytes_AsString(str);
      // need to add string delimiter for Python2
      TString sname = TString::Format("'%s'",stra_name);
      const char * name = sname.Data();
   #else   // for Python3
      PyObject* repr = PyObject_Repr(str);
      PyObject* stra = PyUnicode_AsEncodedString(repr, "utf-8", "~E~");
      const char *name = PyBytes_AsString(stra);
   #endif
return name;
}


RModel Parse(std::string filename, std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> inputDTypes){

    char sep = '/';
    #ifdef _WIN32
    sep = '\\';
    #endif

    size_t isep = filename.rfind(sep, filename.length());
    std::string filename_nodir = filename;
    if (isep != std::string::npos){
      filename_nodir = (filename.substr(isep+1, filename.length() - isep));
    }

    //Check on whether the ONNX file exists
    if(!std::ifstream(filename).good()){
        throw std::runtime_error("Model file "+filename_nodir+" not found!");
    }


    std::time_t ttime = std::time(0);
    std::tm* gmt_time = std::gmtime(&ttime);
    std::string parsetime (std::asctime(gmt_time));

    RModel rmodel(filename_nodir, parsetime);

    //Intializing Python Interpreter and scope dictionaries
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


    //Extracting model information
    //Model is converted to ONNX graph format
    //using PyTorch's internal function with the input shape provided
    PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
    PyRunString("import torch",fGlobalNS,fLocalNS);
    PyRunString("print('Torch Version: '+torch.__version__)",fGlobalNS,fLocalNS);
    PyRunString("from torch.onnx.utils import _model_to_graph",fGlobalNS,fLocalNS);
    PyRunString("from torch.onnx.symbolic_helper import _set_onnx_shape_inference",fGlobalNS,fLocalNS);
    PyRunString(TString::Format("model= torch.jit.load('%s')",filename.c_str()),fGlobalNS,fLocalNS);
    PyRunString("globals().update(locals())",fGlobalNS,fLocalNS);
    PyRunString("model.cpu()",fGlobalNS,fLocalNS);
    PyRunString("model.eval()",fGlobalNS,fLocalNS);

    //Building dummy inputs for the model
    PyRunString("dummyInputs=[]",fGlobalNS,fLocalNS);
    for(long unsigned int it=0;it<inputShapes.size();++it){
        PyRunString("inputShape=[]",fGlobalNS,fLocalNS);
        for(long unsigned int itr=0;itr<inputShapes[it].size();++itr){
            PyRunString(TString::Format("inputShape.append(%d)",(int)inputShapes[it][itr]),fGlobalNS,fLocalNS);
            }
        switch(inputDTypes[it]){
            case ETensorType::FLOAT:{
                PyRunString("dummyInputs.append(torch.rand(*inputShape))",fGlobalNS,fLocalNS);
                break;
            }
            default:
                throw std::runtime_error("Type Error: TMVA SOFIE does not yet support the input tensor data type"+ConvertTypeToString(inputDTypes[it]));
        }
        }

    //Finding example outputs from dummy
    PyRunString("output=model(*dummyInputs)",fGlobalNS,fLocalNS);

    //Getting the ONNX graph from model using the dummy inputs and example outputs
    PyRunString("_set_onnx_shape_inference(True)",fGlobalNS,fLocalNS);
    PyRunString("graph=_model_to_graph(model,dummyInputs,example_outputs=output)",fGlobalNS,fLocalNS);


    //Extracting Input tensor info
    PyRunString("inputs=[x for x in model.graph.inputs()]",fGlobalNS,fLocalNS);
    PyRunString("inputs=inputs[1:]",fGlobalNS,fLocalNS);
    PyRunString("inputNames=[x.debugName() for x in inputs]",fGlobalNS,fLocalNS);
    PyObject* fPInputs= PyDict_GetItemString(fLocalNS,"inputNames");
    std::string fInputName;
    std::vector<size_t>fInputShape;
    ETensorType fInputDType;
    for(Py_ssize_t inputIter=0; inputIter<PyList_Size(fPInputs);++inputIter){
        fInputName  = PyStringAsString(PyList_GetItem(fPInputs,inputIter));
        fInputShape = inputShapes[inputIter];
        fInputDType = inputDTypes[inputIter];
        switch(fInputDType){
            case(ETensorType::FLOAT): {
                rmodel.AddInputTensorInfo(fInputName, ETensorType::FLOAT, fInputShape);
                break;
            }
            default:
                throw std::runtime_error("Type Error: TMVA SOFIE does not yet support the input tensor data type"+ConvertTypeToString(fInputDType));
        }
        }


    //Extracting the model information in list modelData
    PyRunString("modelData=[]",fGlobalNS,fLocalNS);
    PyRunString("for i in graph[0].nodes():\n"
                "    globals().update(locals())\n"
                "    nodeData=[]\n"
                "    nodeData.append(i.kind())\n"
                "    nodeAttributeNames=[x for x in i.attributeNames()]\n"
                "    nodeAttributes={j:i[j] for j in nodeAttributeNames}\n"
                "    nodeData.append(nodeAttributes)\n"
                "    nodeInputs=[x for x in i.inputs()]\n"
                "    nodeInputNames=[x.debugName() for x in nodeInputs]\n"
                "    nodeData.append(nodeInputNames)\n"
                "    nodeOutputs=[x for x in i.outputs()]\n"
                "    nodeOutputNames=[x.debugName() for x in nodeOutputs]\n"
                "    nodeData.append(nodeOutputNames)\n"
                "    nodeDType=[x.type().scalarType() for x in nodeOutputs]\n"
                "    nodeData.append(nodeDType)\n"
                "    modelData.append(nodeData)",fGlobalNS,fLocalNS);

    PyObject* fPModel = PyDict_GetItemString(fLocalNS,"modelData");
    Py_ssize_t fPModelSize = PyList_Size(fPModel);
    PyObject *fNode,*fAttributes,*fInputs,*fOutputs;
    std::string fNodeType;
    ETensorType fNodeDType;
    for(Py_ssize_t fModelIterator=0;fModelIterator<fPModelSize;++fModelIterator){
        fNode     = PyList_GetItem(fPModel,fModelIterator);
        fNodeType = PyStringAsString(PyList_GetItem(fNode,0));

        if(NType.find(fNodeType)==NType.end())
            throw std::runtime_error("Layer error: TMVA SOFIE does not yet suppport layer type"+fNodeType);

        fAttributes = PyList_GetItem(fNode,1);
        fInputs     = PyList_GetItem(fNode,2);
        fOutputs    = PyList_GetItem(fNode,3);
        fNodeDType  = ConvertStringToType(PyStringAsString(PyList_GetItem(PyList_GetItem(fNode,4),0)));

        switch(NType.find(fNodeType)->second){
            case NodeType::GEMM : {
                float attr_alpha = (float)(PyFloat_AsDouble(PyDict_GetItemString(fAttributes,"alpha")));
                float attr_beta = (float)(PyFloat_AsDouble(PyDict_GetItemString(fAttributes,"beta")));
                int_t attr_transA;
                int_t attr_transB;

                if(PyDict_Contains(fAttributes,PyUnicode_FromString("transB"))){
                         attr_transB=(int_t)(PyLong_AsLong(PyDict_GetItemString(fAttributes,"transB")));
                         attr_transA=!attr_transB;
                    }
                else{
                        attr_transA=(int_t)(PyLong_AsLong(PyDict_GetItemString(fAttributes,"transA")));
                        attr_transB=!attr_transA;
                    }
                switch(fNodeDType){
                case ETensorType::FLOAT: {
                    std::unique_ptr<ROperator> op;
                    op.reset(new ROperator_Gemm<float>(attr_alpha, attr_beta, attr_transA, attr_transB, PyStringAsString(PyList_GetItem(fInputs,0)), PyStringAsString(PyList_GetItem(fInputs,1)), PyStringAsString(PyList_GetItem(fInputs,2)), PyStringAsString(PyList_GetItem(fOutputs,0))));
                    rmodel.AddOperator(std::move(op));
                    break;
                    }
                default:
                    throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + ConvertTypeToString(fNodeDType));

                }
                break;
                }

            case NodeType::RELU : {
                switch(fNodeDType){
                    case ETensorType::FLOAT: {
                        std::unique_ptr<ROperator> op;
                        op.reset(new ROperator_Relu<float>(PyStringAsString(PyList_GetItem(fInputs,0)), PyStringAsString(PyList_GetItem(fOutputs,0))));
                        rmodel.AddOperator(std::move(op));
                        break;
                    }
                    default:
                    throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + ConvertTypeToString(fNodeDType));
                }
                break;
                }

            case NodeType::TRANSPOSE:{
                std::vector<int_t> fAttributePermute;
                PyObject* fPPermute=PyDict_GetItemString(fAttributes,"perm");
                for(Py_ssize_t permIter=0; permIter<PyList_Size(fPPermute);++permIter){
                    fAttributePermute.push_back((int_t)PyLong_AsLong(PyList_GetItem(fPPermute,permIter)));
                }
                switch(fNodeDType){
                case ETensorType::FLOAT: {
                std::unique_ptr<ROperator> op;
                op.reset(new ROperator_Transpose<float>(fAttributePermute, PyStringAsString(PyList_GetItem(fInputs,0)), PyStringAsString(PyList_GetItem(fOutputs,0))));
                rmodel.AddOperator(std::move(op));
                break;
                }
                default:
                    throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + ConvertTypeToString(fNodeDType));
                }
                break;
                }

            default:
                throw std::runtime_error("Node Error: TMVA SOFIE does not yet support node type " + fNodeType);
            }
            }


    //Extracting model weights to add the initialized tensors to the RModel
    PyRunString("weightNames=[k for k in graph[1].keys()]",fGlobalNS,fLocalNS);
    PyRunString("weights=[v.numpy() for v in graph[1].values()]",fGlobalNS,fLocalNS);
    PyRunString("weightDTypes=[v.type()[6:-6] for v in graph[1].values()]",fGlobalNS,fLocalNS);
    PyObject* fPWeightNames = PyDict_GetItemString(fLocalNS,"weightNames");
    PyObject* fPWeightTensors = PyDict_GetItemString(fLocalNS,"weights");
    PyObject* fPWeightDTypes = PyDict_GetItemString(fLocalNS,"weightDTypes");
    PyArrayObject* fWeightTensor;
    std::string fWeightName;
    ETensorType fWeightDType;
    std::vector<std::size_t> fWeightShape;
    std::size_t fWeightSize;

    for(Py_ssize_t weightIter=0; weightIter<PyList_Size(fPWeightTensors);++weightIter){
        fWeightTensor = (PyArrayObject*)PyList_GetItem(fPWeightTensors,weightIter);
        fWeightName   = PyStringAsString(PyList_GetItem(fPWeightNames,weightIter));
        fWeightDType  = ConvertStringToType(PyStringAsString(PyList_GetItem(fPWeightDTypes,weightIter)));
        fWeightSize   = 1;
        fWeightShape.clear();
        for(int j=0; j<PyArray_NDIM(fWeightTensor); ++j){
            fWeightShape.push_back((std::size_t)(PyArray_DIM(fWeightTensor,j)));
            fWeightSize*=(std::size_t)(PyArray_DIM(fWeightTensor,j));
        }
        switch(fWeightDType){
            case ETensorType::FLOAT:{
                float* fWeightValue = (float*)PyArray_DATA(fWeightTensor);
                std::shared_ptr<void> fData(malloc(fWeightSize * sizeof(float)), free);
                std::memcpy(fData.get(),fWeightValue,fWeightSize * sizeof(float));
                rmodel.AddInitializedTensor(fWeightName, ETensorType::FLOAT,fWeightShape,fData);
                break;
                }
            default:
                throw std::runtime_error("Type error: TMVA SOFIE does not yet supports weights of data type"+ConvertTypeToString(fWeightDType));
            }
    }


    //Extracting output tensor names
    PyRunString("outputs=[x for x in graph[0].outputs()]",fGlobalNS,fLocalNS);
    PyRunString("outputNames=[x.debugName() for x in outputs]",fGlobalNS,fLocalNS);
    PyObject* fPOutputs= PyDict_GetItemString(fLocalNS,"outputNames");
    std::vector<std::string> fOutputNames;
    for(Py_ssize_t outputIter = 0; outputIter < PyList_Size(fPOutputs);++outputIter){
        fOutputNames.push_back(PyStringAsString(PyList_GetItem(fPOutputs,outputIter)));
        }
    rmodel.AddOutputTensorNameList(fOutputNames);

    return rmodel;
}

RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes){
      std::vector<ETensorType> dtype(inputShapes.size(),ETensorType::FLOAT);
      return Parse(filepath,inputShapes,dtype);
    }
}
}
}
}
