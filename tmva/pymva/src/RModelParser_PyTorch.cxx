// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta 2021

/**********************************************************************************
 * Project : TMVA - a Root-integrated toolkit for multivariate data analysis      *
 * Package : TMVA                                                                 *
 * Function: TMVA::Experimental::SOFIE::PyTorch::Parse                            *
 *                                                                                *
 * Description:                                                                   *
 *      Parser function for translating PyTorch .pt model to RModel object        *
 *                                                                                *
 * Example Usage:                                                                 *
 * ~~~ {.cpp}                                                                     *
 * using TMVA::Experimental::SOFIE;                                               *
 * // Building the vector of input tensor shapes                                  *
 * std::vector<size_t> s1{120,1};                                                 *
 * std::vector<std::vector<size_t>> inputShape{s1};                               *
 * RModel model = PyTorch::Parse("trained_model_dense.pt",inputShape);            *
 * ~~~                                                                            *
 *                                                                                *
 **********************************************************************************/


#include "TMVA/RModelParser_PyTorch.h"

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyTorch{

// Referencing Python utility functions present in PyMethodBase
static void(& PyRunString)(TString, PyObject*, PyObject*) = PyMethodBase::PyRunString;
static const char*(& PyStringAsString)(PyObject*) = PyMethodBase::PyStringAsString;
static std::vector<size_t>(& GetDataFromList)(PyObject*) = PyMethodBase::GetDataFromList;


namespace INTERNAL{

// For searching and calling specific preparatory function for PyTorch ONNX Graph's node
std::unique_ptr<ROperator> MakePyTorchNode(PyObject* fNode);

std::unique_ptr<ROperator> MakePyTorchGemm(PyObject* fNode);      // For instantiating ROperator for PyTorch ONNX's Gemm operator
std::unique_ptr<ROperator> MakePyTorchRelu(PyObject* fNode);      // For instantiating ROperator for PyTorch ONNX's Relu operator
std::unique_ptr<ROperator> MakePyTorchTranspose(PyObject* fNode); // For instantiating ROperator for PyTorch ONNX's Transpose operator
std::unique_ptr<ROperator> MakePyTorchConv(PyObject* fNode);      // For instantiating ROperator for PyTorch ONNX's Conv operator

// For mapping PyTorch ONNX Graph's Node with the preparatory functions for ROperators
using PyTorchMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fNode)>;

const PyTorchMethodMap mapPyTorchNode =
{
    {"onnx::Gemm",      &MakePyTorchGemm},
    {"onnx::Relu",      &MakePyTorchRelu},
    {"onnx::Transpose", &MakePyTorchTranspose},
    {"onnx::Conv",      &MakePyTorchConv}
};


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares equivalent ROperator with respect to PyTorch ONNX node.
///
/// \param[in] fNode Python PyTorch ONNX Graph node
/// \return unique pointer to ROperator object
///
/// Function searches for the passed PyTorch ONNX Graph node in the map, and calls
/// the specific preparatory function, subsequently returning the ROperator object.
///
/// For developing new preparatory functions for supporting PyTorch ONNX Graph nodes
/// in future,  all one needs is to extract the required properties and attributes
/// from the fNode dictionary which contains all the information about any PyTorch ONNX
//  Graph node and after any required transformations, these are passed for instantiating
/// the ROperator object.
///
/// The fNode dictionary which holds all the information about a PyTorch ONNX Graph's node has
/// following structure:-
///
///     dict fNode {  'nodeType'        : Type of node (operator)
///                   'nodeAttributes'  : Attributes of the node
///                   'nodeInputs'      : List of names of input tensors
///                   'nodeOutputs'     : List of names of output tensors
///                   'nodeDType'       : Data-type of the operator node
///                }
///
std::unique_ptr<ROperator> MakePyTorchNode(PyObject* fNode){
        std::string fNodeType = PyStringAsString(PyDict_GetItemString(fNode,"nodeType"));
        auto findNode = mapPyTorchNode.find(fNodeType);
        if(findNode == mapPyTorchNode.end()){
            throw std::runtime_error("TMVA::SOFIE - Parsing PyTorch node " +fNodeType+" is not yet supported ");
        }
        return (findNode->second)(fNode);
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator_Gemm object
///
/// \param[in] fNode Python PyTorch ONNX Graph node
/// \return Unique pointer to ROperator object
///
/// For PyTorch's Linear layer having Gemm operation in its ONNX graph,
/// the names of the input tensor, output tensor are extracted, and then
/// are passed to instantiate a ROperator_Gemm object using the required attributes.
/// fInputs is a list of tensor names, which includes the names of the input tensor
/// and the weight tensors.
std::unique_ptr<ROperator> MakePyTorchGemm(PyObject* fNode){
        PyObject* fAttributes   = PyDict_GetItemString(fNode,"nodeAttributes");
        PyObject* fInputs       = PyDict_GetItemString(fNode,"nodeInputs");
        PyObject* fOutputs      = PyDict_GetItemString(fNode,"nodeOutputs");
        std::string fNodeDType  = PyStringAsString(PyList_GetItem(PyDict_GetItemString(fNode,"nodeDType"),0));

        // Extracting the parameters for Gemm Operator
        std::string fNameA = PyStringAsString(PyList_GetItem(fInputs,0));
        std::string fNameB = PyStringAsString(PyList_GetItem(fInputs,1));
        std::string fNameC = PyStringAsString(PyList_GetItem(fInputs,2));
        std::string fNameY = PyStringAsString(PyList_GetItem(fOutputs,0));
        float fAttrAlpha = (float)(PyFloat_AsDouble(PyDict_GetItemString(fAttributes,"alpha")));
        float fAttrBeta = (float)(PyFloat_AsDouble(PyDict_GetItemString(fAttributes,"beta")));
        int_t fAttrTransA;
        int_t fAttrTransB;

        if(PyDict_Contains(fAttributes,PyUnicode_FromString("transB"))){
            fAttrTransB = (int_t)(PyLong_AsLong(PyDict_GetItemString(fAttributes,"transB")));
            fAttrTransA = !fAttrTransB;
        }
        else{
            fAttrTransA=(int_t)(PyLong_AsLong(PyDict_GetItemString(fAttributes,"transA")));
            fAttrTransB = !fAttrTransA;
        }

        std::unique_ptr<ROperator> op;
        switch(ConvertStringToType(fNodeDType)){
            case ETensorType::FLOAT: {
                op.reset(new ROperator_Gemm<float>(fAttrAlpha, fAttrBeta, fAttrTransA, fAttrTransB, fNameA, fNameB, fNameC, fNameY ));
                break;
                }
                default:
                    throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Gemm does not yet support input type " + fNodeDType);
                }
        return op;
}

//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator_Relu object
///
/// \param[in] fNode Python PyTorch ONNX Graph node
/// \return Unique pointer to ROperator object
///
/// For instantiating a ROperator_Relu object, the names of
/// input & output tensors and the data-type of the Graph node
/// are extracted.
std::unique_ptr<ROperator> MakePyTorchRelu(PyObject* fNode){
        PyObject* fInputs       = PyDict_GetItemString(fNode,"nodeInputs");
        PyObject* fOutputs      = PyDict_GetItemString(fNode,"nodeOutputs");

        std::string fNodeDType  = PyStringAsString(PyList_GetItem(PyDict_GetItemString(fNode,"nodeDType"),0));
        std::string fNameX      = PyStringAsString(PyList_GetItem(fInputs,0));
        std::string fNameY      = PyStringAsString(PyList_GetItem(fOutputs,0));
        std::unique_ptr<ROperator> op;
        switch(ConvertStringToType(fNodeDType)){
            case ETensorType::FLOAT: {
                op.reset(new ROperator_Relu<float>(fNameX,fNameY));
                break;
                }
                default:
                throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Relu does not yet support input type " + fNodeDType);
        }
        return op;
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator_Transpose object
///
/// \param[in] fNode Python PyTorch ONNX Graph node
/// \return Unique pointer to ROperator object
///
/// For Transpose Operator of PyTorch's ONNX Graph, the permute dimensions are found,
/// and are passed in instantiating the ROperator object.
std::unique_ptr<ROperator> MakePyTorchTranspose(PyObject* fNode){
        PyObject* fAttributes   = PyDict_GetItemString(fNode,"nodeAttributes");
        PyObject* fInputs       = PyDict_GetItemString(fNode,"nodeInputs");
        PyObject* fOutputs      = PyDict_GetItemString(fNode,"nodeOutputs");
        std::string fNodeDType  = PyStringAsString(PyList_GetItem(PyDict_GetItemString(fNode,"nodeDType"),0));

        // Extracting the Permute dimensions for transpose
        std::vector<int_t> fAttrPermute;
        PyObject* fPermute=PyDict_GetItemString(fAttributes,"perm");
        for(Py_ssize_t permIter=0; permIter<PyList_Size(fPermute);++permIter){
            fAttrPermute.push_back((int_t)PyLong_AsLong(PyList_GetItem(fPermute,permIter)));
        }
        std::string fNameData   = PyStringAsString(PyList_GetItem(fInputs,0));
        std::string fNameOutput = PyStringAsString(PyList_GetItem(fOutputs,0));

        std::unique_ptr<ROperator> op;
        switch(ConvertStringToType(fNodeDType)){
            case ETensorType::FLOAT: {
                op.reset(new ROperator_Transpose<float>(fAttrPermute, fNameData, fNameOutput));
                break;
            }
            default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Transpose does not yet support input type " + fNodeDType);
        }
        return op;
}


//////////////////////////////////////////////////////////////////////////////////
/// \brief Prepares a ROperator_Conv object
///
/// \param[in] fNode Python PyTorch ONNX Graph node
/// \return Unique pointer to ROperator object
///
/// For Conv Operator of PyTorch's ONNX Graph, attributes like dilations, group,
/// kernel shape, pads and strides are found, and are passed in instantiating the
/// ROperator object with autopad default to `NOTSET`.
std::unique_ptr<ROperator> MakePyTorchConv(PyObject* fNode){
        PyObject* fAttributes   = PyDict_GetItemString(fNode,"nodeAttributes");
        PyObject* fInputs       = PyDict_GetItemString(fNode,"nodeInputs");
        PyObject* fOutputs      = PyDict_GetItemString(fNode,"nodeOutputs");
        std::string fNodeDType  = PyStringAsString(PyList_GetItem(PyDict_GetItemString(fNode,"nodeDType"),0));

        // Extracting the Conv Node Attributes
        PyObject* fDilations       = PyDict_GetItemString(fAttributes,"dilations");
        PyObject* fGroup           = PyDict_GetItemString(fAttributes,"group");
        PyObject* fKernelShape     = PyDict_GetItemString(fAttributes,"kernel_shape");
        PyObject* fPads            = PyDict_GetItemString(fAttributes,"pads");
        PyObject* fStrides         = PyDict_GetItemString(fAttributes,"strides");

        std::string fAttrAutopad = "NOTSET";
        std::vector<size_t> fAttrDilations = GetDataFromList(fDilations);
        size_t fAttrGroup = PyLong_AsLong(fGroup);
        std::vector<size_t> fAttrKernelShape = GetDataFromList(fKernelShape);
        std::vector<size_t> fAttrPads        = GetDataFromList(fPads);
        std::vector<size_t> fAttrStrides     = GetDataFromList(fStrides);
        std::string nameX = PyStringAsString(PyList_GetItem(fInputs,0));
        std::string nameW = PyStringAsString(PyList_GetItem(fInputs,1));
        std::string nameB = PyStringAsString(PyList_GetItem(fInputs,2));
        std::string nameY = PyStringAsString(PyList_GetItem(fOutputs,0));

        std::unique_ptr<ROperator> op;
        switch(ConvertStringToType(fNodeDType)){
            case ETensorType::FLOAT: {
                op.reset(new ROperator_Conv<float>(fAttrAutopad, fAttrDilations, fAttrGroup, fAttrKernelShape, fAttrPads, fAttrStrides, nameX, nameW, nameB, nameY));
                break;
            }
            default:
            throw std::runtime_error("TMVA::SOFIE - Unsupported - Operator Conv does not yet support input type " + fNodeDType);
        }
        return op;
}
}//INTERNAL


//////////////////////////////////////////////////////////////////////////////////
/// \param[in] filename file location of PyTorch .pt model
/// \param[in] inputShapes vector of input shape vectors
/// \param[in] inputDTypes vector of ETensorType for data-types of Input tensors
/// \return    Parsed RModel object
///
/// The `Parse()` function defined in `TMVA::Experimental::SOFIE::PyTorch` will
/// parse a trained PyTorch .pt model into a RModel Object. The parser uses
/// internal functions of PyTorch to convert any PyTorch model into its
/// equivalent ONNX Graph. For this conversion, dummy inputs are built which are
/// passed through the model and the applied operators are recorded for populating
/// the ONNX graph. The `Parse()` function requires the shapes and data-types of
/// the input tensors which are used for building the dummy inputs.
/// After the said conversion, the nodes of the ONNX graph are then traversed to
/// extract properties like Node type, Attributes, input & output tensor names.
/// Function `AddOperator()` is then called on the extracted nodes to add the
/// operator into the RModel object. The nodes are also checked for adding any
/// required routines for executing the generated Inference code.
///
/// The internal function used to convert the model to graph object returns a list
/// which contains a Graph object and a dictionary of weights. This dictionary is
/// used to extract the Initialized tensors for the model. The names and data-types
/// of the Initialized tensors are extracted along with their values in NumPy array,
/// and after approapriate type-conversions, they are added into the RModel object.
///
/// For adding the Input tensor infos, the names of the input tensors are extracted
/// from the PyTorch ONNX graph object. The vector of shapes & data-types passed
/// into the `Parse()` function are used to extract the data-type and the shape
/// of the input tensors. Extracted input tensor infos are then added into the
/// RModel object by calling the `AddInputTensorInfo()` function.
///
/// For the output tensor infos, names of the output tensors are also extracted
/// from the Graph object and are then added into the RModel object by calling the
/// AddOutputTensorNameList() function.
///
/// Example Usage:
/// ~~~ {.cpp}
/// using TMVA::Experimental::SOFIE;
/// //Building the vector of input tensor shapes
/// std::vector<size_t> s1{120,1};
/// std::vector<std::vector<size_t>> inputShape{s1};
/// RModel model = PyTorch::Parse("trained_model_dense.pt",inputShape);
/// ~~~
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

    //Check on whether the PyTorch .pt file exists
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
        PyRunString("dummyInputs.append(torch.rand(*inputShape))",fGlobalNS,fLocalNS);
    }

    //Finding example outputs from dummy
    PyRunString("output=model(*dummyInputs)",fGlobalNS,fLocalNS);

    //Getting the ONNX graph from model using the dummy inputs and example outputs
    PyRunString("_set_onnx_shape_inference(True)",fGlobalNS,fLocalNS);
    PyRunString("graph=_model_to_graph(model,dummyInputs,example_outputs=output)",fGlobalNS,fLocalNS);


    //Extracting the model information in list modelData
    PyRunString("modelData=[]",fGlobalNS,fLocalNS);
    PyRunString("for i in graph[0].nodes():\n"
                "    globals().update(locals())\n"
                "    nodeData={}\n"
                "    nodeData['nodeType']=i.kind()\n"
                "    nodeAttributeNames=[x for x in i.attributeNames()]\n"
                "    nodeAttributes={j:i[j] for j in nodeAttributeNames}\n"
                "    nodeData['nodeAttributes']=nodeAttributes\n"
                "    nodeInputs=[x for x in i.inputs()]\n"
                "    nodeInputNames=[x.debugName() for x in nodeInputs]\n"
                "    nodeData['nodeInputs']=nodeInputNames\n"
                "    nodeOutputs=[x for x in i.outputs()]\n"
                "    nodeOutputNames=[x.debugName() for x in nodeOutputs]\n"
                "    nodeData['nodeOutputs']=nodeOutputNames\n"
                "    nodeDType=[x.type().scalarType() for x in nodeOutputs]\n"
                "    nodeData['nodeDType']=nodeDType\n"
                "    modelData.append(nodeData)",fGlobalNS,fLocalNS);

    PyObject* fPModel = PyDict_GetItemString(fLocalNS,"modelData");
    Py_ssize_t fPModelSize = PyList_Size(fPModel);
    PyObject *fNode;
    std::string fNodeType;


    //Adding operators into the RModel object
    for(Py_ssize_t fModelIterator=0;fModelIterator<fPModelSize;++fModelIterator){
        fNode     = PyList_GetItem(fPModel,fModelIterator);
        fNodeType = PyStringAsString(PyDict_GetItemString(fNode,"nodeType"));

        // Adding required routines for inference code generation
        if(fNodeType == "onnx::Gemm"){
            rmodel.AddBlasRoutines({"Gemm", "Gemv"});
        }
        else if (fNodeType == "onnx::Conv") {
         rmodel.AddBlasRoutines({"Gemm", "Axpy"});
        }
        rmodel.AddOperator(INTERNAL::MakePyTorchNode(fNode));
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

//////////////////////////////////////////////////////////////////////////////////
/// \param[in] filename file location of PyTorch .pt model
/// \param[in] inputShapes vector of input shape vectors
/// \return    Parsed RModel object
///
/// Overloaded Parser function for translating PyTorch .pt model to RModel object.
/// Function only requires the inputShapes vector as a parameter. Function
/// builds the vector of Data-types for the input tensors using Float as default,
/// Function calls the `Parse()` function with the vector of data-types included,
/// subsequently returning the parsed RModel object.
RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes){
      std::vector<ETensorType> dtype(inputShapes.size(),ETensorType::FLOAT);
      return Parse(filepath,inputShapes,dtype);
}
}//PyTorch
}//SOFIE
}//Experimental
}//TMVA
