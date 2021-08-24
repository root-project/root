// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta, 2021

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Functionality for parsing a saved PyTorch .PT model into RModel object    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Sanjiban Sengupta <sanjiban.sg@gmail.com>                                 *
 *                                                                                *
 * Copyright (c) 2021:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/


#ifndef TMVA_SOFIE_RMODELPARSER_PYTORCH
#define TMVA_SOFIE_RMODELPARSER_PYTORCH

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"
#include "TMVA/Types.h"
#include "TMVA/OperatorList.hxx"

#include "Rtypes.h"
#include "TString.h"


namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyTorch{

namespace INTERNAL{
   std::unique_ptr<ROperator> MakePyTorchNode(PyObject* fNode);
   std::unique_ptr<ROperator> MakePyTorchGemm(PyObject* fNode);
   std::unique_ptr<ROperator> MakePyTorchRelu(PyObject* fNode);
   std::unique_ptr<ROperator> MakePyTorchTranspose(PyObject* fNode);

   using PyTorchMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fNode)>;

   const PyTorchMethodMap mapPyTorchNode =
    {
        {"'onnx::Gemm'",      &MakePyTorchGemm},
        {"'onnx::Relu'",      &MakePyTorchRelu},
        {"'onnx::Transpose'", &MakePyTorchTranspose}
    };

}//INTERNAL


const char* PyStringAsString(PyObject* str);
void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS);

RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> dtype);
RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes);
}//PyTorch
}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_PYMVA_RMODELPARSER_PYTORCH
