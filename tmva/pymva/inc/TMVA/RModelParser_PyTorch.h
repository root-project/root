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

#include "TMVA/PyMethodBase.h"

#include "Rtypes.h"
#include "TString.h"


namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyTorch{

// Referencing Python utility functions present in PyMethodBase
static void(& PyRunString)(TString, PyObject*, PyObject*) = PyMethodBase::PyRunString;
static const char*(& PyStringAsString)(PyObject*) = PyMethodBase::PyStringAsString;

namespace INTERNAL{
   // For searching and calling specific preparatory function for PyTorch ONNX Graph's node
   std::unique_ptr<ROperator> MakePyTorchNode(PyObject* fNode);

   std::unique_ptr<ROperator> MakePyTorchGemm(PyObject* fNode);      // For instantiating ROperator for PyTorch ONNX's Gemm operator
   std::unique_ptr<ROperator> MakePyTorchRelu(PyObject* fNode);      // For instantiating ROperator for PyTorch ONNX's Relu operator
   std::unique_ptr<ROperator> MakePyTorchTranspose(PyObject* fNode); // For instantiating ROperator for PyTorch ONNX's Transpose operator

   // For mapping PyTorch ONNX Graph's Node with the preparatory functions for ROperators
   using PyTorchMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fNode)>;

   const PyTorchMethodMap mapPyTorchNode =
    {
        {"onnx::Gemm",      &MakePyTorchGemm},
        {"onnx::Relu",      &MakePyTorchRelu},
        {"onnx::Transpose", &MakePyTorchTranspose}
    };

}//INTERNAL

/// Parser function for translatng PyTorch .pt model into a RModel object.
/// Accepts the file location of a PyTorch model, shapes and data-types of input tensors
/// and returns the equivalent RModel object.
RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> dtype);

/// Overloaded Parser function for translatng PyTorch .pt model into a RModel object.
/// Accepts the file location of a PyTorch model and only the shapes of input tensors.
/// Builds the data-types vector for input tensors and calls the `Parse()` function to
/// return the equivalent RModel object.
RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes);

}//PyTorch
}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_PYMVA_RMODELPARSER_PYTORCH
