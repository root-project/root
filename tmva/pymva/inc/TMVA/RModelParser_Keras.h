// @(#)root/tmva/pymva $Id$
// Author: Sanjiban Sengupta, 2021

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Functionality for parsing a saved Keras .H5 model into RModel object      *
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


#ifndef TMVA_SOFIE_RMODELPARSER_KERAS
#define TMVA_SOFIE_RMODELPARSER_KERAS

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

enum class LayerType{
   DENSE = 0, ACTIVATION = 1, RELU = 2, TRANSPOSE = 3 //order sensitive

};

namespace INTERNAL{
   std::unique_ptr<ROperator> make_ROperator_Gemm(std::string input,std::string output,std::string kernel,std::string bias,std::string dtype);
   std::unique_ptr<ROperator> make_ROperator_Relu(std::string input, std::string output, std::string dtype);
   std::unique_ptr<ROperator> make_ROperator_Transpose(std::string input, std::string output, std::vector<int_t> dims, std::string dtype);

   const std::unordered_map<std::string, LayerType> Type =
    {
        {"'Dense'", LayerType::DENSE},
        {"'Activation'", LayerType::ACTIVATION},
        {"'ReLU'", LayerType::RELU},
        {"'Permute'", LayerType::TRANSPOSE}
    };

  const std::unordered_map<std::string, LayerType> ActivationType =
    {
        {"'relu'", LayerType::RELU},
    };

}

namespace PyKeras{
    void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS);
    const char* PyStringAsString(PyObject* str);
    std::vector<size_t> getShapeFromTuple(PyObject* shapeTuple);
    RModel Parse(std::string filepath);
  };
}
}
}

#endif //TMVA_PYMVA_RMODELPARSER_KERAS
