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

enum class NodeType{
   GEMM = 0, RELU = 1, TRANSPOSE = 2 //order sensitive

};

namespace PyTorch{

    void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS);
    const char* PyStringAsString(PyObject* str);
    RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes, std::vector<ETensorType> dtype);
    RModel Parse(std::string filepath,std::vector<std::vector<size_t>> inputShapes);
  }
}
}
}

#endif //TMVA_PYMVA_RMODELPARSER_PYTORCH
