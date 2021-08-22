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
namespace PyKeras{

namespace INTERNAL{
   std::unique_ptr<ROperator> AddKerasLayer(PyObject* fLayer);
   std::unique_ptr<ROperator> AddKerasActivation(PyObject* fLayer);
   std::unique_ptr<ROperator> AddKerasReLU(PyObject* fLayer);
   std::unique_ptr<ROperator> AddKerasPermute(PyObject* fLayer);

   std::pair<std::unique_ptr<ROperator>,std::unique_ptr<ROperator>> AddKerasLayerWithActivation(PyObject* fLayer);
   std::unique_ptr<ROperator> AddKerasDense(PyObject* fLayer);
   //std::unique_ptr<ROperator> AddKerasConv(PyObject* fLayer);

   using KerasMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fLayer)>;
   using KerasMethodMapWithActivation = std::unordered_map<std::string, std::unique_ptr<ROperator>(*)(PyObject* fLayer)>;

   const KerasMethodMap mapKerasLayer =
    {
        {"'Activation'", &AddKerasActivation},
        {"'Permute'", &AddKerasPermute},

        //For activation layers
        {"'ReLU'", &AddKerasReLU},

        //For layers with activation attributes
        {"'relu'", &AddKerasReLU}
    };

    const KerasMethodMapWithActivation mapKerasLayerWithActivation =
    {
        {"'Dense'", &AddKerasDense},
        //{"'Convolution'",&add_keras_conv}
    };
}//INTERNAL


    void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS);
    const char* PyStringAsString(PyObject* str);
    std::vector<size_t> getShapeFromTuple(PyObject* shapeTuple);
    RModel Parse(std::string filepath);
}//PyKeras
}//SOFIE
}//Experimental
}//TMVA
#endif //TMVA_PYMVA_RMODELPARSER_KERAS
