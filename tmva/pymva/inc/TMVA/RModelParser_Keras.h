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
   void AddKerasLayer(RModel& rmodel, PyObject* fLayer);
   std::unique_ptr<ROperator> MakeKerasActivation(PyObject* fLayer);
   std::unique_ptr<ROperator> MakeKerasReLU(PyObject* fLayer);
   std::unique_ptr<ROperator> MakeKerasPermute(PyObject* fLayer);

   std::unique_ptr<ROperator> MakeKerasDense(PyObject* fLayer);
   //std::unique_ptr<ROperator> AddKerasConv(PyObject* fLayer);

   using KerasMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fLayer)>;
   using KerasMethodMapWithActivation = std::unordered_map<std::string, std::unique_ptr<ROperator>(*)(PyObject* fLayer)>;

   const KerasMethodMap mapKerasLayer =
    {
        {"'Activation'", &MakeKerasActivation},
        {"'Permute'", &MakeKerasPermute},

        //For activation layers
        {"'ReLU'", &MakeKerasReLU},

        //For layers with activation attributes
        {"'relu'", &MakeKerasReLU}
    };

    const KerasMethodMapWithActivation mapKerasLayerWithActivation =
    {
        {"'Dense'", &MakeKerasDense},
        //{"'Convolution'",&add_keras_conv}
    };

    std::vector<size_t> GetShapeFromTuple(PyObject* shapeTuple);

}//INTERNAL
const char* PyStringAsString(PyObject* str);
void PyRunString(TString code, PyObject *fGlobalNS, PyObject *fLocalNS);

RModel Parse(std::string filepath);

}//PyKeras
}//SOFIE
}//Experimental
}//TMVA
#endif //TMVA_PYMVA_RMODELPARSER_KERAS
