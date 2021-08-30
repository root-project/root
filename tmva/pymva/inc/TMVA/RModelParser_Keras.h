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

#include "TMVA/PyMethodBase.h"

#include "Rtypes.h"
#include "TString.h"


namespace TMVA{
namespace Experimental{
namespace SOFIE{
namespace PyKeras{


// Referencing Python utility functions present in PyMethodBase
static void(& PyRunString)(TString, PyObject*, PyObject*) = PyMethodBase::PyRunString;
static const char*(& PyStringAsString)(PyObject*) = PyMethodBase::PyStringAsString;


namespace INTERNAL{
   // For adding Keras layer into RModel object
   void AddKerasLayer(RModel& rmodel, PyObject* fLayer);


   // Declaring Internal Functions for Keras layers which don't have activation as an additional attribute
   std::unique_ptr<ROperator> MakeKerasActivation(PyObject* fLayer);  // For instantiating ROperator for Keras Activation Layer
   std::unique_ptr<ROperator> MakeKerasReLU(PyObject* fLayer);       // For instantiating ROperator for Keras ReLU layer
   std::unique_ptr<ROperator> MakeKerasPermute(PyObject* fLayer);   // For instantiating ROperator for Keras Permute Layer


   // Declaring Internal function for Keras layers which have additional activation attribute
   std::unique_ptr<ROperator> MakeKerasDense(PyObject* fLayer);   // For instantiating ROperator for Keras Dense Layer

   // For mapping Keras layer with the preparatory functions for ROperators
   using KerasMethodMap = std::unordered_map<std::string, std::unique_ptr<ROperator> (*)(PyObject* fLayer)>;
   using KerasMethodMapWithActivation = std::unordered_map<std::string, std::unique_ptr<ROperator>(*)(PyObject* fLayer)>;

   const KerasMethodMap mapKerasLayer =
    {
        {"Activation", &MakeKerasActivation},
        {"Permute", &MakeKerasPermute},

        //For activation layers
        {"ReLU", &MakeKerasReLU},

        //For layers with activation attributes
        {"relu", &MakeKerasReLU}
    };

    const KerasMethodMapWithActivation mapKerasLayerWithActivation =
    {
        {"Dense", &MakeKerasDense},
    };

    // Function which returns values from a Python Tuple object in vector of size_t
    std::vector<size_t> GetShapeFromTuple(PyObject* shapeTuple);

}//INTERNAL

/// Parser function for translatng Keras .h5 model into a RModel object.
/// Accepts the file location of a Keras model and returns the
/// equivalent RModel object.
RModel Parse(std::string filename);

}//PyKeras
}//SOFIE
}//Experimental
}//TMVA
#endif //TMVA_PYMVA_RMODELPARSER_KERAS
