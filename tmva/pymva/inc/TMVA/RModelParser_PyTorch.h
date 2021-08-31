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
