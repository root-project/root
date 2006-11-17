// @(#)root/tmva $Id: Types.cxx,v 1.11 2006/11/17 00:21:35 stelzer Exp $   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Types                                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Joerg Stelzer   <Joerg.Stelzer@cern.ch>  - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#include "TMVA/Types.h"

TMVA::Types* TMVA::Types::fgTypesPtr = 0;

TMVA::Types::Types()
   : fLogger( "Types" )
{
   // constructor
   // fill map that links method names and enums

   fStr2type["Variable"]        = Types::kVariable;
   fStr2type["Cuts"]            = Types::kCuts;
   fStr2type["Likelihood"]      = Types::kLikelihood;
   fStr2type["PDERS"]           = Types::kPDERS;
   fStr2type["HMatrix"]         = Types::kHMatrix;
   fStr2type["Fisher"]          = Types::kFisher;
   fStr2type["CFMlpANN"]        = Types::kCFMlpANN;
   fStr2type["TMlpANN"]         = Types::kTMlpANN;
   fStr2type["BDT"]             = Types::kBDT;
   fStr2type["RuleFit"]         = Types::kRuleFit;
   fStr2type["SVM"]             = Types::kSVM;
   fStr2type["MLP"]             = Types::kMLP;
   fStr2type["BayesClassifier"] = Types::kBayesClassifier;
   fStr2type["Committee"]       = Types::kCommittee;
}
