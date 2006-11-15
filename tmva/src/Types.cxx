// @(#)root/tmva $Id: Types.cxx,v 1.8 2006/10/26 19:55:40 andreas.hoecker Exp $   
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-KP Heidelberg, Germany,                                               * 
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
   fStr2type["Variable"]        = Types::Variable;
   fStr2type["Cuts"]            = Types::Cuts;
   fStr2type["Likelihood"]      = Types::Likelihood;
   fStr2type["PDERS"]           = Types::PDERS;
   fStr2type["HMatrix"]         = Types::HMatrix;
   fStr2type["Fisher"]          = Types::Fisher;
   fStr2type["CFMlpANN"]        = Types::CFMlpANN;
   fStr2type["TMlpANN"]         = Types::TMlpANN;
   fStr2type["BDT"]             = Types::BDT;
   fStr2type["RuleFit"]         = Types::RuleFit;
   fStr2type["SVM"]             = Types::SVM;
   fStr2type["MLP"]             = Types::MLP;
   fStr2type["BayesClassifier"] = Types::BayesClassifier;
   fStr2type["Committee"]       = Types::Committee;
}
