// @(#)root/tmva $Id$
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TActivationIdentity                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Identity activation function for TNeuron                                  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>

#include "TMVA/TActivationIdentity.h"

ClassImp(TMVA::TActivationIdentity)

//______________________________________________________________________________
void TMVA::TActivationIdentity::MakeFunction( std::ostream& fout, const TString& fncName ) 
{
   // writes the identity function source code
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // identity" << std::endl;
   fout << "   return x;" << std::endl;
   fout << "}" << std::endl;
}
