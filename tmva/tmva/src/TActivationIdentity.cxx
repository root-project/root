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

/*! \class TMVA::TActivationIdentity
\ingroup TMVA
Identity activation function for TNeuron
*/

#include "TMVA/TActivationIdentity.h"

#include "Rtypes.h"
#include "TString.h"

#include <iostream>

class TString;

ClassImp(TMVA::TActivationIdentity);

////////////////////////////////////////////////////////////////////////////////
/// writes the identity function source code

void TMVA::TActivationIdentity::MakeFunction( std::ostream& fout, const TString& fncName )
{
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // identity" << std::endl;
   fout << "   return x;" << std::endl;
   fout << "}" << std::endl;
}
