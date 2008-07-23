// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

//
// Created by: Mark Fischler Tues July 19  2005
//

#include "Math/GenVector/GenVector_exception.h"

namespace ROOT {
namespace Math {
bool GenVector_exception::fgOn = false;

void Throw(GenVector_exception & e) { if (GenVector_exception::fgOn) throw e; }


void GenVector::Throw(const char * s) { 
   if (!GenVector_exception::fgOn) return;  
   GenVector_exception e(s);
   throw e; 
}



}  // namespace Math
}  // namespace ROOT
