// @(#)root/mathcore:$Name:  $:$Id: GenVector_exception.cxxv 1.0 2005/06/23 12:00:00 moneta Exp $
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
bool GenVector_exception::on = false;

void Throw(GenVector_exception & e) { if (GenVector_exception::on) throw e; }


}  // namespace Math
}  // namespace ROOT
