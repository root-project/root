// @(#)root/mathcore:$Id$
// Authors: C. Gumpert    09/2011

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2011 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for TDataPointN class 
// 

#include "Math/TDataPointN.h"
namespace ROOT
{
   namespace Math
   {

      template<> UInt_t TDataPointN<Float_t>::kDimension = 0;
      template<> UInt_t TDataPointN<Double_t>::kDimension = 0;

   }//namespace Math
}//namespace ROOT
