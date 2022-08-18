// @(#)root/mathcore:$Id$
// Author: L. Moneta Wed Aug 30 11:10:03 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Implementation file for class UnBinData

#include "Fit/UnBinData.h"
#include "Math/Error.h"

#include <cassert>
#include <cmath>

namespace ROOT {

   namespace Fit {

/// copy constructor
UnBinData::UnBinData(const UnBinData & rhs) :
   FitData(rhs),
   fWeighted(rhs.fWeighted)
{}

///assignment operator
UnBinData & UnBinData::operator= ( const UnBinData & rhs )
{
   FitData::operator=( rhs );
   fWeighted = rhs.fWeighted;
   return *this;
}


   } // end namespace Fit

} // end namespace ROOT

