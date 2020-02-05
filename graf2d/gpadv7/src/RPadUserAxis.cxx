/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPadUserAxis.hxx>

// pin vtable
ROOT::Experimental::RPadUserAxisBase::~RPadUserAxisBase()
{
}

ROOT::Experimental::RPadLength::Normal
ROOT::Experimental::RPadCartesianUserAxis::ToNormal(const RPadLength::User &usercoord) const
{
   return (usercoord.fVal - GetBegin()) / GetSensibleDenominator();
}
