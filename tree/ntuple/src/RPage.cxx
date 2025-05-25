/// \file RPage.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>

ROOT::Internal::RPage::~RPage()
{
   if (fPageAllocator)
      fPageAllocator->DeletePage(*this);
}

const void *ROOT::Internal::RPage::GetPageZeroBuffer()
{
   static const auto pageZero = std::make_unique<unsigned char[]>(kPageZeroSize);
   return pageZero.get();
}
