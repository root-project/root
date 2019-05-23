/// \file RDrawingOptsBase.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDrawingOptsBase.hxx"

#include "ROOT/RDrawingAttr.hxx"
#include "ROOT/TLogger.hxx"

#include "TClass.h"
#include "TDataMember.h"
#include "TMemberInspector.h"

#include <algorithm>
#include <sstream>

ROOT::Experimental::RDrawingOptsBase::RDrawingOptsBase(const RDrawingOptsBase & other):
   fHolder(other.fHolder ? std::make_unique<RDrawingAttrHolder>(*other.fHolder) : nullptr)
{
}

ROOT::Experimental::RDrawingOptsBase &
ROOT::Experimental::RDrawingOptsBase::operator=(const RDrawingOptsBase & other)
{
   if (other.fHolder)
      fHolder = std::make_unique<RDrawingAttrHolder>(*other.fHolder);
   else
      fHolder.reset();

   return *this;
}

ROOT::Experimental::RDrawingAttrHolder &
ROOT::Experimental::RDrawingOptsBase::GetHolder()
{
   if (!fHolder)
      fHolder = std::make_unique<RDrawingAttrHolder>();
   return *fHolder;
}

void ROOT::Experimental::RDrawingOptsBase::SetStyleClasses(const std::vector<std::string> &styles)
{
   GetHolder().SetStyleClasses(styles);
}

const std::vector<std::string> &ROOT::Experimental::RDrawingOptsBase::GetStyleClasses() const
{
   static const std::vector<std::string> sEmpty;
   if (!fHolder)
      return sEmpty;
   return fHolder->GetStyleClasses();
}

std::vector<std::pair<std::string, std::string>> ROOT::Experimental::RDrawingOptsBase::GetModifiedAttributeStrings()
{
   if (fHolder)
      return GetHolder().CustomizedValuesToString(GetName());
   return {};
}