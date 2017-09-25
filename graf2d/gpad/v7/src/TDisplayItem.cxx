/// \file TDisplayItem.cxx
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TDisplayItem.hxx"

#include "TString.h"

ROOT::Experimental::TDisplayItem::TDisplayItem() : fObjectID(), fOption(), fKind(0)
{
}

ROOT::Experimental::TDisplayItem::TDisplayItem(const TDisplayItem &rhs)
   : fObjectID(rhs.fObjectID), fOption(rhs.fOption), fKind(rhs.fKind)
{
}

// pin vtable
ROOT::Experimental::TDisplayItem::~TDisplayItem()
{
}

std::string ROOT::Experimental::TDisplayItem::MakeIDFromPtr(void *ptr)
{
   UInt_t hash = TString::Hash(&ptr, sizeof(ptr));
   TString res = TString::UItoa(hash, 10);
   return std::string(res.Data());
}

void ROOT::Experimental::TDisplayItem::SetObjectIDAsPtr(void *ptr)
{
   std::string id = MakeIDFromPtr(ptr);
   SetObjectID(id.c_str());
}

// =================

ROOT::Experimental::TPadDisplayItem::~TPadDisplayItem()
{
   Clear();
}

void ROOT::Experimental::TPadDisplayItem::Clear()
{
   for (unsigned n = 0; n < fPrimitives.size(); ++n) delete fPrimitives[n];
   fPrimitives.clear();
}
