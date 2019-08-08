// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveTableInfo.hxx>
#include <ROOT/REveManager.hxx>

#include "json.hpp"

using namespace ROOT::Experimental;

REveTableViewInfo::REveTableViewInfo(const std::string &name, const std::string &title) : REveElement(name, title), fConfigChanged(false)
{
}

void REveTableViewInfo::SetDisplayedCollection(ElementId_t collectionId)
{
   fDisplayedCollection = collectionId;

   fConfigChanged = true;
   for (auto &it : fDelegates)
      it();

   fConfigChanged = false;
   StampObjProps();
}

void REveTableViewInfo::AddNewColumnToCurrentCollection(const std::string& expr, const std::string& title, int prec)
{
   if (!fDisplayedCollection) return;

      REveElement* col = gEve->FindElementById(fDisplayedCollection);
      if (!col) {
         printf("REveTableViewInfo::AddNewColumnToCurrentCollection error: collection not found\n");
         return;
      }

      fConfigChanged = true;
      table(col->GetName()).column(title, prec, expr);

       for (auto &it : fDelegates)
          it();

      fConfigChanged = false;

    StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveTableViewInfo::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   auto ret = REveElement::WriteCoreJson(j, rnr_offset);
   j["fDisplayedCollection"] = fDisplayedCollection;
   return ret;
}


