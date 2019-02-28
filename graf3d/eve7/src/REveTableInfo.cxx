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

#include "json.hpp"

using namespace ROOT::Experimental;

void REveTableViewInfo::SetDisplayedCollection(ElementId_t collectionId)
{
    fDisplayedCollection = collectionId;

    for (auto &it : fDelegates)
       it(collectionId);
    StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveTableViewInfo::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   auto ret = REveElement::WriteCoreJson(j, rnr_offset);
   j["fDisplayedCollection"] = fDisplayedCollection;
   //j["_typename"]  = IsA()->GetName();
   return ret;
}
