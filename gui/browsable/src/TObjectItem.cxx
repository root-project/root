/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectItem.hxx>
#include <ROOT/Browsable/RProvider.hxx>

#include "TObject.h"


using namespace ROOT::Experimental::Browsable;

TObjectItem::TObjectItem(const TObject *obj) : TObjectItem(obj->GetName(), obj->IsFolder() ? -1 : 0)
{
   SetClassName(obj->ClassName());

   SetTitle(obj->GetTitle());

   SetIcon(RProvider::GetClassIcon(obj->IsA(), obj->IsFolder()));
}
