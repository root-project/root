// @(#)root/proofplayer:$Id$
// Author: Axel Naumann   2010-06-09

/*************************************************************************
 * Copyright (C) 1995-2010, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TOutputListSelectorDataMap
#define ROOT_TOutputListSelectorDataMap

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOutputListSelectorDataMap                                           //
//                                                                      //
// Set the selector's data members to the corresponding elements of the //
// output list.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TSelector;
class TCollection;

class TOutputListSelectorDataMap: public TObject {
public:

   TOutputListSelectorDataMap(TSelector* sel = 0);
   virtual ~TOutputListSelectorDataMap() {}

   static TOutputListSelectorDataMap* FindInList(TCollection* coll);

   const char* GetName() const;

   Bool_t Init(TSelector* sel);
   Bool_t SetDataMembers(TSelector* sel) const;
   Bool_t Merge(TObject* obj);

   TCollection* GetMap() const { return fMap; }

private:
   TCollection* fMap;
   ClassDef(TOutputListSelectorDataMap, 1)  // Converter from output list to TSelector data members
};


#endif
