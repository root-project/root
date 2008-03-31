// @(#)root/tree:$Id$
// Author: Fons Rademakers   7/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelectorList
#define ROOT_TSelectorList


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorList                                                        //
//                                                                      //
// A TList derived class that makes sure that objects added to it       //
// are not linked to the currently open file (like histograms,          //
// eventlists and trees). Also it makes sure the name of the added      //
// object is unique. This class is used in the TSelector for the        //
// output list.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif


class TSelectorList : public TList {

private:
   Bool_t UnsetDirectory(TObject *obj);
   Bool_t CheckDuplicateName(TObject *obj);

public:
   TSelectorList() : TList() { SetOwner(); }

   void AddFirst(TObject *obj);
   void AddFirst(TObject *obj, Option_t *opt);
   void AddLast(TObject *obj);
   void AddLast(TObject *obj, Option_t *opt);
   void AddAt(TObject *obj, Int_t idx);
   void AddAfter(const TObject *after, TObject *obj);
   void AddAfter(TObjLink *after, TObject *obj);
   void AddBefore(const TObject *before, TObject *obj);
   void AddBefore(TObjLink *before, TObject *obj);

   ClassDef(TSelectorList,1)  //Special TList used in the TSelector
};

#endif
