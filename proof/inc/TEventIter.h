// @(#)root/proof:$Name:  $:$Id: TEventIter.h,v 1.1 2002/01/15 00:45:20 rdm Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEventIter
#define ROOT_TEventIter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TEventIter                                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TDSet;
class TSelector;
class TFile;
class TIter;


class TEventIter : public TObject {

private:
   Bool_t   fIsTree;
   TFile   *fFile;
   TList   *fKeys;
   TString  fClassName;
   TIter   *fNext;
   TObject *fObj;

public:
   TEventIter();
   TEventIter(TDSet *set);

   Bool_t GetNextEvent(TSelector *selector);

   ClassDef(TEventIter,1)  // Event iterator used by TProofPlayer's
};

#endif


