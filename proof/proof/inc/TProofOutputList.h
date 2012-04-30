// @(#)root/proofplayer:$Id$
// Author: G. Ganis   04/08/2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofOutputList
#define ROOT_TProofOutputList

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofList                                                           //
//                                                                      //
// Derivation of TList with an overload of ls() and Print() allowing    //
// to filter out some of the variables.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TList
#include "TList.h"
#endif

class TProofOutputList : public TList {

private:
   TList *fDontShow; // list of reg expression defining what should not be shown
  
public:
   TProofOutputList(const char *dontshow = "PROOF_*");
TProofOutputList(TObject *o) : TList(o), fDontShow(0) { } // for backward compatibility, don't use
   virtual ~TProofOutputList();

   void AttachList(TList *alist);
   
   virtual void ls(Option_t *option="") const ;
   virtual void Print(Option_t *option="") const;
   virtual void Print(Option_t *option, Int_t recurse) const
                                { TCollection::Print(option, recurse); }
   virtual void Print(Option_t *option, const char* wildcard, Int_t recurse=1) const
                                { TCollection::Print(option, wildcard, recurse); }
   virtual void Print(Option_t *option, TPRegexp& regexp, Int_t recurse=1) const
                                { TCollection::Print(option, regexp, recurse);}

   TList *GetDontShowList() { return fDontShow; }
   
   ClassDef(TProofOutputList, 1);  // Output list specific TList derivation
};

#endif
