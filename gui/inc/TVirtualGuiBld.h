// $Id: TVirtualGuiBld.h,v 1.2 2004/09/08 17:34:19 rdm Exp $
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualGuiBld
#define ROOT_TVirtualGuiBld


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGuiBld                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

enum EGuiBldAction { kGuiBldNone, kGuiBldCtor,  kGuiBldProj, 
                     kGuiBldMacro, kGuiBldFunc };

class TGFrame;
//////////////////////////////////////////////////////////////////////////
class TGuiBldAction : public TNamed {

public:
   Int_t       fType;   // type of action
   TString     fAct;    // action
   const char *fPic;    // picture name

   TGuiBldAction(const char *name = 0, const char *title = 0, Int_t type = kGuiBldCtor);
   virtual ~TGuiBldAction();

   ClassDef(TGuiBldAction,0)  // gui builder action
};


//////////////////////////////////////////////////////////////////////////
class TVirtualGuiBld {

protected:
   TGuiBldAction *fAction;   // current action

public:
   TVirtualGuiBld();
   virtual ~TVirtualGuiBld();

   virtual void      AddAction(TGuiBldAction *, const char * /*section*/) = 0;
   virtual void      AddSection(const char * /*section*/) = 0;
   virtual TGFrame  *ExecuteAction() { return 0; }
   virtual void      SetAction(TGuiBldAction *act) { fAction = act; }
   TGuiBldAction    *GetAction() const { return fAction; }
   virtual Bool_t    IsExecutalble() const  { return fAction && !fAction->fAct.IsNull(); }

   ClassDef(TVirtualGuiBld,0)  // ABC for gui builder
};

R__EXTERN TVirtualGuiBld *gGuiBuilder; // global gui builder

#endif
