// $Id$
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
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


#ifndef ROOT_TString
#include "TString.h"
#endif


class TGuiBldAction;
class TVirtualGuiBld {

public:
   TVirtualGuiBld();
   virtual ~TVirtualGuiBld();

   virtual void AddAction(TGuiBldAction *) {  }
   virtual void Show() {}
   virtual void Hide() {}

   ClassDef(TVirtualGuiBld,0)  // ABC for gui builder
};

R__EXTERN TVirtualGuiBld *gGuiBuilder; // global gui builder

#endif
