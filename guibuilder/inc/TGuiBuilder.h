// @(#)root/guibuilder:$Name:  $:$Id: TGFrame.cxx,v 1.78 2004/09/13 09:10:08 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBuilder
#define ROOT_TGuiBuilder


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBuilder                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TVirtualGuiBld
#include "TVirtualGuiBld.h"
#endif


class TGShutter;

class TGuiBuilder : public TVirtualGuiBld, public TGMainFrame {

private:
   TGShutter *fShutter;

public:
   TGuiBuilder(const TGWindow *p = 0);
   virtual ~TGuiBuilder();

   virtual void      AddAction(TGuiBldAction *act, const char *sect);
   virtual void      AddSection(const char *sect);
   virtual TGFrame  *ExecuteAction();
   virtual void      HandleButtons();
   virtual void      Show() { MapRaised(); }
   virtual void      Hide() { UnmapWindow(); }

   ClassDef(TGuiBuilder,0)  // gui builder
};


#endif
