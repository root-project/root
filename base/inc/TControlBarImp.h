// @(#)root/base:$Name:  $:$Id: TControlBarImp.h,v 1.5 2003/10/07 13:41:33 brun Exp $
// Author: Nenad Buncic   20/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TControlBarImp
#define ROOT_TControlBarImp


////////////////////////////////////////////////////////////////////////////////
//                                                                            //
// TControlBarImp                                                             //
//                                                                            //
// ABC describing GUI independent control bar (see TControlBar)               //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif


class TControlBar;
class TControlBarButton;

class TControlBarImp {

protected:
   TControlBar *fControlBar; //TControlBar associated with this implementation
   Int_t        fXpos;       //Initial x position
   Int_t        fYpos;       //Initial y position
   TControlBarButton *fClicked; //Last clicked button 
   
public:
   TControlBarImp(TControlBar *c, const char * = "") : fControlBar(c) { }
   TControlBarImp(TControlBar *c, const char *, Int_t, Int_t) : fControlBar(c) { }
   virtual ~TControlBarImp() { }

   virtual TControlBar *GetControlBar() { return fControlBar; }
   virtual TControlBarButton *GetClicked() { return fClicked; } 

   virtual void Create() { }
   virtual void Hide() { }
   virtual void Show() { }
   virtual void SetFont(const char * /*fontName*/) { }
   virtual void SetTextColor(const char * /*colorName*/) { }
   virtual void SetButtonState(const char * /*label*/, Int_t /*state*/) { }

   ClassDef(TControlBarImp,0)  //GUI independent controlbar abc
};

#endif
