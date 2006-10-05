// @(#)root/gpad:$Name:  $:$Id: TUtilPad.h,v 1.5 2004/07/20 20:55:42 rdm Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TUtilPad
#define ROOT_TUtilPad


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUtilPad                                                             //
//                                                                      //
// misc. pad/canvas  utilities                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualUtilPad
#include "TVirtualUtilPad.h"
#endif


class TUtilPad : public TVirtualUtilPad {

private:
   static Int_t   fgPanelVersion;   //Draw/FitPanel version (0=new, 1=old)

public:
   TUtilPad();
   virtual     ~TUtilPad();
   virtual void  DrawPanel(const TVirtualPad *pad, const TObject *obj);
   virtual void  FitPanel(const TVirtualPad *pad, const TObject *obj);
   virtual void  FitPanelGraph(const TVirtualPad *pad, const TObject *obj);
   virtual void  InspectCanvas(const TObject *obj);
   virtual void  MakeCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh);
   virtual void  RemoveObject(TObject *parent, const TObject *obj);
   static  void  SetPanelVersion(Int_t version=0);

   ClassDef(TUtilPad,0)  //misc. pad/canvas  utilities
};

#endif
