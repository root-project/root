// @(#)root/base:$Name:  $:$Id: TVirtualUtilPad.h,v 1.1 2002/09/15 19:41:51 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualUtilPad
#define ROOT_TVirtualUtilPad


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualUtilPad                                                      //
//                                                                      //
// Abstract interface to the pad/canvas  utilities                      //
//                                                                      //
// This class is called via the TPluginManager from classes that        //
// do not require linking with libGpad except in some rare cases like   //
// the invokation of the DrawPanel or FitPanel.                         //
// The concrete implementation TUtilPad is defined in system.rootrc     //
// and can be overridden by a user to extend the functionality.         //
// This abstract interface has three main goals:                        //
//   - it decouples libGpad from the calling classes                    //
//   - it does not require the use of magic strings like when using     //
//     gROOT->ProcessLine to instantiate classes in libGpad.            //
//   - it allows advanced users to redefine or extend some functions.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif 

class TVirtualPad;

class TVirtualUtilPad : public TNamed {


public:
   TVirtualUtilPad();
   virtual     ~TVirtualUtilPad();
   virtual void  DrawPanel(const TVirtualPad *pad, const TObject *obj) = 0;
   virtual void  FitPanel(const TVirtualPad *pad, const TObject *obj) = 0;
   virtual void  FitPanelGraph(const TVirtualPad *pad, const TObject *obj) = 0;
   virtual void  InspectCanvas(const TObject *obj) = 0;
   virtual void  MakeCanvas(const char *name, const char *title, Int_t wtopx, Int_t wtopy, Int_t ww, Int_t wh) = 0;
   virtual void  RemoveObject(TObject *parent, const TObject *obj) = 0;

   ClassDef(TVirtualUtilPad,0)  //Abstract interface to the pad/canvas  utilities
};

#endif
