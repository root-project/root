// @(#)root/base:$Name:  $:$Id: TVirtualUtil3D.h,v 1.1 2002/09/14 16:19:13 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualUtil3D
#define ROOT_TVirtualUtil3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualUtil3D                                                       //
//                                                                      //
// Abstract interface to the 3-D view utility                           //
//                                                                      //
// This class is called via the TPluginManager from classes that        //
// do not require linking with libG3d except in some rare cases like    //
// rotating a 3-d object in the pad or drawing 3-d axis in the pad.     //
// The concrete implementation TUtil3D is defined in system.rootrc      //
// and can be overridden by a user to extend the functionality.         //
// This abstract interface has three main goals:                        //
//   - it decouples libG3d from the calling classes                     //
//   - it does not require the use of magic strings like when using     //
//     gROOT->ProcessLine to instantiate classes in libG3d.             //
//   - it allows advanced users to redefine or extend some functions.   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class TVirtualPad;
class TList;

class TVirtualUtil3D : public TNamed {


public:
   TVirtualUtil3D();
   virtual     ~TVirtualUtil3D();
   virtual void  DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax) = 0;
   virtual void  ToggleRulers(TVirtualPad *pad) = 0;
   virtual void  ToggleZoom(TVirtualPad *pad) = 0;

   ClassDef(TVirtualUtil3D,0)  //Abstract interface to a the 3-D view utility
};

#endif
