// @(#)root/base:$Name:  $:$Id: TVirtualUtil3D.h,v 1.6 2002/02/22 08:30:37 brun Exp $
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
