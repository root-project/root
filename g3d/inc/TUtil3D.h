// @(#)root/g3d:$Name:  $:$Id: TUtil3D.h,v 1.6 2002/02/22 08:30:37 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TUtil3D
#define ROOT_TUtil3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TUtil3D                                                              //
//                                                                      //
// The default 3-D view utility  class                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TVirtualUtil3D
#include "TVirtualUtil3D.h"
#endif

class TUtil3D : public TVirtualUtil3D {


public:
   TUtil3D();
   virtual     ~TUtil3D();
   virtual void  DrawOutlineCube(TList *outline, Double_t *rmin, Double_t *rmax);
   virtual void  ToggleRulers(TVirtualPad *pad);
   virtual void  ToggleZoom(TVirtualPad *pad);

   ClassDef(TUtil3D,0)  //The default 3-D view utility  class
};

#endif
