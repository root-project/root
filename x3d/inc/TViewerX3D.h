// @(#)root/x3d:$Name$:$Id$
// Author: Rene Brun   05/09/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TViewerX3D
#define ROOT_TViewerX3D


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TViewerX3D                                                           //
//                                                                      //
// C++ interface to the X3D viewer                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif


class TViewerX3D : public TObject {


public:
   TViewerX3D();
   virtual ~TViewerX3D() { }
   static void View(TVirtualPad *pad, Option_t *option);

   ClassDef(TViewerX3D,1)  //C++ interface to the X3D viewer
};

#endif
