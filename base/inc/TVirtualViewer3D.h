// @(#)root/base:$Name:  $:$Id: TVirtualViewer3D.h
// Author: Olivier Couet 05/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TVirtualViewer3D
#define ROOT_TVirtualViewer3D

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualViewer3D                                                     //
//                                                                      //
// Abstract 3D shapes viewer. The concrete implementations are:         //
//                                                                      //
// TViewX3D   : X3d viewer                                              //
// TViewOpenGL: OpenGL viewer                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif

class  TVirtualViewer3D {
	
protected:
   TVirtualPad    *fPad;        // pad to be displayed in a 3D viewer

public:
   TVirtualViewer3D();
   TVirtualViewer3D(TVirtualPad *pad);
   virtual     ~TVirtualViewer3D();
   virtual void CreateScene(Option_t *option);
   virtual void UpdateScene(Option_t *option);

   static  TVirtualViewer3D *Viewer3D(Option_t *option);

   ClassDef(TVirtualViewer3D,0) // Abstract interface to 3D viewers
};

#endif
