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
// TViewerX3D   : X3d viewer                                            //
// TViewerOpenGL: OpenGL viewer                                         //
// TViewerPad3D : visualise the 3D scene in the current Pad             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualPad
#include "TVirtualPad.h"
#endif

class TVirtualViewer3D {

protected:
   TVirtualPad    *fPad;        // pad to be displayed in a 3D viewer

public:
   TVirtualViewer3D() : fPad(0) { }
   TVirtualViewer3D(TVirtualPad *pad);
   virtual     ~TVirtualViewer3D() { }
   virtual void CreateScene(Option_t *option);
   virtual void UpdateScene(Option_t *option);

   static  TVirtualViewer3D *Viewer3D(Option_t *option);

   ClassDef(TVirtualViewer3D,0) // Abstract interface to 3D viewers
};

#endif
