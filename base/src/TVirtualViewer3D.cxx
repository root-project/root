// @(#)root/base:$Name:  $:$Id: TVirtualViewer3D.cxx
// Author: Olivier Couet 05/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TVirtualViewer3D.h"
#include "TPluginManager.h"
#include "TError.h"


ClassImp(TVirtualViewer3D)

//______________________________________________________________________________
TVirtualViewer3D::TVirtualViewer3D(TVirtualPad *pad)
{
   // TVirtualViewer3D constructor.

   fPad = pad;
}

//______________________________________________________________________________
void TVirtualViewer3D::CreateScene(Option_t *option)
{
   ::Info("TVirtualViewer3D::CreateScene", "create scene with option %s", option);
}

//______________________________________________________________________________
void TVirtualViewer3D::UpdateScene(Option_t *option)
{
   ::Info("TVirtualViewer3D::UpdateScene", "update scene with option %s", option);
}

//______________________________________________________________________________
TVirtualViewer3D* TVirtualViewer3D::Viewer3D(Option_t *option)
{
   // Create a Viewer 3D acording to "option".

   TVirtualViewer3D *viewer = 0;
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualViewer3D", option))) {
      if (h->LoadPlugin() == -1) return 0;
      viewer = (TVirtualViewer3D *) h->ExecPlugin(1, gPad);
   }
   return viewer;
}
