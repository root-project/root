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
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualViewer3D.h"
#include "TVirtualPad.h"
//#include "TROOT.h"
#include "TPluginManager.h"
#include "TError.h"


ClassImp(TVirtualViewer3D)

//______________________________________________________________________________
TVirtualViewer3D* TVirtualViewer3D::Viewer3D(TVirtualPad *pad, Option_t *type)
{
   // Create a Viewer 3D of specified type
   TVirtualViewer3D *viewer = 0;
   TPluginHandler *h;
   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualViewer3D", type))) {
      if (h->LoadPlugin() == -1) return 0;
      
      if (!pad) {
         viewer = (TVirtualViewer3D *) h->ExecPlugin(1, gPad); 
      } else {
         viewer = (TVirtualViewer3D *) h->ExecPlugin(1, gPad); 
      }
   }
   return viewer;
}
