// @(#)root/gl:$Id$
// Author:  Thomas Keck  13/02/2015

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLOculusViewer
#define ROOT_TGLOculusViewer

#include "TGLEmbeddedViewer.h"
#include "TGLSAViewer.h"
#include <iostream>

class TGLEmbeddedOculusViewer : public TGLEmbeddedViewer

{

private:
   TGLEmbeddedOculusViewer(const TGLEmbeddedOculusViewer &);             // Not implemented
   TGLEmbeddedOculusViewer & operator=(const TGLEmbeddedOculusViewer &); // Not implemented

public:
   TGLEmbeddedOculusViewer(const TGWindow* w, TVirtualPad* p, TGedEditor* e, int f) : TGLEmbeddedViewer(w, p, e, f) { fStereo = true; }
   virtual ~TGLEmbeddedOculusViewer() { }

   // The magic happens here:
   // As soon as this method is called we initialize the oculus rift
   // and draw the stereoscopic view.
   virtual void DoDrawStereo(Bool_t swap_buffers);

   ClassDef(TGLEmbeddedOculusViewer,0)
};

class TGLSAOculusViewer : public TGLSAViewer

{

private:
   TGLSAOculusViewer(const TGLSAOculusViewer &);             // Not implemented
   TGLSAOculusViewer & operator=(const TGLSAOculusViewer &); // Not implemented

public:
   TGLSAOculusViewer(const TGWindow* w, TVirtualPad* p, TGedEditor* e, TGLFormat* f) : TGLSAViewer(w, p, e, f) { fStereo = true; }
   virtual ~TGLSAOculusViewer() { }

   // The magic happens here:
   // As soon as this method is called we initialize the oculus rift
   // and draw the stereoscopic view.
   virtual void DoDrawStereo(Bool_t swap_buffers);

   ClassDef(TGLSAOculusViewer,0)
};

#endif // ROOT_TGLOculusViewer
