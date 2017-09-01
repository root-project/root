// @(#)root/gl:$Id$
// Author: Bertrand Bellenot 23/01/2008

/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLEmbeddedViewer
#define ROOT_TGLEmbeddedViewer

#include "TGFrame.h"

#include "TGLViewer.h"

class TGLRenderArea;
class TGLEventHandler;
class TGedEditor;

class TGLEmbeddedViewer : public TGLViewer
{
private:
   // GUI components
   TGCompositeFrame  *fFrame;
   Int_t              fBorder;

   void Init(const TGWindow *parent);
   void CreateFrames();

   TGLEmbeddedViewer(const TGLEmbeddedViewer&); // Not implemented
   TGLEmbeddedViewer& operator=(const TGLEmbeddedViewer&); // Not implemented

public:
   TGLEmbeddedViewer(const TGWindow *parent, TVirtualPad *pad=0, Int_t border=2);
   TGLEmbeddedViewer(const TGWindow *parent, TVirtualPad *pad, TGedEditor *ged, Int_t border=2);
   ~TGLEmbeddedViewer();

   virtual void CreateGLWidget();
   virtual void DestroyGLWidget();

   virtual const char *GetName() const { return "GLViewer"; }

   TGCompositeFrame*   GetFrame() const { return fFrame; }

   TGLOrthoCamera     *GetOrthoXOYCamera() { return &fOrthoXOYCamera; }
   TGLOrthoCamera     *GetOrthoXOZCamera() { return &fOrthoXOZCamera; }
   TGLOrthoCamera     *GetOrthoZOYCamera() { return &fOrthoZOYCamera; }
   TGLOrthoCamera     *GetOrthoZOXCamera() { return &fOrthoZOXCamera; }

   ClassDef(TGLEmbeddedViewer, 0); // Embedded GL viewer.
};

#endif
