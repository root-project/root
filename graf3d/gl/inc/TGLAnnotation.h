// @(#)root/gl:$Id$
// Author:  Matevz and Alja Tadel  20/02/2009

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLAnnotation
#define ROOT_TGLAnnotation

#include "TGLOverlay.h"
#include "TGLUtil.h"
#include "TGLFontManager.h"

class TGLViewer;
class TGLViewerBase;
class TGLFont;
class TGTextEdit;
class TGMainFrame;

class TGLAnnotation : public TGLOverlayElement
{
private:
   TGLAnnotation(const TGLAnnotation&);            // Not implemented
   TGLAnnotation& operator=(const TGLAnnotation&); // Not implemented

   void MakeEditor();

protected:
   TGMainFrame      *fMainFrame;
   TGTextEdit       *fTextEdit;

   TGLViewer        *fParent;

   TString           fText;           // annotation text
   Float_t           fLabelFontSize;  // relative font size
   TGLFont           fLabelFont;      // font used to render labels
   TGLFont           fMenuFont;       // font used to render menu buttons

   Pixel_t           fBackColor;      // background color
   Pixel_t           fBackHighColor;  // background active color
   Pixel_t           fTextColor;      // text color
   Pixel_t           fTextHighColor;  // text active color
   Float_t           fAlpha;          // label transparency

   Float_t           fPosX;           // x position [0, 1]
   Float_t           fPosY;           // y position [0, 1]

   Int_t             fMouseX, fMouseY; //! last mouse position
   Bool_t            fInDrag;          //!

   TGLVector3        fPointer;         // picked location in 3D space
   Bool_t            fActive;          // active item identifier

public:
   TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy, TGLVector3 ref);
   virtual ~TGLAnnotation();

   virtual Bool_t MouseEnter(TGLOvlSelectRecord& selRec);
   virtual Bool_t Handle(TGLRnrCtx& rnrCtx, TGLOvlSelectRecord& selRec,
                         Event_t* event);
   virtual void   MouseLeave();

   void CloseEditor();

   void UpdateText();

   virtual void   Render(TGLRnrCtx& rnrCtx);

   ClassDef(TGLAnnotation, 0); // GL-annotation.
};

#endif
