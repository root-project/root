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


   Float_t           fPosX;           // x position [0, 1]
   Float_t           fPosY;           // y position [0, 1]

   Int_t             fMouseX, fMouseY; //! last mouse position
   Bool_t            fInDrag;          //!

   TGLVector3        fPointer;         // picked location in 3D space
   Bool_t            fActive;          // active item identifier

   TGMainFrame      *fMainFrame;       // editors
   TGTextEdit       *fTextEdit;        // editors

   static Color_t    fgBackColor;
   static Color_t    fgTextColor;

protected:
   TGLViewer        *fParent;

   TString           fText;           // annotation text
   Float_t           fTextSize;       // relative font size
   TGLFont           fFont;           // font used to render labels
   TGLFont           fMenuFont;       // font used to render menu buttons

   Color_t           fBackColor;      // background color
   Color_t           fTextColor;      // text color
   Char_t            fTransparency;   // transparency of background

   Bool_t            fDrawRefLine;    // draw 3D refrence line
   Bool_t            fUseColorSet;    // use color set from rnrCtx

public:
   TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy);
   TGLAnnotation(TGLViewerBase *parent, const char *text, Float_t posx, Float_t posy, TGLVector3 ref);
   virtual ~TGLAnnotation();


   void SetTransparency(Char_t x) { fTransparency = x;}
   Char_t GetTransparency() const { return fTransparency;}
   void SetUseColorSet(Bool_t x)  { fUseColorSet = x; }
   Bool_t GetUseColorSet() const  { return fUseColorSet;}
   void SetBackColor(Color_t x)   { fBackColor = x;}
   Color_t GetBackColor() const   { return fBackColor;}
   void SetTextColor(Color_t x)   { fTextColor = x;   }
   Color_t GetTextColor() const   { return fTextColor;}
   void SetTextSize(Float_t x)    { fTextSize = x;   }
   Float_t getTextSize() const    { return fTextSize;}

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
