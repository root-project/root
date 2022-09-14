// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldEditor
#define ROOT_TGuiBldEditor


#include "TGFrame.h"

#include "TGNumberEntry.h"

class TGuiBldHintsEditor;
class TGuiBldNameFrame;
class TGuiBldBorderFrame;
class TGuiBldGeometryFrame;
class TGuiBldDragManager;
class TGTab;
class TGButton;
class TGLabel;
class TGGroupFrame;
class TGCompositeFrame;

//////////////////////////////////////////////////////////////////////////
class TGuiBldEditor : public TGVerticalFrame {

friend class TGuiBldDragManager;

private:
   TGFrame              *fSelected;       // edited frame
   TGuiBldNameFrame     *fNameFrame;      // frame name
   TGuiBldHintsEditor   *fHintsFrame;     // frame hints
   TGuiBldBorderFrame   *fBorderFrame;    // frame border
   TGuiBldGeometryFrame *fGeomFrame;      // frame geom
   TGGroupFrame         *fPositionFrame;  // X,Y coordinates
   TGuiBldDragManager   *fManager;        // main manager
   Bool_t                fEmbedded;       // kTRUE when it is inside guibuilder
   TGTab                *fTab;            // tab frame
   TGCompositeFrame     *fTablay;         // layout tab frame
   Int_t                 fLayoutId;       // the id of layout tab
   TGTextButton         *fLayoutButton;   // button to enable/disable layout
   TGLabel              *fLayoutLabel;    // saying if layout is enabled
   TGNumberEntry        *fXpos;           // X position
   TGNumberEntry        *fYpos;           // Y position

public:
   TGuiBldEditor(const TGWindow *p = nullptr);
   virtual ~TGuiBldEditor();

   Int_t    GetXPos() const { return fXpos->GetIntNumber(); }
   Int_t    GetYPos() const { return fYpos->GetIntNumber(); }
   void     SetXPos(Int_t pos) { fXpos->SetIntNumber(pos); }
   void     SetYPos(Int_t pos) { fYpos->SetIntNumber(pos); }

   TGFrame *GetSelected() const { return fSelected; }
   Bool_t   IsEmbedded() const { return fEmbedded; }
   void     SetEmbedded(Bool_t e = kTRUE) { fEmbedded = e; }
   void     Hide();
   void     UpdateBorder(Int_t);
   void     UpdateBackground(Pixel_t col);
   void     UpdateForeground(Pixel_t col);
   void     Reset();
   TGuiBldHintsEditor *GetHintsEditor() const { return fHintsFrame; }

   void     RemoveFrame(TGFrame *);
   void     TabSelected(Int_t id);
   void     UpdateSelected(TGFrame* = nullptr); //*SIGNAL*
   void     ChangeSelected(TGFrame*);     //*SIGNAL*
   void     SwitchLayout();

   ClassDef(TGuiBldEditor,0)  // frame property editor
};

#endif
