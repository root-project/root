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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldEditor                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGuiBldHintsEditor;
class TGuiBldNameFrame;
class TGuiBldBorderFrame;
class TGTab;

class TGuiBldEditor : public TGCompositeFrame {

private:
   TGFrame              *fSelected;    // editted frame
   TGuiBldNameFrame     *fNameFrame;   // frame name
   TGuiBldHintsEditor   *fHintsFrame;  // frame hints
   TGuiBldBorderFrame   *fBorderFrame; // frame border
   Bool_t                fEmbedded;    // kTRUE when it is inside guibuilder
   TGTab                *fTab;         // tab frame
   Int_t                 fLayoutId;    // the id of layout tab

public:
   TGuiBldEditor(const TGWindow *p = 0);
   virtual ~TGuiBldEditor();

   TGFrame *GetSelected() const { return fSelected; }
   Bool_t   IsEmbedded() const { return fEmbedded; }
   void     SetEmbedded(Bool_t e = kTRUE) { fEmbedded = e; } 
   void     Hide();
   void     UpdateBorder(Int_t);
   void     UpdateBackground(Pixel_t col);
   void     UpdateForeground(Pixel_t col);
   void     Reset();
   TGuiBldHintsEditor *GetHintsEditor() const { return fHintsFrame; }

   void     TabSelected(Int_t id);
   void     UpdateSelected(TGFrame* = 0); //*SIGNAL*
   void     ChangeSelected(TGFrame*);     //*SIGNAL*

   ClassDef(TGuiBldEditor,0)  // frame property editor
};

#endif
