// @(#)root/guibuilder:$Id$
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldHintsEditor
#define ROOT_TGuiBldHintsEditor


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldHintsEditor - layout hints editor                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGButton
#include "TGButton.h"
#endif


class TGuiBldHintsButton;
class TGNumberEntry;
class TGuiBldEditor;
class TGuiBldNameFrame;
class TGuiBldHintsManager;

class TGuiBldHintsEditor : public TGVerticalFrame {

private:
   TGuiBldEditor     *fEditor;        // pointer to main editor
   TGuiBldNameFrame  *fNameFrame;      // frame name
   TGuiBldHintsManager *fHintsManager; // manager of subframes layout   

   void SetMatrixSep();

public:
   TGuiBldHintsButton *fExpandX;  // expand in x direction button
   TGuiBldHintsButton *fExpandY;  // expand in y direction button 
   TGuiBldHintsButton *fCenterX;  // center in x direction button
   TGuiBldHintsButton *fCenterY;  // center in y direction button

   TGTextButton *fHintsLeft;     // button activating left hints
   TGTextButton *fHintsRight;    // button activating right hints
   TGTextButton *fHintsTop;      // button activating top hints
   TGTextButton *fHintsBottom;   // button activating bottom hints

   TGNumberEntry  *fPadTop;      // top side padding
   TGNumberEntry  *fPadBottom;   // bottom side padding 
   TGNumberEntry  *fPadLeft;     // left side padding
   TGNumberEntry  *fPadRight;    // right side padding

public:
   TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e);
   virtual ~TGuiBldHintsEditor() {}

   void     ChangeSelected(TGFrame *);
   void     UpdateState();
   void     LayoutSubframes(Bool_t on);
   void     MatrixLayout();

   ClassDef(TGuiBldHintsEditor,0) // layout hints editor
};

#endif
