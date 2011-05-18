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
class TRootGuiBuilder;

//////////////////////////////////////////////////////////////////////////
class TGuiBldHintsEditor : public TGVerticalFrame {

private:
   TGuiBldEditor        *fEditor;         // pointer to main editor
   TGuiBldNameFrame     *fNameFrame;      // frame name
   TGuiBldHintsManager  *fHintsManager;   // manager of subframes layout
   TGGroupFrame         *fHintsFrame;     // frame with layout hints
   TGGroupFrame         *fPaddingFrame;   // frame with padding

   void                 SetMatrixSep();

public:

   TGCheckButton *fCbLeft;       // button activating left hint
   TGCheckButton *fCbRight;      // button activating right hint
   TGCheckButton *fCbTop;        // button activating top hint
   TGCheckButton *fCbBottom;     // button activating bottom hint
   TGCheckButton *fCbExpandX;    // button activating expand X hint
   TGCheckButton *fCbExpandY;    // button activating expand Y hint
   TGCheckButton *fCbCenterX;    // button activating center X hint
   TGCheckButton *fCbCenterY;    // button activating center Y hint

   TGNumberEntry  *fPadTop;      // top side padding
   TGNumberEntry  *fPadBottom;   // bottom side padding
   TGNumberEntry  *fPadLeft;     // left side padding
   TGNumberEntry  *fPadRight;    // right side padding

   TGCheckButton  *fLayButton;   // enable/disable layout

   TRootGuiBuilder *fBuilder;

public:
   TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e);
   virtual ~TGuiBldHintsEditor() {}

   void     ChangeSelected(TGFrame *);
   void     LayoutSubframes(Bool_t on = kTRUE);
   void     MatrixLayout();
   void     SetPosition();
   void     UpdateState();

   ClassDef(TGuiBldHintsEditor,0) // layout hints editor
};

#endif
