// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldHintsEditor.h,v 1.1 2004/09/13 12:47:35 rdm Exp $
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
// TGuiBldHintsEditor                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGButton
#include "TGButton.h"
#endif


class TGuiBldHintsButton;
class TGNumberEntry;
class TGuiBldEditor;


class TGuiBldHintsEditor : public TGVerticalFrame {

private:
   TGuiBldEditor     *fEditor;

   TGuiBldHintsButton *fExpandX;  //
   TGuiBldHintsButton *fExpandY;  //
   TGuiBldHintsButton *fCenterX;  //
   TGuiBldHintsButton *fCenterY;  //

   TGTextButton *fHintsLeft;  //
   TGTextButton *fHintsRight;  //
   TGTextButton *fHintsTop;  //
   TGTextButton *fHintsBottom;  //

   TGNumberEntry  *fPadTop;
   TGNumberEntry  *fPadBottom;
   TGNumberEntry  *fPadLeft;
   TGNumberEntry  *fPadRight;

public:
   TGuiBldHintsEditor(const TGWindow *p, TGuiBldEditor *e);
   virtual ~TGuiBldHintsEditor() {}

   void     ChangeSelected(TGFrame *);
   void     UpdateState();

   ClassDef(TGuiBldHintsEditor,0) // layout hints editor
};

#endif
