// @(#)root/guibuilder:$Name:  $:$Id: TGFrame.cxx,v 1.78 2004/09/13 09:10:08 rdm Exp $
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


class TGuiBldEditor : public TGMainFrame {

private:
   TGFrame              *fSelected;    // editted frame
   TGuiBldNameFrame     *fNameFrame;   // frame name
   TGuiBldHintsEditor   *fHintsFrame;  // frame hints
   TGuiBldBorderFrame   *fBorderFrame; // frame border

public:
   TGuiBldEditor();
   virtual ~TGuiBldEditor();

   TGFrame *GetSelected() const { return fSelected; }
   Bool_t   IsEmbedded() const;
   void     Hide();
   void     UpdateBorder(Int_t);
   void     UpdateBackground(Pixel_t col);
   void     UpdateForeground(Pixel_t col);

   void     UpdateSelected(TGFrame* = 0); //*SIGNAL*
   void     ChangeSelected(TGFrame*);     //*SIGNAL*

   ClassDef(TGuiBldEditor,0)  // frame property editor
};

#endif
