// Author: Olivier Couet 22/12/2013

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TTextEditor                                                         //
//                                                                      //
//  Editor for changing TText's and TLatex's attributes.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTextEditor
#define ROOT_TTextEditor
#ifndef ROOT_TGedFrame
#include "TGedFrame.h"
#endif

class TText;
class TGTextEntry;
class TGNumberEntry;

class TTextEditor : public TGedFrame {

private:
   TText *fEditedText;

protected:
   TGTextEntry   *fText;  // Text
   TGNumberEntry *fAngle; // Text's angle
   TGNumberEntry *fSize;  // Text's angle
   TGNumberEntry *fXpos;  // Text's X position
   TGNumberEntry *fYpos;  // Text's Y position

   void ConnectSignals2Slots();

public:
   TTextEditor(const TGWindow *p = 0,
                  Int_t width = 140, Int_t height = 30,
                  UInt_t options = kChildFrame,
                  Pixel_t back = GetDefaultFrameBackground());
   ~TTextEditor();

   void SetModel(TObject *);

   void DoAngle();
   void DoSize();
   void DoText(const char*);
   void DoXpos();
   void DoYpos();

   ClassDef(TTextEditor,0)        // text editor
};

#endif // ROOT_TTextEditor

