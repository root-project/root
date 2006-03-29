// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldQuickHandler.h,v 1.3 2004/10/25 12:06:50 rdm Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGuiBldQuickHandler
#define ROOT_TGuiBldQuickHandler


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBldQuickHandler - quick handler for gui builder                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGTextEntry;
class TGTextButton;


class TGuiBldTextDialog : public TGTransientFrame {

private:
   TGWindow       *fSelected; // pointer to selected window
   TGTextEntry    *fEntry;    // entry for text edit
   TGTextButton   *fOK;       // OK button
   TGTextButton   *fCancel;   // cancel button
   TString         fSavedText; // saved text before editting

public:
   TGuiBldTextDialog(const char *win, const char *setter = "SetTitle(char*)", const char *getter = "GetTitle()");
   ~TGuiBldTextDialog();

   void  DoCancel();
   void  DoOK();
   void  CloseWindow();

   ClassDef(TGuiBldTextDialog,0) // text entry dialog
};

////////////////////////////////////////////////////////////////////////////////
class TGuiBldQuickHandler : public TObject {

private:
   TGWindow    *fSelected;    // editted frame

public:
   TGFrame     *fEditor;      // wizard/editor 

public:
   TGuiBldQuickHandler();
   virtual ~TGuiBldQuickHandler();

   virtual Bool_t HandleEvent(TGWindow *win);
   TGWindow *GetSelected() const { return fSelected; }

   ClassDef(TGuiBldQuickHandler,0)  // frame property editor
};


#endif
