// @(#)root/guibuilder:$Name:  $:$Id: TGuiBldQuickHandler.h,v 1.2 2004/10/22 15:21:19 rdm Exp $
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
// TGuiBldQuickHandler                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGTextEntry;
class TGTextButton;


class TGuiBldTextDialog : public TGTransientFrame {

private:
   TGWindow       *fSelected;
   TGTextEntry    *fEntry;
   TGTextButton   *fOK;
   TGTextButton   *fCancel;
   TString         fSavedText;

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
   TGuiBldQuickHandler();
   virtual ~TGuiBldQuickHandler();

   virtual Bool_t HandleEvent(TGWindow *win);

   ClassDef(TGuiBldQuickHandler,0)  // frame property editor
};

#endif
