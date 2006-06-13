// @(#)root/proof:$Name:  $:$Id: TProofProgressLog.h,v 1.3 2006/05/15 09:45:03 brun Exp $
// Author: G Ganis, Aug 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressLog
#define ROOT_TProofProgressLog

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGTextView;
class TGTextButton;
class TProofProgressDialog;


class TProofProgressLog : public TGTransientFrame {

private:
   TGTextView           *fText;   // text widget
   TGTextButton         *fClose;  // close button
   TProofProgressDialog *fDialog; // owner dialog

public:
   TProofProgressLog(TProofProgressDialog *d, Int_t w = 700, Int_t h = 300);
   virtual ~TProofProgressLog();

   void   LoadBuffer(const char *buffer);
   void   AddBuffer(const char *buffer);

   void   LoadFile(const char *file);

   void   Clear(Option_t * = 0);
   void   Popup();

   // slots
   void   CloseWindow();

   ClassDef(TProofProgressLog,0) //Class implementing a log graphic box
};

#endif
