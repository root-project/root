// @(#)root/proof:$Name:  $:$Id: TProofProgressLog.h,v 1.2 2005/08/31 11:11:06 rdm Exp $
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

class TGTransientFrame;
class TGTextView;
class TGTextButton;
class TProofProgressDialog;

class TProofProgressLog {

private:
   TGTransientFrame     *fMain;   // main frame of this widget
   TGTextView           *fText;   // text widget
   TGTextButton         *fClose;  // close button
   TProofProgressDialog *fDialog; // owner dialog

public:
   TProofProgressLog(TProofProgressDialog *d);
   virtual ~TProofProgressLog();

   void   LoadBuffer(const char *buffer);
   void   AddBuffer(const char *buffer);

   void   LoadFile(const char *file);

   void   Clear();
   void   Popup();

   // slots
   void   CloseWindow();
   void   DoClose();

   ClassDef(TProofProgressLog,0) //Class implementing a log graphic box
};

#endif
