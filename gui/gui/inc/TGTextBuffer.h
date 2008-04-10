// @(#)root/gui:$Id$
// Author: Fons Rademakers   05/05/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextBuffer
#define ROOT_TGTextBuffer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextBuffer                                                         //
//                                                                      //
// A text buffer is used in several widgets, like TGTextEntry,          //
// TGFileDialog, etc. It is a little wrapper around the powerful        //
// TString class and used for sinlge line texts. For multi line texts   //
// use TGText.                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif


class TGTextBuffer {

private:
   TString    *fBuffer;

protected:
   TGTextBuffer(const TGTextBuffer& tb): fBuffer(tb.fBuffer) { }
   TGTextBuffer& operator=(const TGTextBuffer& tb)
     {if(this!=&tb) fBuffer=tb.fBuffer; return *this;}

public:
   TGTextBuffer(): fBuffer(new TString) { }
   TGTextBuffer(Int_t length): fBuffer(new TString(length)) { }
   virtual ~TGTextBuffer() { delete fBuffer; }

   UInt_t GetTextLength() const { return fBuffer->Length(); }
   UInt_t GetBufferLength() const { return fBuffer->Capacity(); }
   const char *GetString() const { return fBuffer->Data(); }

   void AddText(Int_t pos, const char *text) { fBuffer->Insert(pos, text); }
   void AddText(Int_t pos, const char *text, Int_t length) { fBuffer->Insert(pos, text, length); }
   void RemoveText(Int_t pos, Int_t length) { fBuffer->Remove(pos, length); }
   void Clear() { fBuffer->Remove(0, fBuffer->Length()); }

   ClassDef(TGTextBuffer,0)  // Text buffer used by widgets like TGTextEntry, etc.
};

#endif
