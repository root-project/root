// @(#)root/gui:$Id$
// Author: Fons Rademakers   05/05/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextBuffer
#define ROOT_TGTextBuffer

#include "TString.h"

class TGTextBuffer {

private:
   TString    fBuffer;

   TGTextBuffer(const TGTextBuffer&) = delete;
   TGTextBuffer& operator=(const TGTextBuffer&) = delete;

public:
   TGTextBuffer() : fBuffer() {}
   TGTextBuffer(Int_t length): fBuffer(length) {}
   virtual ~TGTextBuffer() {}

   UInt_t GetTextLength() const { return fBuffer.Length(); }
   UInt_t GetBufferLength() const { return fBuffer.Capacity(); }
   const char *GetString() const { return fBuffer.Data(); }

   void AddText(Int_t pos, const char *text) { fBuffer.Insert(pos, text); }
   void AddText(Int_t pos, const char *text, Int_t length) { fBuffer.Insert(pos, text, length); }
   void RemoveText(Int_t pos, Int_t length) { fBuffer.Remove(pos, length); }
   void Clear() { fBuffer.Remove(0, fBuffer.Length()); }

   ClassDef(TGTextBuffer,0)  // Text buffer used by widgets like TGTextEntry, etc.
};

#endif
