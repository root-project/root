// $Id$
// Author: Sergey Linev  21.12.2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferText
#define ROOT_TBufferText

#include "TBuffer.h"

class TBufferText : public TBuffer {

protected:
   UShort_t  fPidOffset;

   TBufferText();
   TBufferText(TBuffer::EMode mode, TObject *parent = nullptr);

public:
   // virtual abstract TBuffer methods, which could be redefined here

   virtual TProcessID *GetLastProcessID(TRefTable *reftable) const;
   virtual UInt_t GetTRefExecId();
   virtual TProcessID *ReadProcessID(UShort_t pidf);
   virtual UShort_t WriteProcessID(TProcessID *pid);

   virtual UShort_t GetPidOffset() const { return fPidOffset; }
   virtual void SetPidOffset(UShort_t offset) { fPidOffset = offset; }

   virtual void TagStreamerInfo(TVirtualStreamerInfo * /*info*/);

   // virtual abstract TBuffer methods, which are not used in text streaming

   virtual void Reset() { Error("Reset", "useless"); }
   virtual void InitMap() { Error("InitMap", "useless"); }
   virtual void ResetMap() { Error("ResetMap", "useless"); }
   virtual void SetReadParam(Int_t /*mapsize*/) { Error("SetReadParam", "useless"); }
   virtual void SetWriteParam(Int_t /*mapsize*/) { Error("SetWriteParam", "useless"); }

   virtual Int_t GetBufferDisplacement() const
   {
      Error("GetBufferDisplacement", "useless");
      return 0;
   }
   virtual void SetBufferDisplacement() { Error("SetBufferDisplacement", "useless"); }
   virtual void SetBufferDisplacement(Int_t /*skipped*/) { Error("SetBufferDisplacement", "useless"); }

   virtual Int_t ReadBuf(void * /*buf*/, Int_t /*max*/)
   {
      Error("ReadBuf", "useless in text streamers");
      return 0;
   }
   virtual void WriteBuf(const void * /*buf*/, Int_t /*max*/) { Error("WriteBuf", "useless in text streamers"); }


   ClassDef(TBufferText, 0); // a TBuffer subclass for all text-based streamers
};

#endif
