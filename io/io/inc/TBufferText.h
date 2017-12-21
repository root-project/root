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
   UShort_t fPidOffset;

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

   virtual void TagStreamerInfo(TVirtualStreamerInfo *info);

   // Utilities for TStreamerInfo
   virtual void ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force);
   virtual void ForceWriteInfoClones(TClonesArray *a);
   virtual Int_t ReadClones(TClonesArray *a, Int_t nobjects, Version_t objvers);
   virtual Int_t WriteClones(TClonesArray *a, Int_t nobjects);

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

   virtual char *ReadString(char * /*s*/, Int_t /*max*/)
   {
      Error("ReadString", "useless");
      return 0;
   }
   virtual void WriteString(const char * /*s*/) { Error("WriteString", "useless"); }

   virtual Int_t GetVersionOwner() const
   {
      Error("GetVersionOwner", "useless");
      return 0;
   }
   virtual Int_t GetMapCount() const
   {
      Error("GetMapCount", "useless");
      return 0;
   }
   virtual void GetMappedObject(UInt_t /*tag*/, void *& /*ptr*/, TClass *& /*ClassPtr*/) const
   {
      Error("GetMappedObject", "useless");
   }
   // for the moment object map is individually implemented in JSON/XML, not used in text streamers
   virtual void MapObject(const TObject * /*obj*/, UInt_t /*offset*/ = 1)
   {
      Error("MapObject", "should not be used in text streaming");
   }
   virtual void MapObject(const void * /*obj*/, const TClass * /*cl*/, UInt_t /*offset*/ = 1)
   {
      Error("MapObject", "should not be used in text streaming");
   }

   virtual Version_t ReadVersionForMemberWise(const TClass * /*cl*/ = nullptr)
   {
      Error("ReadVersionForMemberWise", "not defined in text-based streamers");
      return 0;
   }
   virtual UInt_t WriteVersionMemberWise(const TClass * /*cl*/, Bool_t /*useBcnt*/ = kFALSE)
   {
      Error("WriteVersionMemberWise", "not defined in text-based streamers");
      return 0;
   }

   virtual TObject *ReadObject(const TClass * /*cl*/)
   {
      Error("ReadObject", "not yet implemented for text-based streamers");
      return 0;
   }

   // Utilities for TClass
   virtual Int_t ReadClassEmulated(const TClass * /*cl*/, void * /*object*/, const TClass * /*onfile_class*/ = nullptr)
   {
      Error("ReadClassEmulated", "not defined in text-based streamers");
      return 0;
   }

   ClassDef(TBufferText, 0); // a TBuffer subclass for all text-based streamers
};

#endif
