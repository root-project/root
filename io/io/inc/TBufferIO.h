// @(#)root/io:$Id$
// Author: Sergey Linev  21/02/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferIO
#define ROOT_TBufferIO

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBufferIO                                                            //
//                                                                      //
// Direct subclass of TBuffer, implements common methods for            //
// TBufferFile and TBufferText classes                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TBuffer.h"

#include "TString.h"

class TExMap;

class TBufferIO : public TBuffer {

protected:
   enum { kNullTag = 0 }; ///< tag value for nullptr in objects map

   Int_t fMapCount{0};         ///< Number of objects or classes in map
   Int_t fMapSize{0};          ///< Default size of map
   Int_t fDisplacement{0};     ///< Value to be added to the map offsets
   UShort_t fPidOffset{0};     ///< Offset to be added to the pid index in this key/buffer.
   TExMap *fMap{nullptr};      ///< Map containing object,offset pairs for reading/writing
   TExMap *fClassMap{nullptr}; ///< Map containing object,class pairs for reading

   static Int_t fgMapSize; ///< Default map size for all TBuffer objects

   TBufferIO() = default;

   TBufferIO(TBuffer::EMode mode);
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz);
   TBufferIO(TBuffer::EMode mode, Int_t bufsiz, void *buf, Bool_t adopt = kTRUE,
             ReAllocCharFun_t reallocfunc = nullptr);

   ////////////////////////////////////////////////////////////////////////////////
   /// Return hash value for provided object.
   static R__ALWAYS_INLINE ULong_t Void_Hash(const void *ptr) { return TString::Hash(&ptr, sizeof(void *)); }

   // method used in TBufferFile, keep here for full compatibility
   virtual void CheckCount(UInt_t) {}

   Long64_t GetObjectTag(const void *obj);

   virtual void WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse) = 0;

public:
   enum { kMapSize = 503 }; ///< default objects map size

   enum EStatusBits {
      kNotDecompressed = BIT(15),    // indicates a weird buffer, used by TBasket
      kTextBasedStreaming = BIT(18), // indicates if buffer used for XML/SQL object streaming

      kUser1 = BIT(21), // free for user
      kUser2 = BIT(22), // free for user
      kUser3 = BIT(23)  // free for user
   };

   virtual ~TBufferIO();

   virtual Int_t GetVersionOwner() const;

   // See comment in TBuffer::SetPidOffset
   virtual UShort_t GetPidOffset() const { return fPidOffset; }
   virtual void SetPidOffset(UShort_t offset);
   virtual Int_t GetBufferDisplacement() const { return fDisplacement; }
   virtual void SetBufferDisplacement() { fDisplacement = 0; }
   virtual void SetBufferDisplacement(Int_t skipped) { fDisplacement = (Int_t)(Length() - skipped); }

   // Utilities for objects map
   virtual void SetReadParam(Int_t mapsize);
   virtual void SetWriteParam(Int_t mapsize);
   virtual void InitMap();
   virtual void ResetMap();
   virtual void Reset();
   virtual Int_t GetMapCount() const { return fMapCount; }
   virtual void MapObject(const TObject *obj, UInt_t offset = 1);
   virtual void MapObject(const void *obj, const TClass *cl, UInt_t offset = 1);
   virtual Bool_t CheckObject(const TObject *obj);
   virtual Bool_t CheckObject(const void *obj, const TClass *ptrClass);
   virtual void GetMappedObject(UInt_t tag, void *&ptr, TClass *&ClassPtr) const;

   // Utilities for TStreamerInfo
   virtual void ForceWriteInfo(TVirtualStreamerInfo *info, Bool_t force);
   virtual void ForceWriteInfoClones(TClonesArray *a);
   virtual Int_t ReadClones(TClonesArray *a, Int_t nobjects, Version_t objvers);
   virtual Int_t WriteClones(TClonesArray *a, Int_t nobjects);
   virtual void TagStreamerInfo(TVirtualStreamerInfo *info);

   // Special basic ROOT objects and collections
   virtual TProcessID *GetLastProcessID(TRefTable *reftable) const;
   virtual UInt_t GetTRefExecId();
   virtual TProcessID *ReadProcessID(UShort_t pidf);
   virtual UShort_t WriteProcessID(TProcessID *pid);

   virtual Int_t WriteObjectAny(const void *obj, const TClass *ptrClass, Bool_t cacheReuse = kTRUE);
   virtual void WriteObject(const TObject *obj, Bool_t cacheReuse = kTRUE);
   using TBuffer::WriteObject;

   static void SetGlobalReadParam(Int_t mapsize);
   static void SetGlobalWriteParam(Int_t mapsize);
   static Int_t GetGlobalReadParam();
   static Int_t GetGlobalWriteParam();

   ClassDef(TBufferIO, 0) // base class, share methods for TBufferFile and TBufferText
};

#endif
