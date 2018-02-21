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

#include "TBufferIO.h"
#include "TString.h"

class TStreamerBase;
class TExMap;

class TBufferText : public TBufferIO {

protected:

   TBufferText();
   TBufferText(TBuffer::EMode mode, TObject *parent = nullptr);

public:

   virtual ~TBufferText();

   // virtual abstract TBuffer methods, which could be redefined here
   // probably, one can move them to TBufferImpl class, which can be base
   // for TBufferFile and TBufferText

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

   virtual void WriteObject(const TObject *obj, Bool_t cacheReuse = kTRUE);
   using TBuffer::WriteObject;

   virtual Int_t WriteObjectAny(const void *obj, const TClass *ptrClass, Bool_t cacheReuse = kTRUE);

   virtual void StreamObject(void *obj, const std::type_info &typeinfo, const TClass *onFileClass = nullptr);
   virtual void StreamObject(void *obj, const char *className, const TClass *onFileClass = nullptr);
   virtual void StreamObject(TObject *obj);
   using TBuffer::StreamObject;

   // virtual TBuffer methods, which are generic for all text-based streamers

   virtual Int_t ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *object);
   virtual Int_t ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                     void *end_collection);
   virtual Int_t
   ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection);

   virtual void ReadFloat16(Float_t *f, TStreamerElement *ele = nullptr);
   virtual void WriteFloat16(Float_t *f, TStreamerElement *ele = nullptr);
   virtual void ReadDouble32(Double_t *d, TStreamerElement *ele = nullptr);
   virtual void WriteDouble32(Double_t *d, TStreamerElement *ele = nullptr);
   virtual void ReadWithFactor(Float_t *ptr, Double_t factor, Double_t minvalue);
   virtual void ReadWithNbits(Float_t *ptr, Int_t nbits);
   virtual void ReadWithFactor(Double_t *ptr, Double_t factor, Double_t minvalue);
   virtual void ReadWithNbits(Double_t *ptr, Int_t nbits);

   virtual Int_t ReadArrayFloat16(Float_t *&f, TStreamerElement *ele = nullptr);
   virtual Int_t ReadArrayDouble32(Double_t *&d, TStreamerElement *ele = nullptr);

   virtual Int_t ReadStaticArrayFloat16(Float_t *f, TStreamerElement *ele = nullptr);
   virtual Int_t ReadStaticArrayDouble32(Double_t *d, TStreamerElement *ele = nullptr);

   virtual void ReadFastArrayFloat16(Float_t *f, Int_t n, TStreamerElement *ele = nullptr);
   virtual void ReadFastArrayDouble32(Double_t *d, Int_t n, TStreamerElement *ele = nullptr);
   virtual void ReadFastArrayWithFactor(Float_t *ptr, Int_t n, Double_t factor, Double_t minvalue);
   virtual void ReadFastArrayWithNbits(Float_t *ptr, Int_t n, Int_t nbits);
   virtual void ReadFastArrayWithFactor(Double_t *ptr, Int_t n, Double_t factor, Double_t minvalue);
   virtual void ReadFastArrayWithNbits(Double_t *ptr, Int_t n, Int_t nbits);

   virtual void WriteArrayFloat16(const Float_t *f, Int_t n, TStreamerElement *ele = nullptr);
   virtual void WriteArrayDouble32(const Double_t *d, Int_t n, TStreamerElement *ele = nullptr);

   virtual void WriteFastArrayFloat16(const Float_t *d, Int_t n, TStreamerElement *ele = nullptr);
   virtual void WriteFastArrayDouble32(const Double_t *d, Int_t n, TStreamerElement *ele = nullptr);

   // Utilities for TClass
   virtual Int_t ReadClassBuffer(const TClass * /*cl*/, void * /*pointer*/, const TClass * /*onfile_class*/ = nullptr);
   virtual Int_t ReadClassBuffer(const TClass * /*cl*/, void * /*pointer*/, Int_t /*version*/, UInt_t /*start*/,
                                 UInt_t /*count*/, const TClass * /*onfile_class*/ = nullptr);
   virtual Int_t WriteClassBuffer(const TClass *cl, void *pointer);

   // virtual abstract TBuffer methods, which are not used in text streaming

   virtual Int_t CheckByteCount(UInt_t /* startpos */, UInt_t /* bcnt */, const TClass * /* clss */) { return 0; }
   virtual Int_t CheckByteCount(UInt_t /* startpos */, UInt_t /* bcnt */, const char * /* classname */) { return 0; }
   virtual void SetByteCount(UInt_t /* cntpos */, Bool_t /* packInVersion */ = kFALSE) {}
   virtual void SkipVersion(const TClass *cl = nullptr);
   virtual Version_t ReadVersionNoCheckSum(UInt_t *, UInt_t *) { return 0; }

   virtual void Reset() { ResetMap(); }

   virtual Int_t GetBufferDisplacement() const { return fDisplacement; }
   virtual void SetBufferDisplacement() { fDisplacement = 0; }
   virtual void SetBufferDisplacement(Int_t skipped) { fDisplacement = skipped; }

   virtual Int_t ReadBuf(void * /*buf*/, Int_t /*max*/)
   {
      Error("ReadBuf", "useless in text streamers");
      return 0;
   }
   virtual void WriteBuf(const void * /*buf*/, Int_t /*max*/) { Error("WriteBuf", "useless in text streamers"); }

   virtual char *ReadString(char * /*s*/, Int_t /*max*/)
   {
      Error("ReadString", "useless");
      return nullptr;
   }
   virtual void WriteString(const char * /*s*/) { Error("WriteString", "useless"); }

   virtual Int_t GetVersionOwner() const;
   virtual void GetMappedObject(UInt_t tag, void *&ptr, TClass *&ClassPtr) const;
   virtual void MapObject(const TObject *obj, UInt_t offset = 1);
   virtual void MapObject(const void *obj, const TClass *cl, UInt_t offset = 1);

   virtual Bool_t CheckObject(const TObject *obj);
   virtual Bool_t CheckObject(const void *ptr, const TClass *cl);

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
      return nullptr;
   }

   // Utilities for TClass
   virtual Int_t ReadClassEmulated(const TClass * /*cl*/, void * /*object*/, const TClass * /*onfile_class*/ = nullptr)
   {
      Error("ReadClassEmulated", "not defined in text-based streamers");
      return 0;
   }

   virtual void WriteBaseClass(void *start, TStreamerBase *elem);

   virtual void ReadBaseClass(void *start, TStreamerBase *elem);

   static void SetFloatFormat(const char *fmt = "%e");
   static const char *GetFloatFormat();
   static void SetDoubleFormat(const char *fmt = "%.14e");
   static const char *GetDoubleFormat();

   static void CompactFloatString(char *buf, unsigned len);
   static const char *ConvertFloat(Float_t v, char *buf, unsigned len, Bool_t not_optimize = kFALSE);
   static const char *ConvertDouble(Double_t v, char *buf, unsigned len, Bool_t not_optimize = kFALSE);

protected:
   virtual void WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse) = 0;

   // method used in TBufferFile, keep here for full compatibility
   virtual void CheckCount(UInt_t) {}

   Long64_t GetMapEntry(const void *obj);

   ////////////////////////////////////////////////////////////////////////////////
   /// Return hash value for this object.
   static R__ALWAYS_INLINE ULong_t Void_Hash(const void *ptr) { return TString::Hash(&ptr, sizeof(void *)); }

   static const char *fgFloatFmt;  ///<!  printf argument for floats, either "%f" or "%e" or "%10f" and so on
   static const char *fgDoubleFmt; ///<!  printf argument for doubles, either "%f" or "%e" or "%10f" and so on

   ClassDef(TBufferText, 0); // a TBuffer subclass for all text-based streamers
};

#endif
