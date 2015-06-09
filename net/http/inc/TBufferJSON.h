// $Id$
// Author: Sergey Linev  4.03.2014

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBufferJSON
#define ROOT_TBufferJSON

#ifndef ROOT_TBuffer
#include "TBuffer.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

#include <map>

class TVirtualStreamerInfo;
class TStreamerInfo;
class TStreamerElement;
class TObjArray;
class TMemberStreamer;
class TDataMember;
class TJSONStackObj;


class TBufferJSON : public TBuffer {

public:

   TBufferJSON();
   virtual ~TBufferJSON();

   void SetCompact(int level);

   static TString   ConvertToJSON(const TObject *obj, Int_t compact = 0);
   static TString   ConvertToJSON(const void *obj, const TClass *cl, Int_t compact = 0);
   static TString   ConvertToJSON(const void *obj, TDataMember *member, Int_t compact = 0);

   // suppress class writing/reading

   virtual TClass  *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void     WriteClass(const TClass *cl);

   // redefined virtual functions of TBuffer

   virtual Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss); // SL
   virtual Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const char *classname); // SL
   virtual void     SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);  // SL

   virtual void      SkipVersion(const TClass *cl = 0);
   virtual Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0, const TClass *cl = 0);  // SL
   virtual Version_t ReadVersionNoCheckSum(UInt_t *, UInt_t *)
   {
      return 0;
   }
   virtual UInt_t    WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);  // SL

   virtual void    *ReadObjectAny(const TClass *clCast);
   virtual void     SkipObjectAny();

   // these methods used in streamer info to indicate currently streamed element,
   virtual void     IncrementLevel(TVirtualStreamerInfo *);
   virtual void     SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type);
   virtual void     DecrementLevel(TVirtualStreamerInfo *);

   virtual void     ClassBegin(const TClass *, Version_t = -1);
   virtual void     ClassEnd(const TClass *);
   virtual void     ClassMember(const char *name, const char *typeName = 0, Int_t arrsize1 = -1, Int_t arrsize2 = -1);

   virtual void     WriteObject(const TObject *obj);

   virtual void     ReadFloat16(Float_t *f, TStreamerElement *ele = 0);
   virtual void     WriteFloat16(Float_t *f, TStreamerElement *ele = 0);
   virtual void     ReadDouble32(Double_t *d, TStreamerElement *ele = 0);
   virtual void     WriteDouble32(Double_t *d, TStreamerElement *ele = 0);
   virtual void     ReadWithFactor(Float_t *ptr, Double_t factor, Double_t minvalue);
   virtual void     ReadWithNbits(Float_t *ptr, Int_t nbits);
   virtual void     ReadWithFactor(Double_t *ptr, Double_t factor, Double_t minvalue);
   virtual void     ReadWithNbits(Double_t *ptr, Int_t nbits);

   virtual Int_t    ReadArray(Bool_t    *&b);
   virtual Int_t    ReadArray(Char_t    *&c);
   virtual Int_t    ReadArray(UChar_t   *&c);
   virtual Int_t    ReadArray(Short_t   *&h);
   virtual Int_t    ReadArray(UShort_t  *&h);
   virtual Int_t    ReadArray(Int_t     *&i);
   virtual Int_t    ReadArray(UInt_t    *&i);
   virtual Int_t    ReadArray(Long_t    *&l);
   virtual Int_t    ReadArray(ULong_t   *&l);
   virtual Int_t    ReadArray(Long64_t  *&l);
   virtual Int_t    ReadArray(ULong64_t *&l);
   virtual Int_t    ReadArray(Float_t   *&f);
   virtual Int_t    ReadArray(Double_t  *&d);
   virtual Int_t    ReadArrayFloat16(Float_t  *&f, TStreamerElement *ele = 0);
   virtual Int_t    ReadArrayDouble32(Double_t  *&d, TStreamerElement *ele = 0);

   virtual Int_t    ReadStaticArray(Bool_t    *b);
   virtual Int_t    ReadStaticArray(Char_t    *c);
   virtual Int_t    ReadStaticArray(UChar_t   *c);
   virtual Int_t    ReadStaticArray(Short_t   *h);
   virtual Int_t    ReadStaticArray(UShort_t  *h);
   virtual Int_t    ReadStaticArray(Int_t     *i);
   virtual Int_t    ReadStaticArray(UInt_t    *i);
   virtual Int_t    ReadStaticArray(Long_t    *l);
   virtual Int_t    ReadStaticArray(ULong_t   *l);
   virtual Int_t    ReadStaticArray(Long64_t  *l);
   virtual Int_t    ReadStaticArray(ULong64_t *l);
   virtual Int_t    ReadStaticArray(Float_t   *f);
   virtual Int_t    ReadStaticArray(Double_t  *d);
   virtual Int_t    ReadStaticArrayFloat16(Float_t  *f, TStreamerElement *ele = 0);
   virtual Int_t    ReadStaticArrayDouble32(Double_t  *d, TStreamerElement *ele = 0);

   virtual void     ReadFastArray(Bool_t    *b, Int_t n);
   virtual void     ReadFastArray(Char_t    *c, Int_t n);
   virtual void     ReadFastArrayString(Char_t *c, Int_t n);
   virtual void     ReadFastArray(UChar_t   *c, Int_t n);
   virtual void     ReadFastArray(Short_t   *h, Int_t n);
   virtual void     ReadFastArray(UShort_t  *h, Int_t n);
   virtual void     ReadFastArray(Int_t     *i, Int_t n);
   virtual void     ReadFastArray(UInt_t    *i, Int_t n);
   virtual void     ReadFastArray(Long_t    *l, Int_t n);
   virtual void     ReadFastArray(ULong_t   *l, Int_t n);
   virtual void     ReadFastArray(Long64_t  *l, Int_t n);
   virtual void     ReadFastArray(ULong64_t *l, Int_t n);
   virtual void     ReadFastArray(Float_t   *f, Int_t n);
   virtual void     ReadFastArray(Double_t  *d, Int_t n);
   virtual void     ReadFastArrayFloat16(Float_t  *f, Int_t n, TStreamerElement *ele = 0);
   virtual void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele = 0);
   virtual void     ReadFastArrayWithFactor(Float_t *ptr, Int_t n, Double_t factor, Double_t minvalue) ;
   virtual void     ReadFastArrayWithNbits(Float_t *ptr, Int_t n, Int_t nbits);
   virtual void     ReadFastArrayWithFactor(Double_t *ptr, Int_t n, Double_t factor, Double_t minvalue);
   virtual void     ReadFastArrayWithNbits(Double_t *ptr, Int_t n, Int_t nbits) ;

   virtual void     WriteArray(const Bool_t    *b, Int_t n);
   virtual void     WriteArray(const Char_t    *c, Int_t n);
   virtual void     WriteArray(const UChar_t   *c, Int_t n);
   virtual void     WriteArray(const Short_t   *h, Int_t n);
   virtual void     WriteArray(const UShort_t  *h, Int_t n);
   virtual void     WriteArray(const Int_t     *i, Int_t n);
   virtual void     WriteArray(const UInt_t    *i, Int_t n);
   virtual void     WriteArray(const Long_t    *l, Int_t n);
   virtual void     WriteArray(const ULong_t   *l, Int_t n);
   virtual void     WriteArray(const Long64_t  *l, Int_t n);
   virtual void     WriteArray(const ULong64_t *l, Int_t n);
   virtual void     WriteArray(const Float_t   *f, Int_t n);
   virtual void     WriteArray(const Double_t  *d, Int_t n);
   virtual void     WriteArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement *ele = 0);
   virtual void     WriteArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele = 0);
   virtual void     ReadFastArray(void  *start , const TClass *cl, Int_t n = 1, TMemberStreamer *s = 0, const TClass *onFileClass = 0);
   virtual void     ReadFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE, TMemberStreamer *s = 0, const TClass *onFileClass = 0);

   virtual void     WriteFastArray(const Bool_t    *b, Int_t n);
   virtual void     WriteFastArray(const Char_t    *c, Int_t n);
   virtual void     WriteFastArrayString(const Char_t    *c, Int_t n);
   virtual void     WriteFastArray(const UChar_t   *c, Int_t n);
   virtual void     WriteFastArray(const Short_t   *h, Int_t n);
   virtual void     WriteFastArray(const UShort_t  *h, Int_t n);
   virtual void     WriteFastArray(const Int_t     *i, Int_t n);
   virtual void     WriteFastArray(const UInt_t    *i, Int_t n);
   virtual void     WriteFastArray(const Long_t    *l, Int_t n);
   virtual void     WriteFastArray(const ULong_t   *l, Int_t n);
   virtual void     WriteFastArray(const Long64_t  *l, Int_t n);
   virtual void     WriteFastArray(const ULong64_t *l, Int_t n);
   virtual void     WriteFastArray(const Float_t   *f, Int_t n);
   virtual void     WriteFastArray(const Double_t  *d, Int_t n);
   virtual void     WriteFastArrayFloat16(const Float_t  *d, Int_t n, TStreamerElement *ele = 0);
   virtual void     WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele = 0);
   virtual void     WriteFastArray(void  *start,  const TClass *cl, Int_t n = 1, TMemberStreamer *s = 0);
   virtual Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE, TMemberStreamer *s = 0);

   virtual void     StreamObject(void *obj, const type_info &typeinfo, const TClass *onFileClass = 0);
   virtual void     StreamObject(void *obj, const char *className, const TClass *onFileClass = 0);
   virtual void     StreamObject(void *obj, const TClass *cl, const TClass *onFileClass = 0);
   virtual void     StreamObject(TObject *obj);

   virtual   void     ReadBool(Bool_t       &b);
   virtual   void     ReadChar(Char_t       &c);
   virtual   void     ReadUChar(UChar_t     &c);
   virtual   void     ReadShort(Short_t     &s);
   virtual   void     ReadUShort(UShort_t   &s);
   virtual   void     ReadInt(Int_t         &i);
   virtual   void     ReadUInt(UInt_t       &i);
   virtual   void     ReadLong(Long_t       &l);
   virtual   void     ReadULong(ULong_t     &l);
   virtual   void     ReadLong64(Long64_t   &l);
   virtual   void     ReadULong64(ULong64_t &l);
   virtual   void     ReadFloat(Float_t     &f);
   virtual   void     ReadDouble(Double_t   &d);
   virtual   void     ReadCharP(Char_t      *c);
   virtual   void     ReadTString(TString   &s);
   virtual   void     ReadStdString(std::string &s);

   virtual   void     WriteBool(Bool_t       b);
   virtual   void     WriteChar(Char_t       c);
   virtual   void     WriteUChar(UChar_t     c);
   virtual   void     WriteShort(Short_t     s);
   virtual   void     WriteUShort(UShort_t   s);
   virtual   void     WriteInt(Int_t         i);
   virtual   void     WriteUInt(UInt_t       i);
   virtual   void     WriteLong(Long_t       l);
   virtual   void     WriteULong(ULong_t     l);
   virtual   void     WriteLong64(Long64_t   l);
   virtual   void     WriteULong64(ULong64_t l);
   virtual   void     WriteFloat(Float_t     f);
   virtual   void     WriteDouble(Double_t   d);
   virtual   void     WriteCharP(const Char_t *c);
   virtual   void     WriteTString(const TString &s);
   virtual   void     WriteStdString(const std::string &s);

   virtual   Int_t    WriteClones(TClonesArray *a, Int_t nobjects);

   virtual   Int_t    WriteObjectAny(const void *obj, const TClass *ptrClass);
   virtual   Int_t    WriteClassBuffer(const TClass *cl, void *pointer);

   virtual Int_t      ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *object);
   virtual Int_t      ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection);
   virtual Int_t      ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection);

   virtual void       TagStreamerInfo(TVirtualStreamerInfo * /*info*/) {}

   virtual Bool_t     CheckObject(const TObject * /*obj*/);

   virtual Bool_t     CheckObject(const void * /*ptr*/, const TClass * /*cl*/);

   // abstract virtual methods from TBuffer, which should be redefined

   virtual Int_t      ReadBuf(void * /*buf*/, Int_t /*max*/)
   {
      Error("ReadBuf", "useless");
      return 0;
   }
   virtual void       WriteBuf(const void * /*buf*/, Int_t /*max*/)
   {
      Error("WriteBuf", "useless");
   }

   virtual char      *ReadString(char * /*s*/, Int_t /*max*/)
   {
      Error("ReadString", "useless");
      return 0;
   }
   virtual void       WriteString(const char * /*s*/)
   {
      Error("WriteString", "useless");
   }

   virtual Int_t      GetVersionOwner() const
   {
      Error("GetVersionOwner", "useless");
      return 0;
   }
   virtual Int_t      GetMapCount() const
   {
      Error("GetMapCount", "useless");
      return 0;
   }
   virtual void       GetMappedObject(UInt_t /*tag*/, void *&/*ptr*/, TClass *&/*ClassPtr*/) const
   {
      Error("GetMappedObject", "useless");
   }
   virtual void       MapObject(const TObject * /*obj*/, UInt_t /*offset*/ = 1)
   {
      Error("MapObject", "useless");
   }
   virtual void       MapObject(const void * /*obj*/, const TClass * /*cl*/, UInt_t /*offset*/ = 1)
   {
      Error("MapObject", "useless");
   }
   virtual void       Reset()
   {
      Error("Reset", "useless");
   }
   virtual void       InitMap()
   {
      Error("InitMap", "useless");
   }
   virtual void       ResetMap()
   {
      Error("ResetMap", "useless");
   }
   virtual void       SetReadParam(Int_t /*mapsize*/)
   {
      Error("SetReadParam", "useless");
   }
   virtual void       SetWriteParam(Int_t /*mapsize*/)
   {
      Error("SetWriteParam", "useless");
   }

   virtual Version_t  ReadVersionForMemberWise(const TClass * /*cl*/ = 0)
   {
      Error("ReadVersionForMemberWise", "useless");
      return 0;
   }
   virtual UInt_t     WriteVersionMemberWise(const TClass * /*cl*/, Bool_t /*useBcnt*/ = kFALSE)
   {
      Error("WriteVersionMemberWise", "useless");
      return 0;
   }

   virtual TVirtualStreamerInfo *GetInfo()
   {
      Error("GetInfo", "useless");
      return 0;
   }

   virtual TObject   *ReadObject(const TClass * /*cl*/)
   {
      Error("ReadObject", "useless");
      return 0;
   }

   virtual UShort_t   GetPidOffset() const
   {
      Error("GetPidOffset", "useless");
      return 0;
   }
   virtual void       SetPidOffset(UShort_t /*offset*/)
   {
      Error("SetPidOffset", "useless");
   }
   virtual Int_t      GetBufferDisplacement() const
   {
      Error("GetBufferDisplacement", "useless");
      return 0;
   }
   virtual void       SetBufferDisplacement()
   {
      Error("SetBufferDisplacement", "useless");
   }
   virtual void       SetBufferDisplacement(Int_t /*skipped*/)
   {
      Error("SetBufferDisplacement", "useless");
   }

   virtual   TProcessID *GetLastProcessID(TRefTable * /*reftable*/) const
   {
      Error("GetLastProcessID", "useless");
      return 0;
   }
   virtual   UInt_t      GetTRefExecId()
   {
      Error("GetTRefExecId", "useless");
      return 0;
   }
   virtual   TProcessID *ReadProcessID(UShort_t /*pidf*/)
   {
      Error("ReadProcessID", "useless");
      return 0;
   }
   virtual   UShort_t    WriteProcessID(TProcessID * /*pid*/)
   {
      Error("WriteProcessID", "useless");
      return 0;
   }

   // Utilities for TStreamerInfo
   virtual   void     ForceWriteInfo(TVirtualStreamerInfo * /*info*/, Bool_t /*force*/)
   {
      Error("ForceWriteInfo", "useless");
   }
   virtual   void     ForceWriteInfoClones(TClonesArray * /*a*/)
   {
      Error("ForceWriteInfoClones", "useless");
   }
   virtual   Int_t    ReadClones(TClonesArray * /*a*/, Int_t /*nobjects*/, Version_t /*objvers*/)
   {
      Error("ReadClones", "useless");
      return 0;
   }

   // Utilities for TClass
   virtual   Int_t    ReadClassEmulated(const TClass * /*cl*/, void * /*object*/, const TClass * /*onfile_class*/ = 0)
   {
      Error("ReadClassEmulated", "useless");
      return 0;
   }
   virtual   Int_t    ReadClassBuffer(const TClass * /*cl*/, void * /*pointer*/, const TClass * /*onfile_class*/ = 0)
   {
      Error("ReadClassBuffer", "useless");
      return 0;
   }
   virtual   Int_t    ReadClassBuffer(const TClass * /*cl*/, void * /*pointer*/, Int_t /*version*/, UInt_t /*start*/, UInt_t /*count*/, const TClass * /*onfile_class*/ = 0)
   {
      Error("ReadClassBuffer", "useless");
      return 0;
   }

   // end of redefined virtual functions

   static    void     SetFloatFormat(const char *fmt = "%e");
   static const char *GetFloatFormat();


protected:
   // redefined protected virtual functions

   virtual void     WriteObjectClass(const void *actualObjStart, const TClass *actualClass);

   // end redefined protected virtual functions

   TString          JsonWriteMember(const void *ptr, TDataMember *member, TClass *memberClass);

   TJSONStackObj   *PushStack(Int_t inclevel = 0);
   TJSONStackObj   *PopStack();
   TJSONStackObj   *Stack(Int_t depth = 0);

   void             WorkWithClass(TStreamerInfo *info, const TClass *cl = 0);
   void             WorkWithElement(TStreamerElement *elem, Int_t comp_type);


   void             JsonDisablePostprocessing();
   Int_t            JsonSpecialClass(const TClass *cl) const;

   void             JsonStartElement(const TStreamerElement *elem, const TClass *base_class = 0);

   void             PerformPostProcessing(TJSONStackObj *stack, const TStreamerElement *elem = 0);

   void              JsonWriteBasic(Char_t value);
   void              JsonWriteBasic(Short_t value);
   void              JsonWriteBasic(Int_t value);
   void              JsonWriteBasic(Long_t value);
   void              JsonWriteBasic(Long64_t value);
   void              JsonWriteBasic(Float_t value);
   void              JsonWriteBasic(Double_t value);
   void              JsonWriteBasic(Bool_t value);
   void              JsonWriteBasic(UChar_t value);
   void              JsonWriteBasic(UShort_t value);
   void              JsonWriteBasic(UInt_t value);
   void              JsonWriteBasic(ULong_t value);
   void              JsonWriteBasic(ULong64_t value);

   void              JsonWriteConstChar(const char* value, Int_t len = -1);

   void              JsonWriteObject(const void *obj, const TClass *objClass, Bool_t check_map = kTRUE);

   void              JsonStreamCollection(TCollection *obj, const TClass *objClass);

   void              AppendOutput(const char *line0, const char *line1 = 0);

   TString                   fOutBuffer;    //!  main output buffer for json code
   TString                  *fOutput;       //!  current output buffer for json code
   TString                   fValue;        //!  buffer for current value
   std::map<const void *, unsigned>  fJsonrMap;   //!  map of recorded objects, used in JsonR to restore references
   unsigned                  fJsonrCnt;     //!  counter for all objects and arrays
   TObjArray                 fStack;        //!  stack of streamer infos
   Bool_t                    fExpectedChain; //!   flag to resolve situation when several elements of same basic type stored as FastArray
   Int_t                     fCompact;       //!  0 - no any compression, 1 - no spaces in the begin, 2 - no new lines, 3 - no spaces at all
   TString                   fSemicolon;     //!  depending from compression level, " : " or ":"
   TString                   fArraySepar;    //!  depending from compression level, ", " or ","
   TString                   fNumericLocale; //!  stored value of setlocale(LC_NUMERIC), which should be recovered at the end

   static const char *fgFloatFmt;          //!  printf argument for floats and doubles, either "%f" or "%e" or "%10f" and so on

   ClassDef(TBufferJSON, 1) //a specialized TBuffer to only write objects into JSON format
};

#endif


