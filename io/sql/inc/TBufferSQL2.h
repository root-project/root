// @(#)root/sql
// Author: Sergey Linev  20/11/2005

#ifndef ROOT_TBufferSQL2
#define ROOT_TBufferSQL2

#include "TBufferText.h"

#include "TString.h"

#include "TObjArray.h"

class TMap;
class TExMap;
class TVirtualStreamerInfo;
class TStreamerElement;
class TObjArray;
class TMemberStreamer;
class TSQLStackObj;

class TSQLServer;
class TSQLResult;
class TSQLRow;
class TSQLFile;
class TSQLStructure;
class TSQLObjectData;
class TSQLClassInfo;

class TBufferSQL2 : public TBufferText {

   friend class TSQLStructure;

protected:
   TSQLFile *fSQL;               ///<!   instance of TSQLFile
   Int_t fIOVersion;             ///<!   I/O version from TSQLFile
   TSQLStructure *fStructure;    ///<!   structures, created by object storing
   TSQLStructure *fStk;          ///<!   pointer on current active structure (stack head)
   TString fReadBuffer;          ///<!   Buffer for read value
   Int_t fErrorFlag;             ///<!   Error id value
   Int_t fCompressLevel;         ///<!   compress level used to minimize size of data in database
   Int_t fReadVersionBuffer;     ///<!   buffer, used to by ReadVersion method
   Long64_t fObjIdCounter;       ///<!   counter of objects id
   Bool_t fIgnoreVerification;   ///<!   ignore verification of names
   TSQLObjectData *fCurrentData; ///<!
   TObjArray *fObjectsInfos;     ///<!   array of objects info for selected key
   Long64_t fFirstObjId;         ///<!   id of first object to be read from the database
   Long64_t fLastObjId;          ///<!   id of last object correspond to this key
   TMap *fPoolsMap;              ///<!   map of pools with data from different tables

   // TBufferSQL2 objects cannot be copied or assigned
   TBufferSQL2(const TBufferSQL2 &);    // not implemented
   void operator=(const TBufferSQL2 &); // not implemented

   TBufferSQL2();

   // redefined protected virtual functions

   virtual void WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse);

   // end redefined protected virtual functions

   TSQLStructure *PushStack();
   TSQLStructure *PopStack();
   TSQLStructure *Stack(Int_t depth = 0);

   void WorkWithClass(const char *classname, Version_t classversion);
   void WorkWithElement(TStreamerElement *elem, Int_t comp_type);

   Int_t SqlReadArraySize();
   Bool_t SqlObjectInfo(Long64_t objid, TString &clname, Version_t &version);
   TSQLObjectData *SqlObjectData(Long64_t objid, TSQLClassInfo *sqlinfo);

   Bool_t SqlWriteBasic(Char_t value);
   Bool_t SqlWriteBasic(Short_t value);
   Bool_t SqlWriteBasic(Int_t value);
   Bool_t SqlWriteBasic(Long_t value);
   Bool_t SqlWriteBasic(Long64_t value);
   Bool_t SqlWriteBasic(Float_t value);
   Bool_t SqlWriteBasic(Double_t value);
   Bool_t SqlWriteBasic(Bool_t value);
   Bool_t SqlWriteBasic(UChar_t value);
   Bool_t SqlWriteBasic(UShort_t value);
   Bool_t SqlWriteBasic(UInt_t value);
   Bool_t SqlWriteBasic(ULong_t value);
   Bool_t SqlWriteBasic(ULong64_t value);
   Bool_t SqlWriteValue(const char *value, const char *tname);

   void SqlReadBasic(Char_t &value);
   void SqlReadBasic(Short_t &value);
   void SqlReadBasic(Int_t &value);
   void SqlReadBasic(Long_t &value);
   void SqlReadBasic(Long64_t &value);
   void SqlReadBasic(Float_t &value);
   void SqlReadBasic(Double_t &value);
   void SqlReadBasic(Bool_t &value);
   void SqlReadBasic(UChar_t &value);
   void SqlReadBasic(UShort_t &value);
   void SqlReadBasic(UInt_t &value);
   void SqlReadBasic(ULong_t &value);
   void SqlReadBasic(ULong64_t &value);
   const char *SqlReadValue(const char *tname);
   const char *SqlReadCharStarValue();

   Int_t SqlWriteObject(const void *obj, const TClass *objClass, Bool_t cacheReuse, TMemberStreamer *streamer = 0,
                        Int_t streamer_index = 0);
   void *SqlReadObject(void *obj, TClass **cl = 0, TMemberStreamer *streamer = 0, Int_t streamer_index = 0,
                       const TClass *onFileClass = 0);
   void *SqlReadObjectDirect(void *obj, TClass **cl, Long64_t objid, TMemberStreamer *streamer = 0,
                             Int_t streamer_index = 0, const TClass *onFileClass = 0);

   void StreamObjectExtra(void *obj, TMemberStreamer *streamer, const TClass *cl, Int_t n = 0,
                          const TClass *onFileClass = nullptr);

   template <typename T>
   R__ALWAYS_INLINE void SqlReadArrayContent(T *arr, Int_t arrsize, Bool_t withsize);

   template <typename T>
   R__ALWAYS_INLINE Int_t SqlReadArray(T *&arr, Bool_t is_static = kFALSE);

   template <typename T>
   R__ALWAYS_INLINE void SqlReadFastArray(T *arr, Int_t arrsize);

   template <typename T>
   R__ALWAYS_INLINE void SqlWriteArray(T *arr, Int_t arrsize, Bool_t withsize = kFALSE);

public:
   TBufferSQL2(TBuffer::EMode mode, TSQLFile *file = nullptr);
   virtual ~TBufferSQL2();

   void SetCompressionLevel(int level) { fCompressLevel = level; }

   TSQLStructure *GetStructure() const { return fStructure; }

   Int_t GetErrorFlag() const { return fErrorFlag; }

   void SetIgnoreVerification() { fIgnoreVerification = kTRUE; }

   TSQLStructure *SqlWriteAny(const void *obj, const TClass *cl, Long64_t objid);

   void *SqlReadAny(Long64_t keyid, Long64_t objid, TClass **cl, void *obj = nullptr);

   // suppress class writing/reading

   virtual TClass *ReadClass(const TClass *cl = 0, UInt_t *objTag = 0);
   virtual void WriteClass(const TClass *cl);

   // redefined virtual functions of TBuffer

   virtual Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0, const TClass *cl = 0);
   virtual UInt_t WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);

   virtual void *ReadObjectAny(const TClass *clCast);
   virtual void SkipObjectAny();

   virtual void IncrementLevel(TVirtualStreamerInfo *);
   virtual void SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type);
   virtual void DecrementLevel(TVirtualStreamerInfo *);

   virtual void ClassBegin(const TClass *, Version_t = -1);
   virtual void ClassEnd(const TClass *);
   virtual void ClassMember(const char *name, const char *typeName = 0, Int_t arrsize1 = -1, Int_t arrsize2 = -1);

   virtual Int_t ReadArray(Bool_t *&b);
   virtual Int_t ReadArray(Char_t *&c);
   virtual Int_t ReadArray(UChar_t *&c);
   virtual Int_t ReadArray(Short_t *&h);
   virtual Int_t ReadArray(UShort_t *&h);
   virtual Int_t ReadArray(Int_t *&i);
   virtual Int_t ReadArray(UInt_t *&i);
   virtual Int_t ReadArray(Long_t *&l);
   virtual Int_t ReadArray(ULong_t *&l);
   virtual Int_t ReadArray(Long64_t *&l);
   virtual Int_t ReadArray(ULong64_t *&l);
   virtual Int_t ReadArray(Float_t *&f);
   virtual Int_t ReadArray(Double_t *&d);

   virtual Int_t ReadStaticArray(Bool_t *b);
   virtual Int_t ReadStaticArray(Char_t *c);
   virtual Int_t ReadStaticArray(UChar_t *c);
   virtual Int_t ReadStaticArray(Short_t *h);
   virtual Int_t ReadStaticArray(UShort_t *h);
   virtual Int_t ReadStaticArray(Int_t *i);
   virtual Int_t ReadStaticArray(UInt_t *i);
   virtual Int_t ReadStaticArray(Long_t *l);
   virtual Int_t ReadStaticArray(ULong_t *l);
   virtual Int_t ReadStaticArray(Long64_t *l);
   virtual Int_t ReadStaticArray(ULong64_t *l);
   virtual Int_t ReadStaticArray(Float_t *f);
   virtual Int_t ReadStaticArray(Double_t *d);

   virtual void ReadFastArray(Bool_t *b, Int_t n);
   virtual void ReadFastArray(Char_t *c, Int_t n);
   virtual void ReadFastArray(UChar_t *c, Int_t n);
   virtual void ReadFastArray(Short_t *h, Int_t n);
   virtual void ReadFastArray(UShort_t *h, Int_t n);
   virtual void ReadFastArray(Int_t *i, Int_t n);
   virtual void ReadFastArray(UInt_t *i, Int_t n);
   virtual void ReadFastArray(Long_t *l, Int_t n);
   virtual void ReadFastArray(ULong_t *l, Int_t n);
   virtual void ReadFastArray(Long64_t *l, Int_t n);
   virtual void ReadFastArray(ULong64_t *l, Int_t n);
   virtual void ReadFastArray(Float_t *f, Int_t n);
   virtual void ReadFastArray(Double_t *d, Int_t n);
   virtual void ReadFastArrayString(Char_t *c, Int_t n);
   virtual void
   ReadFastArray(void *start, const TClass *cl, Int_t n = 1, TMemberStreamer *s = 0, const TClass *onFileClass = 0);
   virtual void ReadFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE,
                              TMemberStreamer *s = 0, const TClass *onFileClass = 0);

   virtual void WriteArray(const Bool_t *b, Int_t n);
   virtual void WriteArray(const Char_t *c, Int_t n);
   virtual void WriteArray(const UChar_t *c, Int_t n);
   virtual void WriteArray(const Short_t *h, Int_t n);
   virtual void WriteArray(const UShort_t *h, Int_t n);
   virtual void WriteArray(const Int_t *i, Int_t n);
   virtual void WriteArray(const UInt_t *i, Int_t n);
   virtual void WriteArray(const Long_t *l, Int_t n);
   virtual void WriteArray(const ULong_t *l, Int_t n);
   virtual void WriteArray(const Long64_t *l, Int_t n);
   virtual void WriteArray(const ULong64_t *l, Int_t n);
   virtual void WriteArray(const Float_t *f, Int_t n);
   virtual void WriteArray(const Double_t *d, Int_t n);

   virtual void WriteFastArray(const Bool_t *b, Int_t n);
   virtual void WriteFastArray(const Char_t *c, Int_t n);
   virtual void WriteFastArray(const UChar_t *c, Int_t n);
   virtual void WriteFastArray(const Short_t *h, Int_t n);
   virtual void WriteFastArray(const UShort_t *h, Int_t n);
   virtual void WriteFastArray(const Int_t *i, Int_t n);
   virtual void WriteFastArray(const UInt_t *i, Int_t n);
   virtual void WriteFastArray(const Long_t *l, Int_t n);
   virtual void WriteFastArray(const ULong_t *l, Int_t n);
   virtual void WriteFastArray(const Long64_t *l, Int_t n);
   virtual void WriteFastArray(const ULong64_t *l, Int_t n);
   virtual void WriteFastArray(const Float_t *f, Int_t n);
   virtual void WriteFastArray(const Double_t *d, Int_t n);
   virtual void WriteFastArrayString(const Char_t *c, Int_t n);
   virtual void WriteFastArray(void *start, const TClass *cl, Int_t n = 1, TMemberStreamer *s = 0);
   virtual Int_t
   WriteFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE, TMemberStreamer *s = 0);

   virtual void StreamObject(void *obj, const TClass *cl, const TClass *onFileClass = nullptr);
   using TBufferText::StreamObject;

   virtual void ReadBool(Bool_t &b);
   virtual void ReadChar(Char_t &c);
   virtual void ReadUChar(UChar_t &c);
   virtual void ReadShort(Short_t &s);
   virtual void ReadUShort(UShort_t &s);
   virtual void ReadInt(Int_t &i);
   virtual void ReadUInt(UInt_t &i);
   virtual void ReadLong(Long_t &l);
   virtual void ReadULong(ULong_t &l);
   virtual void ReadLong64(Long64_t &l);
   virtual void ReadULong64(ULong64_t &l);
   virtual void ReadFloat(Float_t &f);
   virtual void ReadDouble(Double_t &d);
   virtual void ReadCharP(Char_t *c);
   virtual void ReadTString(TString &s);
   virtual void ReadStdString(std::string *s);
   using TBuffer::ReadStdString;
   virtual void ReadCharStar(char *&s);

   virtual void WriteBool(Bool_t b);
   virtual void WriteChar(Char_t c);
   virtual void WriteUChar(UChar_t c);
   virtual void WriteShort(Short_t s);
   virtual void WriteUShort(UShort_t s);
   virtual void WriteInt(Int_t i);
   virtual void WriteUInt(UInt_t i);
   virtual void WriteLong(Long_t l);
   virtual void WriteULong(ULong_t l);
   virtual void WriteLong64(Long64_t l);
   virtual void WriteULong64(ULong64_t l);
   virtual void WriteFloat(Float_t f);
   virtual void WriteDouble(Double_t d);
   virtual void WriteCharP(const Char_t *c);
   virtual void WriteTString(const TString &s);
   virtual void WriteStdString(const std::string *s);
   using TBuffer::WriteStdString;
   virtual void WriteCharStar(char *s);

   virtual TVirtualStreamerInfo *GetInfo();

   // end of redefined virtual functions

   ClassDef(TBufferSQL2, 0); // a specialized TBuffer to convert data to SQL statements or read data from SQL tables
};

#endif
