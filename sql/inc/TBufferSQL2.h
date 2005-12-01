// @(#)root/net:$Name:  $:$Id: TBufferSQL2.h,v 1.3 2005/11/28 23:22:31 pcanal Exp $
// Author: Sergey Linev  20/11/2005


#ifndef ROOT_TBufferSQL2
#define ROOT_TBufferSQL2


/////////////////////////////////////////////////////////////////////////
//                                                                     //
// TBufferSQL2 class used in TSQLFile to convert binary object data    //
// to SQL statements, supplied to DB server                            //
//                                                                     //
/////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBuffer
#include "TBuffer.h"
#endif

#ifndef ROOT_TString
#include "TString.h"
#endif

#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TExMap;
class TStreamerInfo;
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

class TBufferSQL2 : public TBuffer {
   protected:

      TSQLFile*        fSQL;                  //!   instance of TSQLFile
      TSQLStructure*   fStructure;            //!   structures, created by object storing
      TSQLStructure*   fStk;                  //!   pointer on current active structure (stack head)
      TExMap*          fObjMap;               //!   Map between stored objects and object id
      TObjArray*       fIdArray;              //!   List of used objects ids
      TString          fReadBuffer;           //!   Buffer for read value
      Int_t            fErrorFlag;            //!   Error id value 
      Bool_t           fExpectedChain;        //!   flag to resolve situation when several elements of same basic type stored as FastArray
      Int_t            fCompressLevel;        //!   compress level used to minimize size of data in database
      Int_t            fReadVersionBuffer;    //!   buffer, used to by ReadVersion method
      Int_t            fObjIdCounter;         //!   counter of objects id
      Bool_t           fIgnoreVerification;   //!   ignore verification of names 
      TSQLObjectData*  fCurrentData;          //!   

      // TBufferSQL2 objects cannot be copied or assigned
      TBufferSQL2(const TBufferSQL2 &);       // not implemented
      void operator=(const TBufferSQL2 &);    // not implemented

      TBufferSQL2();

      // redefined protected virtual functions

      virtual void     WriteObject(const void *actualObjStart, const TClass *actualClass);

      // end redefined protected virtual functions

      TSQLStructure*   PushStack();
      TSQLStructure*   PopStack();
      TSQLStructure*   Stack(Int_t depth = 0);

      Bool_t           ProcessPointer(const void* ptr, Int_t& objid);
      void             RegisterPointer(const void* ptr, Int_t objid);
      
      void             WorkWithElement(TStreamerElement* elem, Int_t number);

      Int_t            SqlReadArraySize();

      Bool_t           SqlWriteBasic(Char_t value);
      Bool_t           SqlWriteBasic(Short_t value);
      Bool_t           SqlWriteBasic(Int_t value);
      Bool_t           SqlWriteBasic(Long_t value);
      Bool_t           SqlWriteBasic(Long64_t value);
      Bool_t           SqlWriteBasic(Float_t value);
      Bool_t           SqlWriteBasic(Double_t value);
      Bool_t           SqlWriteBasic(Bool_t value);
      Bool_t           SqlWriteBasic(UChar_t value);
      Bool_t           SqlWriteBasic(UShort_t value);
      Bool_t           SqlWriteBasic(UInt_t value);
      Bool_t           SqlWriteBasic(ULong_t value);
      Bool_t           SqlWriteBasic(ULong64_t value);
      Bool_t           SqlWriteValue(const char* value, const char* tname);

      void             SqlReadBasic(Char_t& value);
      void             SqlReadBasic(Short_t& value);
      void             SqlReadBasic(Int_t& value);
      void             SqlReadBasic(Long_t& value);
      void             SqlReadBasic(Long64_t& value);
      void             SqlReadBasic(Float_t& value);
      void             SqlReadBasic(Double_t& value);
      void             SqlReadBasic(Bool_t& value);
      void             SqlReadBasic(UChar_t& value);
      void             SqlReadBasic(UShort_t& value);
      void             SqlReadBasic(UInt_t& value);
      void             SqlReadBasic(ULong_t& value);
      void             SqlReadBasic(ULong64_t& value);
      const char*      SqlReadValue(const char* tname);
      const char*      SqlReadCharStarValue();

      Int_t            SqlWriteObject(const void* obj, const TClass* objClass, TMemberStreamer *streamer = 0, Int_t streamer_index = 0);
      void*            SqlReadObject(void* obj, TClass** cl = 0, TMemberStreamer *streamer = 0, Int_t streamer_index = 0);
      void*            SqlReadObjectDirect(void* obj, TClass** cl, Int_t objid, TMemberStreamer *streamer = 0, Int_t streamer_index = 0);
    
   public:
   
      TBufferSQL2(TBuffer::EMode mode);
      TBufferSQL2(TBuffer::EMode mode, TSQLFile* file);
      virtual ~TBufferSQL2();
      
      void             SetCompressionLevel(int level) { fCompressLevel = level; }

      TSQLStructure*   GetStructure() const { return fStructure; }
      
      Int_t            GetErrorFlag() const { return fErrorFlag; }
      
      void             SetIgnoreVerification() { fIgnoreVerification = kTRUE; }

      TSQLStructure*   SqlWrite(const TObject* obj, Int_t objid);
      TSQLStructure*   SqlWrite(const void* obj, const TClass* cl, Int_t objid);

      TObject*         SqlRead(Int_t objid);
      void*            SqlReadAny(Int_t objid, TClass** cl);

      // suppress class writing/reading

      virtual TClass*  ReadClass(const TClass* cl = 0, UInt_t* objTag = 0);
      virtual void     WriteClass(const TClass* cl);

      // redefined virtual functions of TBuffer

      virtual Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const TClass *clss); // SL
      virtual Int_t    CheckByteCount(UInt_t startpos, UInt_t bcnt, const char *classname); // SL
      virtual void     SetByteCount(UInt_t cntpos, Bool_t packInVersion = kFALSE);  // SL

      virtual Version_t ReadVersion(UInt_t *start = 0, UInt_t *bcnt = 0, const TClass *cl = 0);  // SL
      virtual UInt_t   WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE);  // SL

      virtual void*    ReadObjectAny(const TClass* clCast);
      virtual void     SkipObjectAny();

      virtual void     IncrementLevel(TStreamerInfo*);
      virtual void     SetStreamerElementNumber(Int_t);
      virtual void     DecrementLevel(TStreamerInfo*);

      virtual void     WriteObject(const TObject *obj);

      virtual void     ReadDouble32 (Double_t *d, TStreamerElement *ele=0);
      virtual void     WriteDouble32(Double_t *d, TStreamerElement *ele=0);

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
      virtual Int_t    ReadArrayDouble32(Double_t  *&d, TStreamerElement *ele=0);

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
      virtual Int_t    ReadStaticArrayDouble32(Double_t  *d, TStreamerElement *ele=0);

      virtual void     ReadFastArray(Bool_t    *b, Int_t n);
      virtual void     ReadFastArray(Char_t    *c, Int_t n);
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
      virtual void     ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement *ele=0);

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
      virtual void     WriteArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele=0);
      virtual void     ReadFastArray(void  *start , const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
      virtual void     ReadFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

      virtual void     WriteFastArray(const Bool_t    *b, Int_t n);
      virtual void     WriteFastArray(const Char_t    *c, Int_t n);
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
      virtual void     WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement *ele=0);
      virtual void     WriteFastArray(void  *start,  const TClass *cl, Int_t n=1, TMemberStreamer *s=0);
      virtual Int_t    WriteFastArray(void **startp, const TClass *cl, Int_t n=1, Bool_t isPreAlloc=kFALSE, TMemberStreamer *s=0);

      virtual void     StreamObject(void *obj, const type_info &typeinfo);
      virtual void     StreamObject(void *obj, const char *className);
      virtual void     StreamObject(void *obj, const TClass *cl);
      virtual void     StreamObject(void *obj, TMemberStreamer *streamer, const TClass *cl, Int_t n = 0);

      virtual TBuffer  &operator>>(Bool_t    &b);
      virtual TBuffer  &operator>>(Char_t    &c);
      virtual TBuffer  &operator>>(UChar_t   &c);
      virtual TBuffer  &operator>>(Short_t   &h);
      virtual TBuffer  &operator>>(UShort_t  &h);
      virtual TBuffer  &operator>>(Int_t     &i);
      virtual TBuffer  &operator>>(UInt_t    &i);
      virtual TBuffer  &operator>>(Long_t    &l);
      virtual TBuffer  &operator>>(ULong_t   &l);
      virtual TBuffer  &operator>>(Long64_t  &l);
      virtual TBuffer  &operator>>(ULong64_t &l);
      virtual TBuffer  &operator>>(Float_t   &f);
      virtual TBuffer  &operator>>(Double_t  &d);
      virtual TBuffer  &operator>>(Char_t    *c);

      virtual TBuffer  &operator<<(Bool_t    b);
      virtual TBuffer  &operator<<(Char_t    c);
      virtual TBuffer  &operator<<(UChar_t   c);
      virtual TBuffer  &operator<<(Short_t   h);
      virtual TBuffer  &operator<<(UShort_t  h);
      virtual TBuffer  &operator<<(Int_t     i);
      virtual TBuffer  &operator<<(UInt_t    i);
      virtual TBuffer  &operator<<(Long_t    l);
      virtual TBuffer  &operator<<(ULong_t   l);
      virtual TBuffer  &operator<<(Long64_t  l);
      virtual TBuffer  &operator<<(ULong64_t l);
      virtual TBuffer  &operator<<(Float_t   f);
      virtual TBuffer  &operator<<(Double_t  d);
      virtual TBuffer  &operator<<(const Char_t *c);

      // end of redefined virtual functions

   ClassDef(TBufferSQL2,1);    //a specialized TBuffer to convert data to SQL statements or read data from SQL tables
};

#endif
