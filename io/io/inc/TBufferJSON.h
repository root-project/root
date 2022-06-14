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

#include "TBufferText.h"
#include "TString.h"

#include <deque>
#include <memory>
#include <string>
#include <vector>

class TVirtualStreamerInfo;
class TStreamerInfo;
class TStreamerElement;
class TMemberStreamer;
class TDataMember;
class TJSONStackObj;

class TBufferJSON final : public TBufferText {

public:

   enum {
     // values 0..3 are exclusive, define text formating, JSON data are same
     kNoCompress    = 0,             ///< no any compression, maximal size of JSON (default)
     kNoIndent      = 1,             ///< remove spaces in the beginning showing JSON indentation level
     kNoNewLine     = 2,             ///< no indent plus skip newline symbols
     kNoSpaces      = 3,             ///< no new lines plus remove all spaces around "," and ":" symbols

     kMapAsObject   = 5,             ///< store std::map, std::unordered_map as JSON object

     // algorithms for array compression - exclusive
     kZeroSuppression = 10,          ///< if array has much zeros in begin and/or end, they will be removed
     kSameSuppression = 20,          ///< zero suppression plus compress many similar values together
     kBase64          = 30,          ///< all binary arrays will be compressed with base64 coding, supported by JSROOT

     kSkipTypeInfo  = 100            ///< do not store typenames in JSON
   };

   TBufferJSON(TBuffer::EMode mode = TBuffer::kWrite);
   virtual ~TBufferJSON();

   void SetCompact(int level);
   void SetTypenameTag(const char *tag = "_typename");
   void SetTypeversionTag(const char *tag = nullptr);
   void SetSkipClassInfo(const TClass *cl);
   Bool_t IsSkipClassInfo(const TClass *cl) const;

   TString StoreObject(const void *obj, const TClass *cl);
   void *RestoreObject(const char *str, TClass **cl);

   static TString ConvertToJSON(const TObject *obj, Int_t compact = 0, const char *member_name = nullptr);
   static TString
   ConvertToJSON(const void *obj, const TClass *cl, Int_t compact = 0, const char *member_name = nullptr);
   static TString ConvertToJSON(const void *obj, TDataMember *member, Int_t compact = 0, Int_t arraylen = -1);

   static Int_t ExportToFile(const char *filename, const TObject *obj, const char *option = nullptr);
   static Int_t ExportToFile(const char *filename, const void *obj, const TClass *cl, const char *option = nullptr);

   static TObject *ConvertFromJSON(const char *str);
   static void *ConvertFromJSONAny(const char *str, TClass **cl = nullptr);

   template <class T>
   static TString ToJSON(const T *obj, Int_t compact = 0, const char *member_name = nullptr)
   {
      return ConvertToJSON(obj, TClass::GetClass<T>(), compact, member_name);
   }

   template <class T>
   static Bool_t FromJSON(T *&obj, const char *json)
   {
      if (obj)
         return kFALSE;
      obj = (T *)ConvertFromJSONChecked(json, TClass::GetClass<T>());
      return obj != nullptr;
   }

   template <class T>
   static std::unique_ptr<T> FromJSON(const std::string &json)
   {
      T *obj = (T *)ConvertFromJSONChecked(json.c_str(), TClass::GetClass<T>());
      return std::unique_ptr<T>(obj);
   }

   // suppress class writing/reading

   TClass *ReadClass(const TClass *cl = nullptr, UInt_t *objTag = nullptr) final;
   void WriteClass(const TClass *cl) final;

   // redefined virtual functions of TBuffer

   Version_t ReadVersion(UInt_t *start = nullptr, UInt_t *bcnt = nullptr, const TClass *cl = nullptr) final;
   UInt_t WriteVersion(const TClass *cl, Bool_t useBcnt = kFALSE) final;

   void *ReadObjectAny(const TClass *clCast) final;
   void SkipObjectAny() final;

   // these methods used in streamer info to indicate currently streamed element,
   void IncrementLevel(TVirtualStreamerInfo *) final;
   void SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type) final;
   void DecrementLevel(TVirtualStreamerInfo *) final;

   void ClassBegin(const TClass *, Version_t = -1) final;
   void ClassEnd(const TClass *) final;
   void ClassMember(const char *name, const char *typeName = nullptr, Int_t arrsize1 = -1, Int_t arrsize2 = -1) final;

   Int_t ReadArray(Bool_t *&b) final;
   Int_t ReadArray(Char_t *&c) final;
   Int_t ReadArray(UChar_t *&c) final;
   Int_t ReadArray(Short_t *&h) final;
   Int_t ReadArray(UShort_t *&h) final;
   Int_t ReadArray(Int_t *&i) final;
   Int_t ReadArray(UInt_t *&i) final;
   Int_t ReadArray(Long_t *&l) final;
   Int_t ReadArray(ULong_t *&l) final;
   Int_t ReadArray(Long64_t *&l) final;
   Int_t ReadArray(ULong64_t *&l) final;
   Int_t ReadArray(Float_t *&f) final;
   Int_t ReadArray(Double_t *&d) final;

   Int_t ReadStaticArray(Bool_t *b) final;
   Int_t ReadStaticArray(Char_t *c) final;
   Int_t ReadStaticArray(UChar_t *c) final;
   Int_t ReadStaticArray(Short_t *h) final;
   Int_t ReadStaticArray(UShort_t *h) final;
   Int_t ReadStaticArray(Int_t *i) final;
   Int_t ReadStaticArray(UInt_t *i) final;
   Int_t ReadStaticArray(Long_t *l) final;
   Int_t ReadStaticArray(ULong_t *l) final;
   Int_t ReadStaticArray(Long64_t *l) final;
   Int_t ReadStaticArray(ULong64_t *l) final;
   Int_t ReadStaticArray(Float_t *f) final;
   Int_t ReadStaticArray(Double_t *d) final;

   void ReadFastArray(Bool_t *b, Int_t n) final;
   void ReadFastArray(Char_t *c, Int_t n) final;
   void ReadFastArrayString(Char_t *c, Int_t n) final;
   void ReadFastArray(UChar_t *c, Int_t n) final;
   void ReadFastArray(Short_t *h, Int_t n) final;
   void ReadFastArray(UShort_t *h, Int_t n) final;
   void ReadFastArray(Int_t *i, Int_t n) final;
   void ReadFastArray(UInt_t *i, Int_t n) final;
   void ReadFastArray(Long_t *l, Int_t n) final;
   void ReadFastArray(ULong_t *l, Int_t n) final;
   void ReadFastArray(Long64_t *l, Int_t n) final;
   void ReadFastArray(ULong64_t *l, Int_t n) final;
   void ReadFastArray(Float_t *f, Int_t n) final;
   void ReadFastArray(Double_t *d, Int_t n) final;
   void ReadFastArray(void *start, const TClass *cl, Int_t n = 1, TMemberStreamer *s = nullptr,
                      const TClass *onFileClass = nullptr) final;
   void ReadFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE,
                      TMemberStreamer *s = nullptr, const TClass *onFileClass = nullptr) final;

   void WriteArray(const Bool_t *b, Int_t n) final;
   void WriteArray(const Char_t *c, Int_t n) final;
   void WriteArray(const UChar_t *c, Int_t n) final;
   void WriteArray(const Short_t *h, Int_t n) final;
   void WriteArray(const UShort_t *h, Int_t n) final;
   void WriteArray(const Int_t *i, Int_t n) final;
   void WriteArray(const UInt_t *i, Int_t n) final;
   void WriteArray(const Long_t *l, Int_t n) final;
   void WriteArray(const ULong_t *l, Int_t n) final;
   void WriteArray(const Long64_t *l, Int_t n) final;
   void WriteArray(const ULong64_t *l, Int_t n) final;
   void WriteArray(const Float_t *f, Int_t n) final;
   void WriteArray(const Double_t *d, Int_t n) final;

   void WriteFastArray(const Bool_t *b, Int_t n) final;
   void WriteFastArray(const Char_t *c, Int_t n) final;
   void WriteFastArrayString(const Char_t *c, Int_t n) final;
   void WriteFastArray(const UChar_t *c, Int_t n) final;
   void WriteFastArray(const Short_t *h, Int_t n) final;
   void WriteFastArray(const UShort_t *h, Int_t n) final;
   void WriteFastArray(const Int_t *i, Int_t n) final;
   void WriteFastArray(const UInt_t *i, Int_t n) final;
   void WriteFastArray(const Long_t *l, Int_t n) final;
   void WriteFastArray(const ULong_t *l, Int_t n) final;
   void WriteFastArray(const Long64_t *l, Int_t n) final;
   void WriteFastArray(const ULong64_t *l, Int_t n) final;
   void WriteFastArray(const Float_t *f, Int_t n) final;
   void WriteFastArray(const Double_t *d, Int_t n) final;
   void WriteFastArray(void *start, const TClass *cl, Int_t n = 1, TMemberStreamer *s = nullptr) final;
   Int_t WriteFastArray(void **startp, const TClass *cl, Int_t n = 1, Bool_t isPreAlloc = kFALSE,
                        TMemberStreamer *s = nullptr) final;

   void StreamObject(void *obj, const TClass *cl, const TClass *onFileClass = nullptr) final;
   using TBufferText::StreamObject;

   void ReadBool(Bool_t &b) final;
   void ReadChar(Char_t &c) final;
   void ReadUChar(UChar_t &c) final;
   void ReadShort(Short_t &s) final;
   void ReadUShort(UShort_t &s) final;
   void ReadInt(Int_t &i) final;
   void ReadUInt(UInt_t &i) final;
   void ReadLong(Long_t &l) final;
   void ReadULong(ULong_t &l) final;
   void ReadLong64(Long64_t &l) final;
   void ReadULong64(ULong64_t &l) final;
   void ReadFloat(Float_t &f) final;
   void ReadDouble(Double_t &d) final;
   void ReadCharP(Char_t *c) final;
   void ReadTString(TString &s) final;
   void ReadStdString(std::string *s) final;
   using TBuffer::ReadStdString;
   void ReadCharStar(char *&s) final;

   void WriteBool(Bool_t b) final;
   void WriteChar(Char_t c) final;
   void WriteUChar(UChar_t c) final;
   void WriteShort(Short_t s) final;
   void WriteUShort(UShort_t s) final;
   void WriteInt(Int_t i) final;
   void WriteUInt(UInt_t i) final;
   void WriteLong(Long_t l) final;
   void WriteULong(ULong_t l) final;
   void WriteLong64(Long64_t l) final;
   void WriteULong64(ULong64_t l) final;
   void WriteFloat(Float_t f) final;
   void WriteDouble(Double_t d) final;
   void WriteCharP(const Char_t *c) final;
   void WriteTString(const TString &s) final;
   void WriteStdString(const std::string *s) final;
   using TBuffer::WriteStdString;
   void WriteCharStar(char *s) final;

   TVirtualStreamerInfo *GetInfo() final;

   // end of redefined virtual functions

   void ReadBaseClass(void *start, TStreamerBase *elem) final;

protected:
   // redefined protected virtual functions

   void WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse) final;

   // end redefined protected virtual functions

   static void *ConvertFromJSONChecked(const char *str, const TClass *expectedClass);

   TString JsonWriteMember(const void *ptr, TDataMember *member, TClass *memberClass, Int_t arraylen);

   TJSONStackObj *PushStack(Int_t inclevel = 0, void *readnode = nullptr);
   TJSONStackObj *PopStack();
   TJSONStackObj *Stack() { return fStack.back().get(); }

   void WorkWithClass(TStreamerInfo *info, const TClass *cl = nullptr);
   void WorkWithElement(TStreamerElement *elem, Int_t);

   void JsonDisablePostprocessing();
   Int_t JsonSpecialClass(const TClass *cl) const;

   TJSONStackObj *JsonStartObjectWrite(const TClass *obj_class, TStreamerInfo *info = nullptr);

   void JsonStartElement(const TStreamerElement *elem, const TClass *base_class);

   void PerformPostProcessing(TJSONStackObj *stack, const TClass *obj_cl = nullptr);

   void JsonWriteBasic(Char_t value);
   void JsonWriteBasic(Short_t value);
   void JsonWriteBasic(Int_t value);
   void JsonWriteBasic(Long_t value);
   void JsonWriteBasic(Long64_t value);
   void JsonWriteBasic(Float_t value);
   void JsonWriteBasic(Double_t value);
   void JsonWriteBasic(Bool_t value);
   void JsonWriteBasic(UChar_t value);
   void JsonWriteBasic(UShort_t value);
   void JsonWriteBasic(UInt_t value);
   void JsonWriteBasic(ULong_t value);
   void JsonWriteBasic(ULong64_t value);

   void JsonWriteConstChar(const char *value, Int_t len = -1, const char * /*typname*/ = nullptr);

   void JsonWriteObject(const void *obj, const TClass *objClass, Bool_t check_map = kTRUE);

   void JsonWriteCollection(TCollection *obj, const TClass *objClass);

   void JsonReadCollection(TCollection *obj, const TClass *objClass);

   void JsonReadTObjectMembers(TObject *obj, void *node = nullptr);

   void *JsonReadObject(void *obj, const TClass *objClass = nullptr, TClass **readClass = nullptr);

   void AppendOutput(const char *line0, const char *line1 = nullptr);

   void JsonPushValue();

   template <typename T>
   R__ALWAYS_INLINE void JsonWriteArrayCompress(const T *vname, Int_t arrsize, const char *typname);

   template <typename T>
   R__ALWAYS_INLINE void JsonReadBasic(T &value);

   template <typename T>
   R__ALWAYS_INLINE Int_t JsonReadArray(T *value);

   template <typename T>
   R__ALWAYS_INLINE void JsonReadFastArray(T *arr, Int_t arrsize, bool asstring = false);

   template <typename T>
   R__ALWAYS_INLINE void JsonWriteFastArray(const T *arr, Int_t arrsize, const char *typname,
                                            void (TBufferJSON::*method)(const T *, Int_t, const char *));

   TString fOutBuffer;                 ///<!  main output buffer for json code
   TString *fOutput{nullptr};          ///<!  current output buffer for json code
   TString fValue;                     ///<!  buffer for current value
   unsigned fJsonrCnt{0};              ///<!  counter for all objects, used for referencing
   std::deque<std::unique_ptr<TJSONStackObj>> fStack; ///<!  hierarchy of currently streamed element
   Int_t fCompact{0};                  ///<!  0 - no any compression, 1 - no spaces in the begin, 2 - no new lines, 3 - no spaces at all
   Bool_t fMapAsObject{kFALSE};        ///<! when true, std::map will be converted into JSON object
   TString fSemicolon;                 ///<!  depending from compression level, " : " or ":"
   Int_t fArrayCompact{0};             ///<!  0 - no array compression, 1 - exclude leading/trailing zeros, 2 - check value repetition
   TString fArraySepar;                ///<!  depending from compression level, ", " or ","
   TString fNumericLocale;             ///<!  stored value of setlocale(LC_NUMERIC), which should be recovered at the end
   TString fTypeNameTag;               ///<! JSON member used for storing class name, when empty - no class name will be stored
   TString fTypeVersionTag;            ///<! JSON member used to store class version, default empty
   std::vector<const TClass *> fSkipClasses; ///<! list of classes, which class info is not stored

   ClassDefOverride(TBufferJSON, 0) // a specialized TBuffer to only write objects into JSON format
};

#endif
