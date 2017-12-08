//
// Author: Sergey Linev  4.03.2014

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TBufferJSON
\ingroup IO

Class for serializing object into JavaScript Object Notation (JSON) format.
It creates such object representation, which can be directly
used in JavaScript ROOT (JSROOT) for drawing.

TBufferJSON implements TBuffer interface, therefore most of
ROOT and user classes can be converted into JSON.
There are certain limitations for classes with custom streamers,
which should be equipped specially for this purposes (see TCanvas::Streamer()
as example).

To perform conversion, one should use TBufferJSON::ConvertToJSON method like:
~~~{.cpp}
   TH1* h1 = new TH1I("h1","title",100, 0, 10);
   h1->FillRandom("gaus",10000);
   TString json = TBufferJSON::ConvertToJSON(h1);
~~~
*/

#include "TBufferJSON.h"

#include <typeinfo>
#include <string>
#include <string.h>
#include <locale.h>

#include "Compression.h"

#include "TArrayI.h"
#include "TObjArray.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TClassEdit.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TDataMember.h"
#include "TMap.h"
#include "TExMap.h"
#include "TMethodCall.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TMemberStreamer.h"
#include "TStreamer.h"
#include "TStreamerInfoActions.h"
#include "RVersion.h"
#include "Riostream.h"
#include "RZip.h"
#include "TClonesArray.h"
#include "TVirtualMutex.h"
#include "TInterpreter.h"

#ifdef R__VISUAL_CPLUSPLUS
#define FLong64 "%I64d"
#define FULong64 "%I64u"
#else
#define FLong64 "%lld"
#define FULong64 "%llu"
#endif

#include "json.hpp"

ClassImp(TBufferJSON);

const char *TBufferJSON::fgFloatFmt = "%e";
const char *TBufferJSON::fgDoubleFmt = "%.14e";

enum { json_TArray = 100, json_TCollection = -130, json_TString = 110, json_stdstring = 120 };

///////////////////////////////////////////////////////////////
// TArrayIndexProducer is used to correctly create
/// JSON array separators for multi-dimensional JSON arrays
/// It fully reproduces array dimensions as in original ROOT classes
/// Contrary to binary I/O, which always writes flat arrays

class TArrayIndexProducer {
protected:
   Int_t fTotalLen;
   Int_t fCnt;
   const char *fSepar;
   TArrayI fIndicies;
   TArrayI fMaxIndex;
   TString fRes;
   Bool_t fIsArray;

public:
   TArrayIndexProducer(TStreamerElement *elem, Int_t arraylen, const char *separ)
      : fTotalLen(0), fCnt(-1), fSepar(separ), fIndicies(), fMaxIndex(), fRes(), fIsArray(kFALSE)
   {
      Bool_t usearrayindx = elem && (elem->GetArrayDim() > 0);
      Bool_t isloop = elem && ((elem->GetType() == TStreamerInfo::kStreamLoop) ||
                               (elem->GetType() == TStreamerInfo::kOffsetL + TStreamerInfo::kStreamLoop));
      Bool_t usearraylen = (arraylen > (isloop ? 0 : 1));

      if (usearrayindx && (arraylen > 0)) {
         if (isloop) {
            usearrayindx = kFALSE;
            usearraylen = kTRUE;
         } else if (arraylen != elem->GetArrayLength()) {
            printf("Problem with JSON coding of element %s type %d \n", elem->GetName(), elem->GetType());
         }
      }

      if (usearrayindx) {
         fTotalLen = elem->GetArrayLength();
         fMaxIndex.Set(elem->GetArrayDim());
         for (int dim = 0; dim < elem->GetArrayDim(); dim++)
            fMaxIndex[dim] = elem->GetMaxIndex(dim);
         fIsArray = fTotalLen > 1;
      } else if (usearraylen) {
         fTotalLen = arraylen;
         fMaxIndex.Set(1);
         fMaxIndex[0] = arraylen;
         fIsArray = kTRUE;
      }

      if (fMaxIndex.GetSize() > 0) {
         fIndicies.Set(fMaxIndex.GetSize());
         fIndicies.Reset(0);
      }
   }

   TArrayIndexProducer(TDataMember *member, Int_t extradim, const char *separ)
      : fTotalLen(0), fCnt(-1), fSepar(separ), fIndicies(), fMaxIndex(), fRes(), fIsArray(kFALSE)
   {
      Int_t ndim = member->GetArrayDim();
      if (extradim > 0)
         ndim++;

      if (ndim > 0) {
         fIndicies.Set(ndim);
         fIndicies.Reset(0);
         fMaxIndex.Set(ndim);
         fTotalLen = 1;
         for (int dim = 0; dim < member->GetArrayDim(); dim++) {
            fMaxIndex[dim] = member->GetMaxIndex(dim);
            fTotalLen *= member->GetMaxIndex(dim);
         }

         if (extradim > 0) {
            fMaxIndex[ndim - 1] = extradim;
            fTotalLen *= extradim;
         }
      }
      fIsArray = fTotalLen > 1;
   }

   /// returns number of array dimensions
   Int_t NumDimensions() const  { return fIndicies.GetSize(); }

   /// return array with current index
   TArrayI &GetIndices() { return fIndicies; };

   /// returns total number of elements in array
   Int_t TotalLength() const { return fTotalLen; }

   Int_t ReduceDimension()
   {
      // reduce one dimension of the array
      // return size of reduced dimension
      if (fMaxIndex.GetSize() == 0)
         return 0;
      Int_t ndim = fMaxIndex.GetSize() - 1;
      Int_t len = fMaxIndex[ndim];
      fMaxIndex.Set(ndim);
      fIndicies.Set(ndim);
      fTotalLen = fTotalLen / len;
      fIsArray = fTotalLen > 1;
      return len;
   }

   Bool_t IsArray() const { return fIsArray; }

   Bool_t IsDone() const
   {
      // return true when iteration over all arrays indexes are done
      return !IsArray() || (fCnt >= fTotalLen);
   }

   const char *GetBegin()
   {
      ++fCnt;
      // return starting separator
      fRes.Clear();
      for (Int_t n = 0; n < fIndicies.GetSize(); ++n)
         fRes.Append("[");
      return fRes.Data();
   }

   const char *GetEnd()
   {
      // return ending separator
      fRes.Clear();
      for (Int_t n = 0; n < fIndicies.GetSize(); ++n)
         fRes.Append("]");
      return fRes.Data();
   }

   /// increment indexes and returns intermediate or last separator
   const char *NextSeparator()
   {
      if (++fCnt >= fTotalLen)
         return GetEnd();

      Int_t cnt = fIndicies.GetSize() - 1;
      fIndicies[cnt]++;

      fRes.Clear();

      while ((cnt >= 0) && (cnt < fIndicies.GetSize())) {
         if (fIndicies[cnt] >= fMaxIndex[cnt]) {
            fRes.Append("]");
            fIndicies[cnt--] = 0;
            if (cnt >= 0)
               fIndicies[cnt]++;
            continue;
         }
         fRes.Append(fIndicies[cnt] == 0 ? "[" : fSepar);
         cnt++;
      }
      return fRes.Data();
   }

   JSONObject_t ExtractNode(JSONObject_t topnode, bool next = true)
   {
      if (!IsArray()) return topnode;
      nlohmann::json *subnode = &((*((nlohmann::json *) topnode))[fIndicies[0]]);
      for (int k=1;k<fIndicies.GetSize();++k)
         subnode = &((*subnode)[fIndicies[k]]);
      if (next) NextSeparator();
      return subnode;
   }

};

// TJSONStackObj is used to keep stack of object hierarchy,
// stored in TBuffer. For instance, data for parent class(es)
// stored in subnodes, but initial object node will be kept.

class TJSONStackObj : public TObject {
public:
   TStreamerInfo *fInfo;       //!
   TStreamerElement *fElem;    //! element in streamer info
   Bool_t fIsStreamerInfo;     //!
   Bool_t fIsElemOwner;        //!
   Bool_t fIsPostProcessed;    //! indicate that value is written
   Bool_t fIsObjStarted;       //! indicate that object writing started, should be closed in postprocess
   Bool_t fAccObjects;         //! if true, accumulate whole objects in values
   TObjArray fValues;          //! raw values
   Int_t fLevel;               //! indent level
   TArrayIndexProducer *fIndx; //! producer of ndim indexes

   JSONObject_t fNode;            //! reading JSON node
   Int_t                fStlIndx;  //! index of object in STL container
   Version_t            fClVersion; //! keep actual class version, workaround for ReadVersion in custom streamer

   TJSONStackObj()
      : TObject(), fInfo(nullptr), fElem(nullptr), fIsStreamerInfo(kFALSE), fIsElemOwner(kFALSE),
        fIsPostProcessed(kFALSE), fIsObjStarted(kFALSE), fAccObjects(kFALSE), fValues(), fLevel(0), fIndx(nullptr),
        fNode(nullptr), fStlIndx(-1), fClVersion(0)
   {
      fValues.SetOwner(kTRUE);
   }

   virtual ~TJSONStackObj()
   {
      if (fIsElemOwner)
         delete fElem;
      if (fIndx)
         delete fIndx;
   }

   Bool_t IsStreamerInfo() const { return fIsStreamerInfo; }

   Bool_t IsStreamerElement() const { return !fIsStreamerInfo && (fElem != 0); }

   void PushValue(TString &v)
   {
      fValues.Add(new TObjString(v));
      v.Clear();
   }

   void PushIntValue(Int_t v)
   {
      fValues.Add(new TObjString(TString::Itoa(v,10)));
   }

   TArrayIndexProducer *MakeReadIndexes()
   {
      if (!fElem || (fElem->GetType() <= TStreamerInfo::kOffsetL) || (fElem->GetType() >= TStreamerInfo::kOffsetL + 20) ||
          (fElem->GetArrayDim() < 2))
         return nullptr;

      TArrayIndexProducer *indx = new TArrayIndexProducer(fElem, -1, "");

      if (!indx->IsArray() || (indx->NumDimensions() < 2)) {
         delete indx; // no need for single dimension - it can be handled directly
         return nullptr;
      }

      return indx;
   }

};

////////////////////////////////////////////////////////////////////////////////
/// Creates buffer object to serialize data into json.

TBufferJSON::TBufferJSON(TBuffer::EMode mode)
   : TBuffer(mode), fOutBuffer(), fOutput(0), fValue(), fJsonrMap(), fReadMap(), fJsonrCnt(0), fStack(), fCompact(0),
     fSemicolon(" : "), fArraySepar(", "), fNumericLocale()
{
   fBufSize = 1000000000;

   SetParent(0);
   SetBit(kCannotHandleMemberWiseStreaming);
   // SetBit(kTextBasedStreaming);

   fOutBuffer.Capacity(10000);
   fValue.Capacity(1000);
   fOutput = &fOutBuffer;

   // checks if setlocale(LC_NUMERIC) returns others than "C"
   // in this case locale will be changed and restored at the end of object conversion

   char *loc = setlocale(LC_NUMERIC, 0);
   if ((loc != 0) && (strcmp(loc, "C") != 0)) {
      fNumericLocale = loc;
      setlocale(LC_NUMERIC, "C");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// destroy buffer

TBufferJSON::~TBufferJSON()
{
   fStack.Delete();

   if (fNumericLocale.Length() > 0)
      setlocale(LC_NUMERIC, fNumericLocale.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Converts object, inherited from TObject class, to JSON string
/// Lower digit of compact parameter define formatting rules
///  - 0 - no any compression, human-readable form
///  - 1 - exclude spaces in the begin
///  - 2 - remove newlines
///  - 3 - exclude spaces as much as possible
///
/// Second digit of compact parameter defines algorithm for arrays compression
///  - 0 - no compression, standard JSON array
///  - 1 - exclude leading, trailing zeros, required JSROOT v5
///  - 2 - check values repetition and empty gaps, required JSROOT v5
///
/// Maximal compression achieved when compact parameter equal to 23
/// When member_name specified, converts only this data member

TString TBufferJSON::ConvertToJSON(const TObject *obj, Int_t compact, const char *member_name)
{
   TClass *clActual = 0;
   void *ptr = (void *)obj;

   if (obj != 0) {
      clActual = TObject::Class()->GetActualClass(obj);
      if (!clActual)
         clActual = TObject::Class();
      else if (clActual != TObject::Class())
         ptr = (void *)((Long_t)obj - clActual->GetBaseClassOffset(TObject::Class()));
   }

   return ConvertToJSON(ptr, clActual, compact, member_name);
}

////////////////////////////////////////////////////////////////////////////////
// Set level of space/newline/array compression
// Lower digit of compact parameter define formatting rules
//  - 0 - no any compression, human-readable form
//  - 1 - exclude spaces in the begin
//  - 2 - remove newlines
//  - 3 - exclude spaces as much as possible
//
// Second digit of compact parameter defines algorithm for arrays compression
//  - 0 - no compression, standard JSON array
//  - 1 - exclude leading, trailing zeros, required JSROOT v5
//  - 2 - check values repetition and empty gaps, required JSROOT v5

void TBufferJSON::SetCompact(int level)
{
   fCompact = level;
   fSemicolon = (fCompact % 10 > 2) ? ":" : " : ";
   fArraySepar = (fCompact % 10 > 2) ? "," : ", ";
}

////////////////////////////////////////////////////////////////////////////////
/// Converts any type of object to JSON string
/// One should provide pointer on object and its class name
/// Lower digit of compact parameter define formatting rules
///  - 0 - no any compression, human-readable form
///  - 1 - exclude spaces in the begin
///  - 2 - remove newlines
///  - 3 - exclude spaces as much as possible
///
/// Second digit of compact parameter defines algorithm for arrays compression
///  - 0 - no compression, standard JSON array
///  - 1 - exclude leading, trailing zeros, required JSROOT v5
///  - 2 - check values repetition and empty gaps, required JSROOT v5
///
/// Maximal compression achieved when compact parameter equal to 23
/// When member_name specified, converts only this data member

TString TBufferJSON::ConvertToJSON(const void *obj, const TClass *cl, Int_t compact, const char *member_name)
{
   if ((member_name != 0) && (obj != 0)) {
      TRealData *rdata = cl->GetRealData(member_name);
      TDataMember *member = rdata ? rdata->GetDataMember() : 0;
      if (member == 0) {
         TIter iter(cl->GetListOfRealData());
         while ((rdata = dynamic_cast<TRealData *>(iter())) != 0) {
            member = rdata->GetDataMember();
            if (member && strcmp(member->GetName(), member_name) == 0)
               break;
         }
      }
      if (member == 0)
         return TString();

      Int_t arraylen = -1;
      if (member->GetArrayIndex() != 0) {
         TRealData *idata = cl->GetRealData(member->GetArrayIndex());
         TDataMember *imember = (idata != 0) ? idata->GetDataMember() : 0;
         if ((imember != 0) && (strcmp(imember->GetTrueTypeName(), "int") == 0)) {
            arraylen = *((int *)((char *)obj + idata->GetThisOffset()));
         }
      }

      void *ptr = (char *)obj + rdata->GetThisOffset();
      if (member->IsaPointer())
         ptr = *((char **)ptr);

      return TBufferJSON::ConvertToJSON(ptr, member, compact, arraylen);
   }

   TBufferJSON buf;

   buf.SetCompact(compact);

   buf.JsonWriteObject(obj, cl);

   return buf.fOutBuffer.Length() ? buf.fOutBuffer : buf.fValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Converts selected data member into json
/// Parameter ptr specifies address in memory, where data member is located
/// compact parameter defines compactness of produced JSON (from 0 to 3)
/// arraylen (when specified) is array length for this data member,  //[fN] case

TString TBufferJSON::ConvertToJSON(const void *ptr, TDataMember *member, Int_t compact, Int_t arraylen)
{
   if ((ptr == 0) || (member == 0))
      return TString("null");

   Bool_t stlstring = !strcmp(member->GetTrueTypeName(), "string");

   Int_t isstl = member->IsSTLContainer();

   TClass *mcl = member->IsBasic() ? 0 : gROOT->GetClass(member->GetTypeName());

   if ((mcl != 0) && (mcl != TString::Class()) && !stlstring && !isstl &&
       (mcl->GetBaseClassOffset(TArray::Class()) != 0) && (arraylen <= 0) && (member->GetArrayDim() == 0))
      return TBufferJSON::ConvertToJSON(ptr, mcl, compact);

   TBufferJSON buf;

   buf.SetCompact(compact);

   return buf.JsonWriteMember(ptr, member, mcl, arraylen);
}

////////////////////////////////////////////////////////////////////////////////
/// Convert object into JSON and store in text file
/// Returns size of the produce file
/// Used in TObject::SaveAs()

Int_t TBufferJSON::ExportToFile(const char *filename, const TObject *obj, const char *option)
{
   if (!obj || !filename || (*filename == 0))
      return 0;

   Int_t compact = strstr(filename, ".json.gz") ? 3 : 0;
   if (option && (*option >= '0') && (*option <= '3'))
      compact = TString(option).Atoi();

   TString json = TBufferJSON::ConvertToJSON(obj, compact);

   std::ofstream ofs(filename);

   if (strstr(filename, ".json.gz")) {
      const char *objbuf = json.Data();
      Long_t objlen = json.Length();

      unsigned long objcrc = R__crc32(0, NULL, 0);
      objcrc = R__crc32(objcrc, (const unsigned char *)objbuf, objlen);

      // 10 bytes (ZIP header), compressed data, 8 bytes (CRC and original length)
      Int_t buflen = 10 + objlen + 8;
      if (buflen < 512)
         buflen = 512;

      char *buffer = (char *)malloc(buflen);
      if (buffer == 0)
         return 0; // failure

      char *bufcur = buffer;

      *bufcur++ = 0x1f; // first byte of ZIP identifier
      *bufcur++ = 0x8b; // second byte of ZIP identifier
      *bufcur++ = 0x08; // compression method
      *bufcur++ = 0x00; // FLAG - empty, no any file names
      *bufcur++ = 0;    // empty timestamp
      *bufcur++ = 0;    //
      *bufcur++ = 0;    //
      *bufcur++ = 0;    //
      *bufcur++ = 0;    // XFL (eXtra FLags)
      *bufcur++ = 3;    // OS   3 means Unix
      // strcpy(bufcur, "item.json");
      // bufcur += strlen("item.json")+1;

      char dummy[8];
      memcpy(dummy, bufcur - 6, 6);

      // R__memcompress fills first 6 bytes with own header, therefore just overwrite them
      unsigned long ziplen = R__memcompress(bufcur - 6, objlen + 6, (char *)objbuf, objlen);

      memcpy(bufcur - 6, dummy, 6);

      bufcur += (ziplen - 6); // jump over compressed data (6 byte is extra ROOT header)

      *bufcur++ = objcrc & 0xff; // CRC32
      *bufcur++ = (objcrc >> 8) & 0xff;
      *bufcur++ = (objcrc >> 16) & 0xff;
      *bufcur++ = (objcrc >> 24) & 0xff;

      *bufcur++ = objlen & 0xff;         // original data length
      *bufcur++ = (objlen >> 8) & 0xff;  // original data length
      *bufcur++ = (objlen >> 16) & 0xff; // original data length
      *bufcur++ = (objlen >> 24) & 0xff; // original data length

      ofs.write(buffer, bufcur - buffer);

      free(buffer);
   } else {
      ofs << json.Data();
   }

   ofs.close();

   return json.Length();
}

////////////////////////////////////////////////////////////////////////////////
/// Convert object into JSON and store in text file
/// Returns size of the produce file

Int_t TBufferJSON::ExportToFile(const char *filename, const void *obj, const TClass *cl, const char *option)
{
   if (!obj || !cl || !filename || (*filename == 0))
      return 0;

   Int_t compact = strstr(filename, ".json.gz") ? 3 : 0;
   if (option && (*option >= '0') && (*option <= '3'))
      compact = TString(option).Atoi();

   TString json = TBufferJSON::ConvertToJSON(obj, cl, compact);

   std::ofstream ofs(filename);

   if (strstr(filename, ".json.gz")) {
      const char *objbuf = json.Data();
      Long_t objlen = json.Length();

      unsigned long objcrc = R__crc32(0, NULL, 0);
      objcrc = R__crc32(objcrc, (const unsigned char *)objbuf, objlen);

      // 10 bytes (ZIP header), compressed data, 8 bytes (CRC and original length)
      Int_t buflen = 10 + objlen + 8;
      if (buflen < 512)
         buflen = 512;

      char *buffer = (char *)malloc(buflen);
      if (buffer == 0)
         return 0; // failure

      char *bufcur = buffer;

      *bufcur++ = 0x1f; // first byte of ZIP identifier
      *bufcur++ = 0x8b; // second byte of ZIP identifier
      *bufcur++ = 0x08; // compression method
      *bufcur++ = 0x00; // FLAG - empty, no any file names
      *bufcur++ = 0;    // empty timestamp
      *bufcur++ = 0;    //
      *bufcur++ = 0;    //
      *bufcur++ = 0;    //
      *bufcur++ = 0;    // XFL (eXtra FLags)
      *bufcur++ = 3;    // OS   3 means Unix
      // strcpy(bufcur, "item.json");
      // bufcur += strlen("item.json")+1;

      char dummy[8];
      memcpy(dummy, bufcur - 6, 6);

      // R__memcompress fills first 6 bytes with own header, therefore just overwrite them
      unsigned long ziplen = R__memcompress(bufcur - 6, objlen + 6, (char *)objbuf, objlen);

      memcpy(bufcur - 6, dummy, 6);

      bufcur += (ziplen - 6); // jump over compressed data (6 byte is extra ROOT header)

      *bufcur++ = objcrc & 0xff; // CRC32
      *bufcur++ = (objcrc >> 8) & 0xff;
      *bufcur++ = (objcrc >> 16) & 0xff;
      *bufcur++ = (objcrc >> 24) & 0xff;

      *bufcur++ = objlen & 0xff;         // original data length
      *bufcur++ = (objlen >> 8) & 0xff;  // original data length
      *bufcur++ = (objlen >> 16) & 0xff; // original data length
      *bufcur++ = (objlen >> 24) & 0xff; // original data length

      ofs.write(buffer, bufcur - buffer);

      free(buffer);
   } else {
      ofs << json.Data();
   }

   ofs.close();

   return json.Length();
}

////////////////////////////////////////////////////////////////////////////////
/// Read TObject-based class from JSON, produced by ConvertToJSON() method.
/// If object does not inherit from TObject class, return 0.

TObject *TBufferJSON::ConvertFromJSON(const char *str)
{
   TClass *cl = nullptr;
   void *obj = ConvertFromJSONAny(str, &cl);

   if (!cl || !obj)
      return nullptr;

   Int_t delta = cl->GetBaseClassOffset(TObject::Class());

   if (delta < 0) {
      cl->Destructor(obj);
      return nullptr;
   }

   return (TObject *)(((char *)obj) + delta);
}

////////////////////////////////////////////////////////////////////////////////
/// Read object from JSON, produced by ConvertToJSON() method.

void *TBufferJSON::ConvertFromJSONAny(const char *str, TClass **cl)
{
   if (cl)
      *cl = nullptr;

   nlohmann::json docu = nlohmann::json::parse(str);

   if (docu.is_null())
      return nullptr;

   if (!docu.is_object()) {
      // Error("ConvertFromJSONAny", "Only JSON objects are supported");
      return nullptr;
   }

   TBufferJSON buf(TBuffer::kRead);

   void *obj = buf.JsonReadAny(&docu, 0, cl);

   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Convert single data member to JSON structures
/// Returns string with converted member

TString TBufferJSON::JsonWriteMember(const void *ptr, TDataMember *member, TClass *memberClass, Int_t arraylen)
{
   if (member == 0)
      return "null";

   if (gDebug > 2)
      Info("JsonWriteMember", "Write member %s type %s ndim %d", member->GetName(), member->GetTrueTypeName(),
           member->GetArrayDim());

   Int_t tid = member->GetDataType() ? member->GetDataType()->GetType() : kNoType_t;
   if (strcmp(member->GetTrueTypeName(), "const char*") == 0)
      tid = kCharStar;
   else if (!member->IsBasic() || (tid == kOther_t) || (tid == kVoid_t))
      tid = kNoType_t;

   if (ptr == 0)
      return (tid == kCharStar) ? "\"\"" : "null";

   PushStack(0);
   fValue.Clear();

   if (tid != kNoType_t) {

      TArrayIndexProducer indx(member, arraylen, fArraySepar.Data());

      Int_t shift = 1;

      if (indx.IsArray() && (tid == kChar_t))
         shift = indx.ReduceDimension();

      char *ppp = (char *)ptr;

      if (indx.IsArray())
         fOutBuffer.Append(indx.GetBegin());

      do {
         fValue.Clear();

         switch (tid) {
         case kChar_t:
            if (shift > 1)
               JsonWriteConstChar((Char_t *)ppp, shift);
            else
               JsonWriteBasic(*((Char_t *)ppp));
            break;
         case kShort_t: JsonWriteBasic(*((Short_t *)ppp)); break;
         case kInt_t: JsonWriteBasic(*((Int_t *)ppp)); break;
         case kLong_t: JsonWriteBasic(*((Long_t *)ppp)); break;
         case kFloat_t: JsonWriteBasic(*((Float_t *)ppp)); break;
         case kCounter: JsonWriteBasic(*((Int_t *)ppp)); break;
         case kCharStar: JsonWriteConstChar((Char_t *)ppp); break;
         case kDouble_t: JsonWriteBasic(*((Double_t *)ppp)); break;
         case kDouble32_t: JsonWriteBasic(*((Double_t *)ppp)); break;
         case kchar: JsonWriteBasic(*((char *)ppp)); break;
         case kUChar_t: JsonWriteBasic(*((UChar_t *)ppp)); break;
         case kUShort_t: JsonWriteBasic(*((UShort_t *)ppp)); break;
         case kUInt_t: JsonWriteBasic(*((UInt_t *)ppp)); break;
         case kULong_t: JsonWriteBasic(*((ULong_t *)ppp)); break;
         case kBits: JsonWriteBasic(*((UInt_t *)ppp)); break;
         case kLong64_t: JsonWriteBasic(*((Long64_t *)ppp)); break;
         case kULong64_t: JsonWriteBasic(*((ULong64_t *)ppp)); break;
         case kBool_t: JsonWriteBasic(*((Bool_t *)ppp)); break;
         case kFloat16_t: JsonWriteBasic(*((Float_t *)ppp)); break;
         case kOther_t:
         case kVoid_t: break;
         }

         fOutBuffer.Append(fValue);
         if (indx.IsArray())
            fOutBuffer.Append(indx.NextSeparator());

         ppp += shift * member->GetUnitSize();

      } while (!indx.IsDone());

      fValue = fOutBuffer;

   } else if (memberClass == TString::Class()) {
      TString *str = (TString *)ptr;
      JsonWriteConstChar(str ? str->Data() : 0);
   } else if ((member->IsSTLContainer() == ROOT::kSTLvector) || (member->IsSTLContainer() == ROOT::kSTLlist) ||
              (member->IsSTLContainer() == ROOT::kSTLforwardlist)) {

      if (memberClass)
         ((TClass *)memberClass)->Streamer((void *)ptr, *this);
      else
         fValue = "[]";

      if (fValue == "0")
         fValue = "[]";

   } else if (memberClass && memberClass->GetBaseClassOffset(TArray::Class()) == 0) {
      TArray *arr = (TArray *)ptr;
      if ((arr != 0) && (arr->GetSize() > 0)) {
         arr->Streamer(*this);
         // WriteFastArray(arr->GetArray(), arr->GetSize());
         if (Stack()->fValues.GetLast() > 0) {
            Warning("TBufferJSON", "When streaming TArray, more than 1 object in the stack, use second item");
            fValue = Stack()->fValues.At(1)->GetName();
         }
      } else
         fValue = "[]";
   } else if (memberClass && !strcmp(memberClass->GetName(), "string")) {
      // here value contains quotes, stack can be ignored
      ((TClass *)memberClass)->Streamer((void *)ptr, *this);
   }
   PopStack();

   if (fValue.Length())
      return fValue;

   if ((memberClass == 0) || (member->GetArrayDim() > 0) || (arraylen > 0))
      return "<not supported>";

   return TBufferJSON::ConvertToJSON(ptr, memberClass);
}

////////////////////////////////////////////////////////////////////////////////
/// Check that object already stored in the buffer

Bool_t TBufferJSON::CheckObject(const TObject *obj)
{
   if (obj == 0)
      return kTRUE;

   return fJsonrMap.find(obj) != fJsonrMap.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Check that object already stored in the buffer

Bool_t TBufferJSON::CheckObject(const void *ptr, const TClass * /*cl*/)
{
   if (ptr == 0)
      return kTRUE;

   return fJsonrMap.find(ptr) != fJsonrMap.end();
}

////////////////////////////////////////////////////////////////////////////////
/// Convert object into json structures.
/// !!! Should be used only by TBufferJSON itself.
/// Use ConvertToJSON() methods to convert object to json

void TBufferJSON::WriteObject(const TObject *obj, Bool_t cacheReuse /* = kTRUE */)
{
   if (gDebug > 1)
      Info("WriteObject", "Object %p", obj);

   WriteObjectAny(obj, TObject::Class(), cacheReuse);
}

////////////////////////////////////////////////////////////////////////////////
/// add new level to the structures stack

TJSONStackObj *TBufferJSON::PushStack(Int_t inclevel)
{
   TJSONStackObj *curr = Stack();
   TJSONStackObj *stack = new TJSONStackObj();
   stack->fLevel = (curr ? curr->fLevel : 0) + inclevel;
   fStack.Add(stack);
   return stack;
}

////////////////////////////////////////////////////////////////////////////////
/// add new level to the structures stack for reading

TJSONStackObj *TBufferJSON::PushStackR(JSONObject_t current, Bool_t simple)
{
   if (!simple) {
      printf("Not a simple case, how we should support it?\n");
      // current = fXML->GetChild(current);
      // fXML->SkipEmpty(current);
   }

   TJSONStackObj *stack = new TJSONStackObj();
   stack->fNode = current;
   fStack.Add(stack);
   return stack;
}

////////////////////////////////////////////////////////////////////////////////
/// remove one level from stack

TJSONStackObj *TBufferJSON::PopStack()
{
   TObject *last = fStack.Last();
   if (last != 0) {
      fStack.Remove(last);
      delete last;
      fStack.Compress();
   }
   return dynamic_cast<TJSONStackObj *>(fStack.Last());
}

////////////////////////////////////////////////////////////////////////////////
/// return stack object of specified depth

TJSONStackObj *TBufferJSON::Stack(Int_t depth)
{
   TJSONStackObj *stack = 0;
   if (depth <= fStack.GetLast())
      stack = dynamic_cast<TJSONStackObj *>(fStack.At(fStack.GetLast() - depth));
   return stack;
}

////////////////////////////////////////////////////////////////////////////////
/// Append two string to the output JSON, normally separate by line break

void TBufferJSON::AppendOutput(const char *line0, const char *line1)
{
   if (line0 != nullptr)
      fOutput->Append(line0);

   if (line1 != nullptr) {
      if (fCompact % 10 < 2)
         fOutput->Append("\n");

      if (strlen(line1) > 0) {
         if (fCompact % 10 < 1) {
            TJSONStackObj *stack = Stack();
            if ((stack != nullptr) && (stack->fLevel > 0))
               fOutput->Append(' ', stack->fLevel);
         }
         fOutput->Append(line1);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Start new class member in JSON structures

void TBufferJSON::JsonStartElement(const TStreamerElement *elem, const TClass *base_class)
{
   const char *elem_name = nullptr;

   if (!base_class) {
      elem_name = elem->GetName();
   } else {
      switch (JsonSpecialClass(base_class)) {
      case TClassEdit::kVector: elem_name = "fVector"; break;
      case TClassEdit::kList: elem_name = "fList"; break;
      case TClassEdit::kForwardlist: elem_name = "fForwardlist"; break;
      case TClassEdit::kDeque: elem_name = "fDeque"; break;
      case TClassEdit::kMap: elem_name = "fMap"; break;
      case TClassEdit::kMultiMap: elem_name = "fMultiMap"; break;
      case TClassEdit::kSet: elem_name = "fSet"; break;
      case TClassEdit::kMultiSet: elem_name = "fMultiSet"; break;
      case TClassEdit::kUnorderedSet: elem_name = "fUnorderedSet"; break;
      case TClassEdit::kUnorderedMultiSet: elem_name = "fUnorderedMultiSet"; break;
      case TClassEdit::kUnorderedMap: elem_name = "fUnorderedMap"; break;
      case TClassEdit::kUnorderedMultiMap: elem_name = "fUnorderedMultiMap"; break;
      case TClassEdit::kBitSet: elem_name = "fBitSet"; break;
      case json_TArray: elem_name = "fArray"; break;
      case json_TString:
      case json_stdstring: elem_name = "fString"; break;
      }
   }

   if (!elem_name)
      return;

   if (IsReading()) {
      TJSONStackObj *stack = Stack();
      if (stack && stack->fNode) {
         nlohmann::json &json = *((nlohmann::json *)stack->fNode);

         if (json.count(elem_name) != 1) {
            Error("JsonStartElement", "Missing JSON structure for element %s", elem_name);
         } else {
            nlohmann::json &sub = json[elem_name];
            stack->fNode = &sub;
         }

      } else {
         Error("JsonStartElement", "Missing JSON node");
      }

   } else {
      AppendOutput(",", "\"");
      AppendOutput(elem_name);
      AppendOutput("\"");
      AppendOutput(fSemicolon.Data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// disable post-processing of the code
void TBufferJSON::JsonDisablePostprocessing()
{
   TJSONStackObj *stack = Stack();
   if (stack != 0)
      stack->fIsPostProcessed = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// return non-zero value when class has special handling in JSON
/// it is TCollection (-130), TArray (100), TString (110), std::string (120) and STL containers (1..6)

Int_t TBufferJSON::JsonSpecialClass(const TClass *cl) const
{
   if (!cl)
      return 0;

   Bool_t isarray = strncmp("TArray", cl->GetName(), 6) == 0;
   if (isarray)
      isarray = ((TClass *)cl)->GetBaseClassOffset(TArray::Class()) == 0;
   if (isarray)
      return json_TArray;

   // negative value used to indicate that collection stored as object
   if (((TClass *)cl)->GetBaseClassOffset(TCollection::Class()) == 0)
      return json_TCollection;

   // special case for TString - it is saved as string in JSON
   if (cl == TString::Class())
      return json_TString;

   bool isstd = TClassEdit::IsStdClass(cl->GetName());
   int isstlcont(ROOT::kNotSTL);
   if (isstd)
      isstlcont = cl->GetCollectionType();
   if (isstlcont > 0)
      return isstlcont;

   // also special handling for STL string, which handled similar to TString
   if (isstd && !strcmp(cl->GetName(), "string"))
      return json_stdstring;

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Write object to buffer
/// If object was written before, only pointer will be stored
/// If check_map==kFALSE, object will be stored in any case and pointer will not be registered in the map

void TBufferJSON::JsonWriteObject(const void *obj, const TClass *cl, Bool_t check_map)
{
   // static int  cnt = 0;

   if (!cl)
      obj = nullptr;

   // if (cnt++>100) return;

   if (gDebug > 0)
      Info("JsonWriteObject", "Object %p class %s check_map %s", obj, cl ? cl->GetName() : "null",
           check_map ? "true" : "false");

   Int_t special_kind = JsonSpecialClass(cl);

   TString fObjectOutput, *fPrevOutput(nullptr);

   TJSONStackObj *stack = Stack();

   if (stack && stack->fAccObjects && ((fValue.Length() > 0) || (stack->fValues.GetLast() >= 0))) {
      // accumulate data of super-object in stack

      if (fValue.Length() > 0) {
         stack->fValues.Add(new TObjString(fValue));
         fValue.Clear();
      }

      // redirect output to local buffer, use it later as value
      fPrevOutput = fOutput;
      fOutput = &fObjectOutput;
   } else if ((special_kind <= 0) || (special_kind > json_TArray)) {
      // FIXME: later post processing should be active for all special classes, while they all keep output in the value
      JsonDisablePostprocessing();
   }

   if (!obj) {
      AppendOutput("null");
      goto post_process;
   }

   if (special_kind <= 0) {
      // add element name which should correspond to the object
      if (check_map) {
         std::map<const void *, unsigned>::const_iterator iter = fJsonrMap.find(obj);
         if (iter != fJsonrMap.end()) {
            // old-style refs, coded into string like "$ref12"
            // AppendOutput(Form("\"$ref:%u\"", iter->second));
            // new-style refs, coded into extra object {"$ref":12}, auto-detected by JSROOT 4.8 and higher
            AppendOutput(Form("{\"$ref\":%u}", iter->second));
            goto post_process;
         }
         fJsonrMap[obj] = fJsonrCnt;
      }

      fJsonrCnt++; // object counts is important in dereferencing part

      stack = PushStack(2);
      AppendOutput("{", "\"_typename\"");
      AppendOutput(fSemicolon.Data());
      AppendOutput("\"");
      AppendOutput(cl->GetName());
      AppendOutput("\"");
   } else {
      // for array, string and STL collections different handling -
      // they not recognized at the end as objects in JSON
      stack = PushStack(0);
   }

   if (gDebug > 3)
      Info("JsonWriteObject", "Starting object %p write for class: %s", obj, cl->GetName());

   stack->fAccObjects = special_kind < ROOT::kSTLend;

   if (special_kind == json_TCollection)
      JsonStreamCollection((TCollection *)obj, cl);
   else
      ((TClass *)cl)->Streamer((void *)obj, *this);

   if (gDebug > 3)
      Info("JsonWriteObject", "Done object %p write for class: %s", obj, cl->GetName());

   if (special_kind == json_TArray) {
      if (stack->fValues.GetLast() != 0)
         Error("JsonWriteObject", "Problem when writing array");
      stack->fValues.Delete();
   } else if ((special_kind == json_TString) || (special_kind == json_stdstring)) {
      if (stack->fValues.GetLast() > 1)
         Error("JsonWriteObject", "Problem when writing TString or std::string");
      stack->fValues.Delete();
      AppendOutput(fValue.Data());
      fValue.Clear();
   } else if ((special_kind > 0) && (special_kind <= TClassEdit::kBitSet)) {
      // here make STL container processing

      if (stack->fValues.GetLast() < 0) {
         // empty container
         if (fValue != "0")
            Error("JsonWriteObject", "With empty stack fValue!=0");
         fValue = "[]";
      } else {

         Int_t size = TString(stack->fValues.At(0)->GetName()).Atoi();

         if ((stack->fValues.GetLast() == 0) && ((size > 1) || (fValue.Index("[") == 0))) {
            // case of simple vector, array already in the value
            stack->fValues.Delete();
            if (fValue.Length() == 0) {
               Error("JsonWriteObject", "Empty value when it should contain something");
               fValue = "[]";
            }

         } else {
            const char *separ = "[";

            if (fValue.Length() > 0) {
               stack->fValues.Add(new TObjString(fValue));
               fValue.Clear();
            }

            if ((size * 2 == stack->fValues.GetLast()) &&
                ((special_kind == TClassEdit::kMap) || (special_kind == TClassEdit::kMultiMap) ||
                 (special_kind == TClassEdit::kUnorderedMap) || (special_kind == TClassEdit::kUnorderedMultiMap))) {
               // special handling for std::map.
               // Create entries like { '$pair': 'typename' , 'first' : key, 'second' : value }

               TString pairtype = cl->GetName();
               if (pairtype.Index("multimap<") == 0)
                  pairtype.Replace(0, 9, "pair<");
               else if (pairtype.Index("map<") == 0)
                  pairtype.Replace(0, 4, "pair<");
               else
                  pairtype = "TPair";
               pairtype = TString("\"") + pairtype + TString("\"");
               for (Int_t k = 1; k < stack->fValues.GetLast(); k += 2) {
                  fValue.Append(separ);
                  separ = fArraySepar.Data();
                  // fJsonrCnt++; // do not add entry in the map, can conflict with objects inside values
                  fValue.Append("{");
                  fValue.Append("\"$pair\"");
                  fValue.Append(fSemicolon);
                  fValue.Append(pairtype.Data());
                  fValue.Append(fArraySepar);
                  fValue.Append("\"first\"");
                  fValue.Append(fSemicolon);
                  fValue.Append(stack->fValues.At(k)->GetName());
                  fValue.Append(fArraySepar);
                  fValue.Append("\"second\"");
                  fValue.Append(fSemicolon);
                  fValue.Append(stack->fValues.At(k + 1)->GetName());
                  fValue.Append("}");
               }
            } else {
               // for most stl containers write just like blob, but skipping first element with size
               for (Int_t k = 1; k <= stack->fValues.GetLast(); k++) {
                  fValue.Append(separ);
                  separ = fArraySepar.Data();
                  fValue.Append(stack->fValues.At(k)->GetName());
               }
            }

            fValue.Append("]");
            stack->fValues.Delete();
         }
      }
   }

   // reuse post-processing code for TObject or TRef
   PerformPostProcessing(stack, cl);

   if ((special_kind == 0) && ((stack->fValues.GetLast() >= 0) || (fValue.Length() > 0))) {
      if (gDebug > 0)
         Info("JsonWriteObject", "Create blob value for class %s", cl->GetName());

      AppendOutput(fArraySepar.Data(), "\"_blob\"");
      AppendOutput(fSemicolon.Data());

      const char *separ = "[";

      for (Int_t k = 0; k <= stack->fValues.GetLast(); k++) {
         AppendOutput(separ);
         separ = fArraySepar.Data();
         AppendOutput(stack->fValues.At(k)->GetName());
      }

      if (fValue.Length() > 0) {
         AppendOutput(separ);
         AppendOutput(fValue.Data());
      }

      AppendOutput("]");

      fValue.Clear();
      stack->fValues.Delete();
   }

   PopStack();

   if (special_kind <= 0) {
      AppendOutput(0, "}");
   }

post_process:

   if (fPrevOutput) {
      fOutput = fPrevOutput;
      // for STL containers and TArray object in fValue itself
      if ((special_kind <= 0) || (special_kind > json_TArray))
         fValue = fObjectOutput;
      else if (fObjectOutput.Length() != 0)
         Error("JsonWriteObject", "Non-empty object output for special class %s", cl->GetName());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Recreate object from json structure.
/// Return pointer to read object.
/// if (cl!=0) returns pointer to class of object

void *TBufferJSON::JsonReadAny(JSONObject_t node, void *obj, TClass **cl)
{
   if (!node)
      return nullptr;

   PushStackR(node);

   void *res = JsonReadObject(obj, nullptr, cl);

   PopStack();

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// store content of collection

void TBufferJSON::JsonStreamCollection(TCollection *col, const TClass *)
{
   AppendOutput(",", "\"name\"");
   AppendOutput(fSemicolon.Data());
   AppendOutput("\"");
   AppendOutput(col->GetName());
   AppendOutput("\",", "\"arr\"");
   AppendOutput(fSemicolon.Data());

   // collection treated as JS Array
   AppendOutput("[");

   bool islist = col->InheritsFrom(TList::Class());
   TMap *map = 0;
   if (col->InheritsFrom(TMap::Class()))
      map = dynamic_cast<TMap *>(col);

   TString sopt;
   if (islist) {
      sopt.Capacity(500);
      sopt = "[";
   }

   TIter iter(col);
   TObject *obj;
   Bool_t first = kTRUE;
   while ((obj = iter()) != 0) {
      if (!first)
         AppendOutput(fArraySepar.Data());

      if (map) {
         // fJsonrCnt++; // do not account map pair as JSON object
         AppendOutput("{", "\"$pair\"");
         AppendOutput(fSemicolon.Data());
         AppendOutput("\"TPair\"");
         AppendOutput(fArraySepar.Data(), "\"first\"");
         AppendOutput(fSemicolon.Data());
      }

      WriteObjectAny(obj, TObject::Class());

      if (map) {
         AppendOutput(fArraySepar.Data(), "\"second\"");
         AppendOutput(fSemicolon.Data());
         WriteObjectAny(map->GetValue(obj), TObject::Class());
         AppendOutput("", "}");
      }

      if (islist) {
         if (!first)
            sopt.Append(fArraySepar.Data());
         sopt.Append("\"");
         sopt.Append(iter.GetOption());
         sopt.Append("\"");
      }

      first = kFALSE;
   }

   AppendOutput("]");

   if (islist) {
      sopt.Append("]");
      AppendOutput(",", "\"opt\"");
      AppendOutput(fSemicolon.Data());
      AppendOutput(sopt.Data());
   }
   fValue.Clear();
}

////////////////////////////////////////////////////////////////////////////////
/// Read object from current JSON node

void *TBufferJSON::JsonReadObject(void *obj, const TClass *objClass, TClass **readClass)
{
   if (readClass)
      *readClass = nullptr;

   TJSONStackObj *stack = Stack();

   if (!stack || !stack->fNode)
      return obj;

   nlohmann::json *json = (nlohmann::json *)stack->fNode;

   // check if null pointer
   if (json->is_null()) return nullptr;

   Bool_t process_stl = (stack->fStlIndx >= 0);
   if (process_stl)
      json = &(json->at(stack->fStlIndx++));

   // enum { json_TArray = 100, json_TCollection = -130, json_TString = 110, json_stdstring = 120 };
   Int_t special_kind = JsonSpecialClass(objClass);

   //if (gDebug>2)
   //   Info("JsonReadObject", "Special kind %d process_stl %s clname %s", special_kind, (process_stl ? "true" : "false"), (objClass ? objClass->GetName() : "---"));

   // Extract pointer
   if (json->is_object() && (json->size() == 1) && (json->find("$ref") != json->end())) {
      unsigned refid = json->at("$ref").get<unsigned>();

      if (gDebug>2)
         Info("JsonReadObject", "Extract object reference %u", refid);

      auto elem = fReadMap.find(refid);
      if (elem == fReadMap.end()) {
         Error("JsonReadObject", "Fail to find object for reference %u", refid);
         return nullptr;
      }

      if (readClass)
         *readClass = elem->second.cl;
      return elem->second.obj;
   }

   // special case of strings - they do not create JSON object, but just string
   if ((special_kind == json_stdstring) || (special_kind == json_TString)) {
      if (!obj)
         obj = objClass->New();

      if (gDebug > 2)
         Info("JsonReadObject","Read string from %s", json->dump().c_str());

      if (special_kind == json_stdstring)
         *((std::string *) obj) = json->get<std::string>();
      else
         *((TString *) obj) = json->get<std::string>().c_str();

      if (readClass)
         *readClass = (TClass *) objClass;

      return obj;
   }

   // from now all operations performed with sub-element,
   // stack should be repaired at the end
   if (process_stl)
      stack = PushStackR(json);

   Bool_t isBase = (stack->fElem && objClass) ? stack->fElem->IsBase() : kFALSE; // base class

   TClass *jsonClass = nullptr;

   if (isBase) {
      // base class has special handling - no additional level and no extra refid

      jsonClass = (TClass *) objClass;
      if (!obj) {
         Error("JsonReadObject", "No object when reading base class");
         if (process_stl) PopStack();
         return obj;
      }

      if (gDebug > 1)
         Info("JsonReadObject", "Reading baseclass %s ptr %p", objClass->GetName(), obj);

   } else if ((special_kind == json_TArray) || ((special_kind > 0) && (special_kind < ROOT::kSTLend))) {

      jsonClass = (TClass *) objClass;

      if (!obj)
         obj = jsonClass->New();

      if (!json->is_array()) Error("JsonReadObject", "Not array when expecting such %s", json->dump().c_str());

      // add to stack array size, which will be extracted before reading array itself by custom TArray streamer
      stack->PushIntValue(json->size());

   } else {

      std::string clname = json->at("_typename").get<std::string>();

      jsonClass = clname.empty() ? nullptr : TClass::GetClass(clname.c_str());

      if (!jsonClass) {
         Error("JsonReadObject", "Cannot find class %s", clname.c_str());
         if (process_stl) PopStack();
         return obj;
      }

      if (objClass && (jsonClass!=objClass)) {
         Error("JsonReadObject", "Class mismatch between provided %s and in JSON %s", objClass->GetName(), jsonClass->GetName());
      }

      if (!obj)
         obj = jsonClass->New();

      if (gDebug > 1)
         Info("JsonReadObject", "Reading object of class %s refid %u ptr %p", clname.c_str(), fJsonrCnt, obj);

      // add new element to the reading map
      fReadMap[fJsonrCnt++] = ObjectEntry(obj, jsonClass);
   }

   // there are two ways to handle custom streamers
   // either prepare data before streamer and tweak basic function which are reading values like UInt32_t
   // or try reimplement custom streamer here

   if (jsonClass == TObject::Class()) {
      // for TObject we reimplement custom streamer - it is much easier

      if (gDebug > 1)
         Info("JsonReadObject", "Reading TObject from %s", json->dump().c_str());

      TObject *tobj = (TObject *)obj;

      UInt_t uid = json->at("fUniqueID").get<unsigned>();
      UInt_t bits = json->at("fBits").get<unsigned>();
      // UInt32_t pid = json["fPID"].get<unsigned>();

      tobj->SetUniqueID(uid);
      // there is no method to set all bits directly - do it one by one
      for (unsigned n=0;n<32;n++)
         tobj->SetBit(n, (bits & BIT(n)) != 0);

   } else {

      // special handling of STL which coded into arrays
      if ((special_kind > 0) && (special_kind < ROOT::kSTLend)) stack->fStlIndx = 0;

      // workaround for missing version in JSON structures
      stack->fClVersion = jsonClass->GetClassVersion();

      jsonClass->Streamer((void *)obj, *this);

      stack->fClVersion = 0;

      stack->fStlIndx = -1; // reset STL index for itself to prevent looping
   }

   // return back stack position
   if (process_stl) PopStack();

   if (gDebug > 1)
      Info("JsonReadObject", "Reading object of class %s done", jsonClass->GetName());

   if (readClass)
      *readClass = jsonClass;

   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Read data for specified class

Int_t TBufferJSON::ReadClassBuffer(const TClass *cl, void *ptr, const TClass *)
{
   TStreamerInfo *sinfo = (TStreamerInfo *)cl->GetStreamerInfo();

   if (gDebug > 1)
      Info("ReadClassBuffer", "Deserialize object %s sinfo ver %d", cl->GetName(),
           (sinfo ? sinfo->GetClassVersion() : -1111));

   // deserialize the object, using read text actions
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)ptr);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Deserialize information from a buffer into an object.
///
/// Note: This function is called by the xxx::Streamer() functions in
/// rootcint-generated dictionaries.
/// This function assumes that the class version and the byte count
/// information have been read.
///
/// \param[in] version The version number of the class
/// \param[in] start   The starting position in the buffer b
/// \param[in] count   The number of bytes for this object in the buffer
///

Int_t TBufferJSON::ReadClassBuffer(const TClass *cl, void *ptr, Int_t /*version*/, UInt_t /*start*/, UInt_t /*count*/, const TClass * /*onFileClass*/)
{
   TStreamerInfo *sinfo = (TStreamerInfo *)cl->GetStreamerInfo();

   if (gDebug > 1)
      Info("ReadClassBuffer", "Deserialize object %s sinfo ver %d", cl->GetName(),
           (sinfo ? sinfo->GetClassVersion() : -1111));

   // deserialize the object, using read text actions
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)ptr);

   return 0;
}



////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and ReadBuffer functions
/// and indent new level in json structure.
/// This call indicates, that TStreamerInfo functions starts streaming
/// object data of correspondent class

void TBufferJSON::IncrementLevel(TVirtualStreamerInfo *info)
{
   if (gDebug > 2)
      Info("IncrementLevel", "Class: %s", (info ? info->GetClass()->GetName() : "custom"));

   WorkWithClass((TStreamerInfo *)info);
}

////////////////////////////////////////////////////////////////////////////////
/// Prepares buffer to stream data of specified class

void TBufferJSON::WorkWithClass(TStreamerInfo *sinfo, const TClass *cl)
{
   if (sinfo)
      cl = sinfo->GetClass();

   if (!cl)
      return;

   if (gDebug > 3)
      Info("WorkWithClass", "Class: %s", cl->GetName());

   TJSONStackObj *stack = Stack();

   if (IsReading()) {
      stack = PushStackR(stack->fNode);
   } else if (stack && stack->IsStreamerElement() && !stack->fIsObjStarted &&
              ((stack->fElem->GetType() == TStreamerInfo::kObject) ||
               (stack->fElem->GetType() == TStreamerInfo::kAny))) {

      stack->fIsObjStarted = kTRUE;

      fJsonrCnt++; // count object, but do not keep reference

      stack = PushStack(2);
      AppendOutput("{", "\"_typename\"");
      AppendOutput(fSemicolon.Data());
      AppendOutput("\"");
      AppendOutput(cl->GetName());
      AppendOutput("\"");
   } else {
      stack = PushStack(0);
   }

   stack->fInfo = sinfo;
   stack->fIsStreamerInfo = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and ReadBuffer functions
/// and decrease level in json structure.

void TBufferJSON::DecrementLevel(TVirtualStreamerInfo *info)
{
   if (gDebug > 2)
      Info("DecrementLevel", "Class: %s", (info ? info->GetClass()->GetName() : "custom"));

   TJSONStackObj *stack = Stack();

   if (stack->IsStreamerElement()) {
      if (gDebug > 3)
         Info("DecrementLevel", "    Perform post-processing elem: %s", stack->fElem->GetName());

      PerformPostProcessing(stack);

      stack = PopStack(); // remove stack of last element
   }

   if (stack->fInfo != (TStreamerInfo *)info)
      Error("DecrementLevel", "    Mismatch of streamer info");

   PopStack(); // back from data of stack info

   if (gDebug > 3)
      Info("DecrementLevel", "Class: %s done", (info ? info->GetClass()->GetName() : "custom"));
}

////////////////////////////////////////////////////////////////////////////////
/// Return current streamer info element

TVirtualStreamerInfo *TBufferJSON::GetInfo()
{
   TJSONStackObj *stack = Stack();
   return stack ? stack->fInfo : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and ReadBuffer functions
/// and add/verify next element of json structure
/// This calls allows separate data, correspondent to one class member, from another

void TBufferJSON::SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type)
{
   if (gDebug > 3)
      Info("SetStreamerElementNumber", "Element name %s", elem->GetName());

   WorkWithElement(elem, comp_type);
}

////////////////////////////////////////////////////////////////////////////////
/// This is call-back from streamer which indicates
/// that class member will be streamed
/// Name of element used in JSON

void TBufferJSON::WorkWithElement(TStreamerElement *elem, Int_t)
{
   TJSONStackObj *stack = Stack();
   if (!stack) {
      Error("WorkWithElement", "stack is empty");
      return;
   }

   if (gDebug > 0)
      Info("WorkWithElement", "    Start element %s type %d typename %s", elem ? elem->GetName() : "---",
           elem ? elem->GetType() : -1, elem ? elem->GetTypeName() : "---");

   if (stack->IsStreamerElement()) {
      // this is post processing

      if (IsWriting()) {
         if (gDebug > 3)
            Info("WorkWithElement", "    Perform post-processing elem: %s", stack->fElem->GetName());
         PerformPostProcessing(stack);
      }

      stack = PopStack(); // go level back
   }

   fValue.Clear();

   if (!stack) {
      Error("WorkWithElement", "Lost of stack");
      return;
   }

   TStreamerInfo *info = stack->fInfo;
   if (!stack->IsStreamerInfo()) {
      Error("WorkWithElement", "Problem in Inc/Dec level");
      return;
   }

   Int_t number = info ? info->GetElements()->IndexOf(elem) : -1;

   if (!elem) {
      Error("WorkWithElement", "streamer info returns elem = 0");
      return;
   }

   TClass *base_class = elem->IsBase() ? elem->GetClassPointer() : nullptr;

   stack = IsReading() ? PushStackR(stack->fNode) : PushStack(0);
   stack->fElem = (TStreamerElement *)elem;
   stack->fIsElemOwner = (number < 0);

   JsonStartElement(elem, base_class);

   if ((elem->GetType() == TStreamerInfo::kOffsetL + TStreamerInfo::kStreamLoop) && (elem->GetArrayDim() > 0)) {
      // array of array, start handling here
      stack->fIndx = new TArrayIndexProducer(elem, -1, fArraySepar.Data());
      if (IsWriting())
         AppendOutput(stack->fIndx->GetBegin());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Should be called in the beginning of custom class streamer.
/// Informs buffer data about class which will be streamed now.
///
/// ClassBegin(), ClassEnd() and ClassMember() should be used in
/// custom class streamers to specify which kind of data are
/// now streamed. Such information is used to correctly
/// convert class data to JSON. Without that functions calls
/// classes with custom streamers cannot be used with TBufferJSON

void TBufferJSON::ClassBegin(const TClass *cl, Version_t)
{
   WorkWithClass(0, cl);
}

////////////////////////////////////////////////////////////////////////////////
/// Should be called at the end of custom streamer
/// See TBufferJSON::ClassBegin for more details

void TBufferJSON::ClassEnd(const TClass *)
{
   DecrementLevel(0);
}

////////////////////////////////////////////////////////////////////////////////
/// Method indicates name and typename of class member,
/// which should be now streamed in custom streamer
/// Following combinations are supported:
/// 1. name = "ClassName", typeName = 0 or typename==ClassName
///    This is a case, when data of parent class "ClassName" should be streamed.
///     For instance, if class directly inherited from TObject, custom
///     streamer should include following code:
/// ~~~{.cpp}
///       b.ClassMember("TObject");
///       TObject::Streamer(b);
/// ~~~
/// 2. Basic data type
/// ~~~{.cpp}
///      b.ClassMember("fInt","Int_t");
///      b >> fInt;
/// ~~~
/// 3. Array of basic data types
/// ~~~{.cpp}
///      b.ClassMember("fArr","Int_t", 5);
///      b.ReadFastArray(fArr, 5);
/// ~~~
/// 4. Object as data member
/// ~~~{.cpp}
///      b.ClassMember("fName","TString");
///      fName.Streamer(b);
/// ~~~
/// 5. Pointer on object as data member
/// ~~~{.cpp}
///      b.ClassMember("fObj","TObject*");
///      b.StreamObject(fObj);
/// ~~~
///
/// arrsize1 and arrsize2 arguments (when specified) indicate first and
/// second dimension of array. Can be used for array of basic types.
/// See ClassBegin() method for more details.

void TBufferJSON::ClassMember(const char *name, const char *typeName, Int_t arrsize1, Int_t arrsize2)
{
   if (typeName == 0)
      typeName = name;

   if ((name == 0) || (strlen(name) == 0)) {
      Error("ClassMember", "Invalid member name");
      return;
   }

   TString tname = typeName;

   Int_t typ_id = -1;

   if (strcmp(typeName, "raw:data") == 0)
      typ_id = TStreamerInfo::kMissing;

   if (typ_id < 0) {
      TDataType *dt = gROOT->GetType(typeName);
      if (dt != 0)
         if ((dt->GetType() > 0) && (dt->GetType() < 20))
            typ_id = dt->GetType();
   }

   if (typ_id < 0)
      if (strcmp(name, typeName) == 0) {
         TClass *cl = TClass::GetClass(tname.Data());
         if (cl != 0)
            typ_id = TStreamerInfo::kBase;
      }

   if (typ_id < 0) {
      Bool_t isptr = kFALSE;
      if (tname[tname.Length() - 1] == '*') {
         tname.Resize(tname.Length() - 1);
         isptr = kTRUE;
      }
      TClass *cl = TClass::GetClass(tname.Data());
      if (cl == 0) {
         Error("ClassMember", "Invalid class specifier %s", typeName);
         return;
      }

      if (cl->IsTObject())
         typ_id = isptr ? TStreamerInfo::kObjectp : TStreamerInfo::kObject;
      else
         typ_id = isptr ? TStreamerInfo::kAnyp : TStreamerInfo::kAny;

      if ((cl == TString::Class()) && !isptr)
         typ_id = TStreamerInfo::kTString;
   }

   TStreamerElement *elem = 0;

   if (typ_id == TStreamerInfo::kMissing) {
      elem = new TStreamerElement(name, "title", 0, typ_id, "raw:data");
   } else if (typ_id == TStreamerInfo::kBase) {
      TClass *cl = TClass::GetClass(tname.Data());
      if (cl != 0) {
         TStreamerBase *b = new TStreamerBase(tname.Data(), "title", 0);
         b->SetBaseVersion(cl->GetClassVersion());
         elem = b;
      }
   } else if ((typ_id > 0) && (typ_id < 20)) {
      elem = new TStreamerBasicType(name, "title", 0, typ_id, typeName);
   } else if ((typ_id == TStreamerInfo::kObject) || (typ_id == TStreamerInfo::kTObject) ||
              (typ_id == TStreamerInfo::kTNamed)) {
      elem = new TStreamerObject(name, "title", 0, tname.Data());
   } else if (typ_id == TStreamerInfo::kObjectp) {
      elem = new TStreamerObjectPointer(name, "title", 0, tname.Data());
   } else if (typ_id == TStreamerInfo::kAny) {
      elem = new TStreamerObjectAny(name, "title", 0, tname.Data());
   } else if (typ_id == TStreamerInfo::kAnyp) {
      elem = new TStreamerObjectAnyPointer(name, "title", 0, tname.Data());
   } else if (typ_id == TStreamerInfo::kTString) {
      elem = new TStreamerString(name, "title", 0);
   }

   if (elem == 0) {
      Error("ClassMember", "Invalid combination name = %s type = %s", name, typeName);
      return;
   }

   if (arrsize1 > 0) {
      elem->SetArrayDim(arrsize2 > 0 ? 2 : 1);
      elem->SetMaxIndex(0, arrsize1);
      if (arrsize2 > 0)
         elem->SetMaxIndex(1, arrsize2);
   }

   // we indicate that there is no streamerinfo
   WorkWithElement(elem, -1);
}

////////////////////////////////////////////////////////////////////////////////
/// Function is converts TObject and TString structures to more compact representation

void TBufferJSON::PerformPostProcessing(TJSONStackObj *stack, const TClass *obj_cl)
{
   if (stack->fIsPostProcessed || IsReading())
      return;

   const TStreamerElement *elem = stack->fElem;

   if (!elem && !obj_cl)
      return;

   stack->fIsPostProcessed = kTRUE;

   // when element was written as separate object, close only braces and exit
   if (stack->fIsObjStarted) {
      AppendOutput("", "}");
      return;
   }

   Bool_t isTObject(kFALSE), isTRef(kFALSE), isTString(kFALSE), isSTLstring(kFALSE), isOffsetPArray(kFALSE),
      isTArray(kFALSE);

   if (obj_cl) {
      if (obj_cl == TObject::Class())
         isTObject = kTRUE;
      else if (obj_cl == TRef::Class())
         isTRef = kTRUE;
      else
         return;
   } else {
      const char *typname = elem->IsBase() ? elem->GetName() : elem->GetTypeName();
      isTObject = (elem->GetType() == TStreamerInfo::kTObject) || (strcmp("TObject", typname) == 0);
      isTString = elem->GetType() == TStreamerInfo::kTString;
      isSTLstring = elem->GetType() == TStreamerInfo::kSTLstring;
      isOffsetPArray = (elem->GetType() > TStreamerInfo::kOffsetP) && (elem->GetType() < TStreamerInfo::kOffsetP + 20);
      isTArray = (strncmp("TArray", typname, 6) == 0);
   }

   if (isTString || isSTLstring) {
      // just remove all kind of string length information

      if (gDebug > 3)
         Info("PerformPostProcessing", "reformat string value = '%s'", fValue.Data());

      stack->fValues.Delete();
   } else if (isOffsetPArray) {
      // basic array with [fN] comment

      if ((stack->fValues.GetLast() < 0) && (fValue == "0")) {
         fValue = "[]";
      } else if ((stack->fValues.GetLast() == 0) && (strcmp(stack->fValues.Last()->GetName(), "1") == 0)) {
         stack->fValues.Delete();
      } else {
         Error("PerformPostProcessing", "Wrong values for kOffsetP element %s", (elem ? elem->GetName() : "---"));
         stack->fValues.Delete();
         fValue = "[]";
      }
   } else if (isTObject || isTRef) {
      // complex workaround for TObject/TRef streamer
      // would be nice if other solution can be found
      // Here is not supported TRef on TRef (double reference)

      Int_t cnt = stack->fValues.GetLast() + 1;
      if (fValue.Length() > 0)
         cnt++;

      if (cnt < 2 || cnt > 3) {
         if (gDebug > 0)
            Error("PerformPostProcessing", "When storing TObject/TRef, strange number of items %d", cnt);
         AppendOutput(",", "\"dummy\"");
         AppendOutput(fSemicolon.Data());
      } else {
         AppendOutput(",", "\"fUniqueID\"");
         AppendOutput(fSemicolon.Data());
         AppendOutput(stack->fValues.At(0)->GetName());
         AppendOutput(",", "\"fBits\"");
         AppendOutput(fSemicolon.Data());
         AppendOutput((stack->fValues.GetLast() > 0) ? stack->fValues.At(1)->GetName() : fValue.Data());
         if (cnt == 3) {
            AppendOutput(",", "\"fPID\"");
            AppendOutput(fSemicolon.Data());
            AppendOutput((stack->fValues.GetLast() > 1) ? stack->fValues.At(2)->GetName() : fValue.Data());
         }

         stack->fValues.Delete();
         fValue.Clear();
         return;
      }

   } else if (isTArray) {
      // for TArray one deletes complete stack
      stack->fValues.Delete();
   }

   if (elem && elem->IsBase() && (fValue.Length() == 0)) {
      // here base class data already completely stored
      return;
   }

   if (stack->fValues.GetLast() >= 0) {
      // append element blob data just as abstract array, user is responsible to decode it
      AppendOutput("[");
      for (Int_t n = 0; n <= stack->fValues.GetLast(); n++) {
         AppendOutput(stack->fValues.At(n)->GetName());
         AppendOutput(fArraySepar.Data());
      }
   }

   if (fValue.Length() == 0) {
      AppendOutput("null");
   } else {
      AppendOutput(fValue.Data());
      fValue.Clear();
   }

   if (stack->fValues.GetLast() >= 0)
      AppendOutput("]");
}

////////////////////////////////////////////////////////////////////////////////
/// suppressed function of TBuffer

TClass *TBufferJSON::ReadClass(const TClass *, UInt_t *)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// suppressed function of TBuffer

void TBufferJSON::WriteClass(const TClass *)
{
}

////////////////////////////////////////////////////////////////////////////////
/// suppressed function of TBuffer

Int_t TBufferJSON::CheckByteCount(UInt_t /*r_s */, UInt_t /*r_c*/, const TClass * /*cl*/)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// suppressed function of TBuffer

Int_t TBufferJSON::CheckByteCount(UInt_t, UInt_t, const char *)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// suppressed function of TBuffer

void TBufferJSON::SetByteCount(UInt_t, Bool_t)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Skip class version from I/O buffer.

void TBufferJSON::SkipVersion(const TClass *cl)
{
   ReadVersion(0, 0, cl);
}

////////////////////////////////////////////////////////////////////////////////
/// read version value from buffer

Version_t TBufferJSON::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass *cl)
{
   Version_t res = cl ? cl->GetClassVersion() : 0;

   if (start)
      *start = 0;
   if (bcnt)
      *bcnt = 0;

   if (!cl) {
      TJSONStackObj *stack = Stack();
      if (stack && stack->fClVersion) {
         res = stack->fClVersion;
         stack->fClVersion = 0;
      }
   }

   if (gDebug > 3)
      Info("ReadVersion", "Result: %d Class: %s", res, (cl ? cl->GetName() : "---"));

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Ignored in TBufferJSON

UInt_t TBufferJSON::WriteVersion(const TClass * /*cl*/, Bool_t /* useBcnt */)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read object from buffer. Only used from TBuffer

void *TBufferJSON::ReadObjectAny(const TClass *expectedClass)
{
   if (gDebug > 2)
      Info("ReadObjectAny", "From current JSON node");
   void *res = JsonReadObject(nullptr, expectedClass);
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Skip any kind of object from buffer

void TBufferJSON::SkipObjectAny()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Write object to buffer. Only used from TBuffer

void TBufferJSON::WriteObjectClass(const void *actualObjStart,
                                   const TClass *actualClass,
                                   Bool_t cacheReuse)
{
   if (gDebug > 3)
      Info("WriteObjectClass", "Class %s", (actualClass ? actualClass->GetName() : " null"));

   JsonWriteObject(actualObjStart, actualClass, cacheReuse);
}

#define TJSONPushValue()    \
   if (fValue.Length() > 0) \
      Stack()->PushValue(fValue);

// macro to read array, which include size attribute
#define TBufferJSON_ReadArray(tname, vname)  \
   {                                         \
      printf("JSON::ReadArray %p\n", vname); \
      if (!vname)                            \
         return 0;                           \
      return 1;                              \
   }

////////////////////////////////////////////////////////////////////////////////
/// read a Float16_t from the buffer

void TBufferJSON::ReadFloat16(Float_t *, TStreamerElement * /*ele*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// read a Double32_t from the buffer

void TBufferJSON::ReadDouble32(Double_t *, TStreamerElement * /*ele*/)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the factor and minimun value have
/// been specified
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferJSON does not optimize space in this case.

void TBufferJSON::ReadWithFactor(Float_t *, Double_t /* factor */, Double_t /* minvalue */)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Float16_t from the buffer when the number of bits is specified
/// (explicitly or not)
/// see comments about Float16_t encoding at TBufferFile::WriteFloat16().
/// Currently TBufferJSON does not optimize space in this case.

void TBufferJSON::ReadWithNbits(Float_t *, Int_t /* nbits */)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the factor and minimun value have
/// been specified
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferJSON does not optimize space in this case.

void TBufferJSON::ReadWithFactor(Double_t *, Double_t /* factor */, Double_t /* minvalue */)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Read a Double32_t from the buffer when the number of bits is specified
/// (explicitly or not)
/// see comments about Double32_t encoding at TBufferFile::WriteDouble32().
/// Currently TBufferJSON does not optimize space in this case.

void TBufferJSON::ReadWithNbits(Double_t *, Int_t /* nbits */)
{
}

////////////////////////////////////////////////////////////////////////////////
/// write a Float16_t to the buffer

void TBufferJSON::WriteFloat16(Float_t *f, TStreamerElement * /*ele*/)
{
   TJSONPushValue();

   JsonWriteBasic(*f);
}

////////////////////////////////////////////////////////////////////////////////
/// write a Double32_t to the buffer

void TBufferJSON::WriteDouble32(Double_t *d, TStreamerElement * /*ele*/)
{
   TJSONPushValue();

   JsonWriteBasic(*d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Bool_t from buffer

Int_t TBufferJSON::ReadArray(Bool_t *&b)
{
   TBufferJSON_ReadArray(Bool_t, b);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Char_t from buffer

Int_t TBufferJSON::ReadArray(Char_t *&c)
{
   TBufferJSON_ReadArray(Char_t, c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UChar_t from buffer

Int_t TBufferJSON::ReadArray(UChar_t *&c)
{
   TBufferJSON_ReadArray(UChar_t, c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Short_t from buffer

Int_t TBufferJSON::ReadArray(Short_t *&h)
{
   TBufferJSON_ReadArray(Short_t, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UShort_t from buffer

Int_t TBufferJSON::ReadArray(UShort_t *&h)
{
   TBufferJSON_ReadArray(UShort_t, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Int_t from buffer

Int_t TBufferJSON::ReadArray(Int_t *&i)
{
   TBufferJSON_ReadArray(Int_t, i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UInt_t from buffer

Int_t TBufferJSON::ReadArray(UInt_t *&i)
{
   TBufferJSON_ReadArray(UInt_t, i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long_t from buffer

Int_t TBufferJSON::ReadArray(Long_t *&l)
{
   TBufferJSON_ReadArray(Long_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong_t from buffer

Int_t TBufferJSON::ReadArray(ULong_t *&l)
{
   TBufferJSON_ReadArray(ULong_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long64_t from buffer

Int_t TBufferJSON::ReadArray(Long64_t *&l)
{
   TBufferJSON_ReadArray(Long64_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong64_t from buffer

Int_t TBufferJSON::ReadArray(ULong64_t *&l)
{
   TBufferJSON_ReadArray(ULong64_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float_t from buffer

Int_t TBufferJSON::ReadArray(Float_t *&f)
{
   TBufferJSON_ReadArray(Float_t, f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double_t from buffer

Int_t TBufferJSON::ReadArray(Double_t *&d)
{
   TBufferJSON_ReadArray(Double_t, d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float16_t from buffer

Int_t TBufferJSON::ReadArrayFloat16(Float_t *&f, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadArray(Float_t, f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double32_t from buffer

Int_t TBufferJSON::ReadArrayDouble32(Double_t *&d, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadArray(Double_t, d);
}

// dummy macro to read array from json buffer
#define TBufferJSON_ReadStaticArray(vname)         \
   {                                               \
      printf("JSON::ReadStaticArray %p\n", vname); \
      if (!vname)                                  \
         return 0;                                 \
      return 1;                                    \
   }

////////////////////////////////////////////////////////////////////////////////
/// Read array of Bool_t from buffer

Int_t TBufferJSON::ReadStaticArray(Bool_t *b)
{
   TBufferJSON_ReadStaticArray(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Char_t from buffer

Int_t TBufferJSON::ReadStaticArray(Char_t *c)
{
   TBufferJSON_ReadStaticArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UChar_t from buffer

Int_t TBufferJSON::ReadStaticArray(UChar_t *c)
{
   TBufferJSON_ReadStaticArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Short_t from buffer

Int_t TBufferJSON::ReadStaticArray(Short_t *h)
{
   TBufferJSON_ReadStaticArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UShort_t from buffer

Int_t TBufferJSON::ReadStaticArray(UShort_t *h)
{
   TBufferJSON_ReadStaticArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Int_t from buffer

Int_t TBufferJSON::ReadStaticArray(Int_t *i)
{
   TBufferJSON_ReadStaticArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UInt_t from buffer

Int_t TBufferJSON::ReadStaticArray(UInt_t *i)
{
   TBufferJSON_ReadStaticArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long_t from buffer

Int_t TBufferJSON::ReadStaticArray(Long_t *l)
{
   TBufferJSON_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong_t from buffer

Int_t TBufferJSON::ReadStaticArray(ULong_t *l)
{
   TBufferJSON_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long64_t from buffer

Int_t TBufferJSON::ReadStaticArray(Long64_t *l)
{
   TBufferJSON_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong64_t from buffer

Int_t TBufferJSON::ReadStaticArray(ULong64_t *l)
{
   TBufferJSON_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float_t from buffer

Int_t TBufferJSON::ReadStaticArray(Float_t *f)
{
   TBufferJSON_ReadStaticArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double_t from buffer

Int_t TBufferJSON::ReadStaticArray(Double_t *d)
{
   TBufferJSON_ReadStaticArray(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float16_t from buffer

Int_t TBufferJSON::ReadStaticArrayFloat16(Float_t *f, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadStaticArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double32_t from buffer

Int_t TBufferJSON::ReadStaticArrayDouble32(Double_t *d, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadStaticArray(d);
}

// macro to read content of array, which not include size of array
// macro also treat situation, when instead of one single array chain
// of several elements should be produced
#define TBufferJSON_ReadFastArray(arg, cast_type, asstr)               \
   if (!arg || (n<=0)) return;                                         \
   TJSONStackObj *stack = Stack();                                     \
   if (stack && stack->fNode) {                                        \
      nlohmann::json &json = *((nlohmann::json *)stack->fNode);        \
      TArrayIndexProducer *indexes = stack->MakeReadIndexes();         \
      if (indexes) { /* at least two dims */                           \
         TArrayI &indx = indexes->GetIndices();                        \
         Int_t lastdim = indx.GetSize() - 1;                           \
         if (indexes->TotalLength() != n) Error("ReadFastArray", "Mismatch %d-dim array sizes %d %d", lastdim+1, n, (int) indexes->TotalLength()); \
         for (int cnt=0;cnt<n;++cnt) {                                 \
            nlohmann::json *elem = &json[indx[0]];                     \
            for (int k=1;k<lastdim;++k) elem = &((*elem)[indx[k]]);    \
            arg[cnt] = asstr ? elem->get<std::string>()[indx[lastdim]] : (*elem)[indx[lastdim]].get<cast_type>(); \
            indexes->NextSeparator();                                  \
         }                                                             \
         delete indexes;                                               \
      } else if (asstr) {                                              \
         std::string str = json.get<std::string>();                    \
         for (int cnt=0;cnt<n;++cnt) arg[cnt] = (cnt < (int) str.length()) ? str[cnt] : 0; \
      } else {                                                         \
         if ((int) json.size() != n) Error("ReadFastArray", "Mismatch array sizes %d %d", n, (int) json.size()); \
         for (int cnt=0;cnt<n;++cnt) arg[cnt] = json[cnt].get<cast_type>(); \
      }                                                                 \
   }

////////////////////////////////////////////////////////////////////////////////
/// read array of Bool_t from buffer

void TBufferJSON::ReadFastArray(Bool_t *b, Int_t n)
{
   TBufferJSON_ReadFastArray(b, bool, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Char_t from buffer

void TBufferJSON::ReadFastArray(Char_t *c, Int_t n)
{
   TBufferJSON_ReadFastArray(c, char, true);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Char_t from buffer

void TBufferJSON::ReadFastArrayString(Char_t *c, Int_t n)
{
   TBufferJSON_ReadFastArray(c, char, true);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of UChar_t from buffer

void TBufferJSON::ReadFastArray(UChar_t *c, Int_t n)
{
   TBufferJSON_ReadFastArray(c, unsigned, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Short_t from buffer

void TBufferJSON::ReadFastArray(Short_t *h, Int_t n)
{
   TBufferJSON_ReadFastArray(h, int, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of UShort_t from buffer

void TBufferJSON::ReadFastArray(UShort_t *h, Int_t n)
{
   TBufferJSON_ReadFastArray(h, unsigned int, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Int_t from buffer

void TBufferJSON::ReadFastArray(Int_t *i, Int_t n)
{
   TBufferJSON_ReadFastArray(i, int, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of UInt_t from buffer

void TBufferJSON::ReadFastArray(UInt_t *i, Int_t n)
{
   TBufferJSON_ReadFastArray(i, unsigned, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Long_t from buffer

void TBufferJSON::ReadFastArray(Long_t *l, Int_t n)
{
   TBufferJSON_ReadFastArray(l, Long_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of ULong_t from buffer

void TBufferJSON::ReadFastArray(ULong_t *l, Int_t n)
{
   TBufferJSON_ReadFastArray(l, ULong_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Long64_t from buffer

void TBufferJSON::ReadFastArray(Long64_t *l, Int_t n)
{
   TBufferJSON_ReadFastArray(l, Long64_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of ULong64_t from buffer

void TBufferJSON::ReadFastArray(ULong64_t *l, Int_t n)
{
   TBufferJSON_ReadFastArray(l, ULong64_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float_t from buffer

void TBufferJSON::ReadFastArray(Float_t *f, Int_t n)
{
   TBufferJSON_ReadFastArray(f, Float_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double_t from buffer

void TBufferJSON::ReadFastArray(Double_t *d, Int_t n)
{
   TBufferJSON_ReadFastArray(d, Double_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferJSON::ReadFastArrayFloat16(Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadFastArray(f, Float_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferJSON::ReadFastArrayWithFactor(Float_t *f, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   TBufferJSON_ReadFastArray(f, Float_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Float16_t from buffer

void TBufferJSON::ReadFastArrayWithNbits(Float_t *f, Int_t n, Int_t /*nbits*/)
{
   TBufferJSON_ReadFastArray(f, Float_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferJSON::ReadFastArrayDouble32(Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_ReadFastArray(d, Double_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferJSON::ReadFastArrayWithFactor(Double_t *d, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   TBufferJSON_ReadFastArray(d, Double_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// read array of Double32_t from buffer

void TBufferJSON::ReadFastArrayWithNbits(Double_t *d, Int_t n, Int_t /*nbits*/)
{
   TBufferJSON_ReadFastArray(d, Double_t, false);
}

////////////////////////////////////////////////////////////////////////////////
/// Read an array of 'n' objects from the I/O buffer.
/// Stores the objects read starting at the address 'start'.
/// The objects in the array are assume to be of class 'cl'.
/// Copied code from TBufferFile

void TBufferJSON::ReadFastArray(void *start, const TClass *cl, Int_t n, TMemberStreamer *streamer,
                                const TClass *onFileClass)
{
   if (gDebug>1)
      Info("ReadFastArray", "void* n:%d cl:%s streamer:%s", n, cl->GetName(), (streamer ? "on" : "off"));

   if (streamer) {
      Info("ReadFastArray", "(void*) Calling streamer - not handled correctly");
      streamer->SetOnFileClass(onFileClass);
      (*streamer)(*this,start,0);
      return;
   }

   int objectSize = cl->Size();
   char *obj = (char*)start;

   TJSONStackObj *stack = Stack();
   JSONObject_t topnode = stack->fNode, subnode = topnode;
   if (stack->fIndx) subnode = stack->fIndx->ExtractNode(topnode);

   TArrayIndexProducer indexes(stack->fElem, n, "");

   if (gDebug>1)
      Info("ReadFastArray", "Indexes ndim:%d totallen:%d", indexes.NumDimensions(), indexes.TotalLength());

   for (Int_t j = 0; j < n; j++, obj += objectSize) {

      stack->fNode = indexes.ExtractNode(subnode);

      JsonReadObject(obj, cl);
   }

   // restore top node - show we use stack here?
   stack->fNode = topnode;
}

////////////////////////////////////////////////////////////////////////////////
/// redefined here to avoid warning message from gcc

void TBufferJSON::ReadFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc,
                                TMemberStreamer *streamer, const TClass *onFileClass)
{
   if (gDebug>1) Info("ReadFastArray", "void** n:%d cl:%s prealloc:%s", n, cl->GetName(), (isPreAlloc ? "true" : "false"));

   if (streamer) {
      Info("ReadFastArray", "(void**) Calling streamer - not handled correctly");
      if (isPreAlloc) {
         for (Int_t j=0;j<n;j++) {
            if (!start[j]) start[j] = cl->New();
         }
      }
      streamer->SetOnFileClass(onFileClass);
      (*streamer)(*this,(void*)start,0);
      return;
   }

   TJSONStackObj *stack = Stack();
   JSONObject_t topnode = stack->fNode, subnode = topnode;
   if (stack->fIndx) subnode = stack->fIndx->ExtractNode(topnode);

   TArrayIndexProducer indexes(stack->fElem, n, "");

   for (Int_t j=0; j<n; j++) {

      stack->fNode = indexes.ExtractNode(subnode);

      if (!isPreAlloc) {
         void *old = start[j];
         start[j] = JsonReadObject(nullptr, cl);
         if (old && old!=start[j] && TStreamerInfo::CanDelete())
            ((TClass*)cl)->Destructor(old,kFALSE); // call delete and destruct
      } else {
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         JsonReadObject(start[j], cl);
      }
   }

   stack->fNode = topnode;
}

#define TJSONWriteArrayCompress(vname, arrsize, typname)                                                       \
   {                                                                                                           \
      if ((fCompact < 10) || (arrsize < 6)) {                                                                  \
         fValue.Append("[");                                                                                   \
         for (Int_t indx = 0; indx < arrsize; indx++) {                                                        \
            if (indx > 0)                                                                                      \
               fValue.Append(fArraySepar.Data());                                                              \
            JsonWriteBasic(vname[indx]);                                                                       \
         }                                                                                                     \
         fValue.Append("]");                                                                                   \
      } else {                                                                                                 \
         fValue.Append("{");                                                                                   \
         fValue.Append(TString::Format("\"$arr\":\"%s\"%s\"len\":%d", typname, fArraySepar.Data(), arrsize));  \
         Int_t aindx(0), bindx(arrsize);                                                                       \
         while ((aindx < arrsize) && (vname[aindx] == 0))                                                      \
            aindx++;                                                                                           \
         while ((aindx < bindx) && (vname[bindx - 1] == 0))                                                    \
            bindx--;                                                                                           \
         if (aindx < bindx) {                                                                                  \
            TString suffix("");                                                                                \
            Int_t p(aindx), suffixcnt(-1), lastp(0);                                                           \
            while (p < bindx) {                                                                                \
               if (vname[p] == 0) {                                                                            \
                  p++;                                                                                         \
                  continue;                                                                                    \
               }                                                                                               \
               Int_t p0(p++), pp(0), nsame(1);                                                                 \
               if (fCompact < 20) {                                                                            \
                  pp = bindx;                                                                                  \
                  p = bindx + 1;                                                                               \
                  nsame = 0;                                                                                   \
               }                                                                                               \
               for (; p <= bindx; ++p) {                                                                       \
                  if ((p < bindx) && (vname[p] == vname[p - 1])) {                                             \
                     nsame++;                                                                                  \
                     continue;                                                                                 \
                  }                                                                                            \
                  if (vname[p - 1] == 0) {                                                                     \
                     if (nsame > 9) {                                                                          \
                        nsame = 0;                                                                             \
                        break;                                                                                 \
                     }                                                                                         \
                  } else if (nsame > 5) {                                                                      \
                     if (pp) {                                                                                 \
                        p = pp;                                                                                \
                        nsame = 0;                                                                             \
                     } else                                                                                    \
                        pp = p;                                                                                \
                     break;                                                                                    \
                  }                                                                                            \
                  pp = p;                                                                                      \
                  nsame = 1;                                                                                   \
               }                                                                                               \
               if (pp <= p0)                                                                                   \
                  continue;                                                                                    \
               if (++suffixcnt > 0)                                                                            \
                  suffix.Form("%d", suffixcnt);                                                                \
               if (p0 != lastp)                                                                                \
                  fValue.Append(TString::Format("%s\"p%s\":%d", fArraySepar.Data(), suffix.Data(), p0));       \
               lastp = pp; /* remember cursor, it may be the same */                                           \
               fValue.Append(TString::Format("%s\"v%s\":", fArraySepar.Data(), suffix.Data()));                \
               if ((nsame > 1) || (pp - p0 == 1)) {                                                            \
                  JsonWriteBasic(vname[p0]);                                                                   \
                  if (nsame > 1)                                                                               \
                     fValue.Append(TString::Format("%s\"n%s\":%d", fArraySepar.Data(), suffix.Data(), nsame)); \
               } else {                                                                                        \
                  fValue.Append("[");                                                                          \
                  for (Int_t indx = p0; indx < pp; indx++) {                                                   \
                     if (indx > p0)                                                                            \
                        fValue.Append(fArraySepar.Data());                                                     \
                     JsonWriteBasic(vname[indx]);                                                              \
                  }                                                                                            \
                  fValue.Append("]");                                                                          \
               }                                                                                               \
            }                                                                                                  \
         }                                                                                                     \
         fValue.Append("}");                                                                                   \
      }                                                                                                        \
   }

// macro call TBufferJSON method without typname
#define TJSONWriteConstChar(vname, arrsize, typname) \
   {                                                 \
      JsonWriteConstChar(vname, arrsize);            \
   }

// macro to write array, which include size
#define TBufferJSON_WriteArray(vname, typname)    \
   {                                              \
      TJSONPushValue();                           \
      TJSONWriteArrayCompress(vname, n, typname); \
   }

////////////////////////////////////////////////////////////////////////////////
/// Write array of Bool_t to buffer

void TBufferJSON::WriteArray(const Bool_t *b, Int_t n)
{
   TBufferJSON_WriteArray(b, "Bool");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Char_t to buffer

void TBufferJSON::WriteArray(const Char_t *c, Int_t n)
{
   TBufferJSON_WriteArray(c, "Int8");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UChar_t to buffer

void TBufferJSON::WriteArray(const UChar_t *c, Int_t n)
{
   TBufferJSON_WriteArray(c, "Uint8");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Short_t to buffer

void TBufferJSON::WriteArray(const Short_t *h, Int_t n)
{
   TBufferJSON_WriteArray(h, "Int16");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UShort_t to buffer

void TBufferJSON::WriteArray(const UShort_t *h, Int_t n)
{
   TBufferJSON_WriteArray(h, "Uint16");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Int_ to buffer

void TBufferJSON::WriteArray(const Int_t *i, Int_t n)
{
   TBufferJSON_WriteArray(i, "Int32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UInt_t to buffer

void TBufferJSON::WriteArray(const UInt_t *i, Int_t n)
{
   TBufferJSON_WriteArray(i, "Uint32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long_t to buffer

void TBufferJSON::WriteArray(const Long_t *l, Int_t n)
{
   TBufferJSON_WriteArray(l, "Int64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong_t to buffer

void TBufferJSON::WriteArray(const ULong_t *l, Int_t n)
{
   TBufferJSON_WriteArray(l, "Uint64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long64_t to buffer

void TBufferJSON::WriteArray(const Long64_t *l, Int_t n)
{
   TBufferJSON_WriteArray(l, "Int64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong64_t to buffer

void TBufferJSON::WriteArray(const ULong64_t *l, Int_t n)
{
   TBufferJSON_WriteArray(l, "Uint64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float_t to buffer

void TBufferJSON::WriteArray(const Float_t *f, Int_t n)
{
   TBufferJSON_WriteArray(f, "Float32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double_t to buffer

void TBufferJSON::WriteArray(const Double_t *d, Int_t n)
{
   TBufferJSON_WriteArray(d, "Float64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float16_t to buffer

void TBufferJSON::WriteArrayFloat16(const Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_WriteArray(f, "Float32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double32_t to buffer

void TBufferJSON::WriteArrayDouble32(const Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_WriteArray(d, "Float64");
}

// write array without size attribute
// macro also treat situation, when instead of one single array
// chain of several elements should be produced
#define TBufferJSON_WriteFastArray(vname, method, typname)                             \
   {                                                                                   \
      TJSONPushValue();                                                                \
      if (n <= 0) { /*fJsonrCnt++;*/                                                   \
         fValue.Append("[]");                                                          \
         return;                                                                       \
      }                                                                                \
      TStreamerElement *elem = Stack(0)->fElem;                                        \
      if ((elem != 0) && (elem->GetArrayDim() > 1) && (elem->GetArrayLength() == n)) { \
         TArrayI indexes(elem->GetArrayDim() - 1);                                     \
         indexes.Reset(0);                                                             \
         Int_t cnt = 0, shift = 0, len = elem->GetMaxIndex(indexes.GetSize());         \
         while (cnt >= 0) {                                                            \
            if (indexes[cnt] >= elem->GetMaxIndex(cnt)) {                              \
               fValue.Append("]");                                                     \
               indexes[cnt--] = 0;                                                     \
               if (cnt >= 0)                                                           \
                  indexes[cnt]++;                                                      \
               continue;                                                               \
            }                                                                          \
            fValue.Append(indexes[cnt] == 0 ? "[" : fArraySepar.Data());               \
            if (++cnt == indexes.GetSize()) {                                          \
               method((vname + shift), len, typname);                                  \
               indexes[--cnt]++;                                                       \
               shift += len;                                                           \
            }                                                                          \
         }                                                                             \
      } else {                                                                         \
         method(vname, n, typname);                                                    \
      }                                                                                \
   }

////////////////////////////////////////////////////////////////////////////////
/// Write array of Bool_t to buffer

void TBufferJSON::WriteFastArray(const Bool_t *b, Int_t n)
{
   TBufferJSON_WriteFastArray(b, TJSONWriteArrayCompress, "Bool");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Char_t to buffer

void TBufferJSON::WriteFastArray(const Char_t *c, Int_t n)
{
   TBufferJSON_WriteFastArray(c, TJSONWriteConstChar, "Int8");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Char_t to buffer

void TBufferJSON::WriteFastArrayString(const Char_t *c, Int_t n)
{
   TBufferJSON_WriteFastArray(c, TJSONWriteConstChar, "Int8");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UChar_t to buffer

void TBufferJSON::WriteFastArray(const UChar_t *c, Int_t n)
{
   TBufferJSON_WriteFastArray(c, TJSONWriteArrayCompress, "Uint8");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Short_t to buffer

void TBufferJSON::WriteFastArray(const Short_t *h, Int_t n)
{
   TBufferJSON_WriteFastArray(h, TJSONWriteArrayCompress, "Int16");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UShort_t to buffer

void TBufferJSON::WriteFastArray(const UShort_t *h, Int_t n)
{
   TBufferJSON_WriteFastArray(h, TJSONWriteArrayCompress, "Uint16");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Int_t to buffer

void TBufferJSON::WriteFastArray(const Int_t *i, Int_t n)
{
   TBufferJSON_WriteFastArray(i, TJSONWriteArrayCompress, "Int32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UInt_t to buffer

void TBufferJSON::WriteFastArray(const UInt_t *i, Int_t n)
{
   TBufferJSON_WriteFastArray(i, TJSONWriteArrayCompress, "Uint32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long_t to buffer

void TBufferJSON::WriteFastArray(const Long_t *l, Int_t n)
{
   TBufferJSON_WriteFastArray(l, TJSONWriteArrayCompress, "Int64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong_t to buffer

void TBufferJSON::WriteFastArray(const ULong_t *l, Int_t n)
{
   TBufferJSON_WriteFastArray(l, TJSONWriteArrayCompress, "Uint64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long64_t to buffer

void TBufferJSON::WriteFastArray(const Long64_t *l, Int_t n)
{
   TBufferJSON_WriteFastArray(l, TJSONWriteArrayCompress, "Int64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong64_t to buffer

void TBufferJSON::WriteFastArray(const ULong64_t *l, Int_t n)
{
   TBufferJSON_WriteFastArray(l, TJSONWriteArrayCompress, "Uint64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float_t to buffer

void TBufferJSON::WriteFastArray(const Float_t *f, Int_t n)
{
   TBufferJSON_WriteFastArray(f, TJSONWriteArrayCompress, "Float32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double_t to buffer

void TBufferJSON::WriteFastArray(const Double_t *d, Int_t n)
{
   TBufferJSON_WriteFastArray(d, TJSONWriteArrayCompress, "Float64");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float16_t to buffer

void TBufferJSON::WriteFastArrayFloat16(const Float_t *f, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_WriteFastArray(f, TJSONWriteArrayCompress, "Float32");
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double32_t to buffer

void TBufferJSON::WriteFastArrayDouble32(const Double_t *d, Int_t n, TStreamerElement * /*ele*/)
{
   TBufferJSON_WriteFastArray(d, TJSONWriteArrayCompress, "Float64");
}

////////////////////////////////////////////////////////////////////////////////
/// Recall TBuffer function to avoid gcc warning message

void TBufferJSON::WriteFastArray(void *start, const TClass *cl, Int_t n, TMemberStreamer *streamer)
{
   if (gDebug > 2)
      Info("WriteFastArray", "void *start cl %s n %d streamer %p", cl ? cl->GetName() : "---", n, streamer);

   if (streamer) {
      JsonDisablePostprocessing();
      (*streamer)(*this, start, 0);
      return;
   }

   if (n < 0) {
      // special handling of empty StreamLoop
      AppendOutput("null");
      JsonDisablePostprocessing();
   } else {

      char *obj = (char *)start;
      if (!n)
         n = 1;
      int size = cl->Size();

      TArrayIndexProducer indexes(Stack(0)->fElem, n, fArraySepar.Data());

      if (indexes.IsArray()) {
         JsonDisablePostprocessing();
         AppendOutput(indexes.GetBegin());
      }

      for (Int_t j = 0; j < n; j++, obj += size) {

         if (j > 0)
            AppendOutput(indexes.NextSeparator());

         JsonWriteObject(obj, cl, kFALSE);

         if (indexes.IsArray() && (fValue.Length() > 0)) {
            AppendOutput(fValue.Data());
            fValue.Clear();
         }
      }

      if (indexes.IsArray())
         AppendOutput(indexes.GetEnd());
   }

   if (Stack(0)->fIndx)
      AppendOutput(Stack(0)->fIndx->NextSeparator());
}

////////////////////////////////////////////////////////////////////////////////
/// Recall TBuffer function to avoid gcc warning message

Int_t TBufferJSON::WriteFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer)
{
   if (gDebug > 2)
      Info("WriteFastArray", "void **startp cl %s n %d streamer %p", cl->GetName(), n, streamer);

   if (streamer) {
      JsonDisablePostprocessing();
      (*streamer)(*this, (void *)start, 0);
      return 0;
   }

   if (n <= 0)
      return 0;

   Int_t res = 0;

   TArrayIndexProducer indexes(Stack(0)->fElem, n, fArraySepar.Data());

   if (indexes.IsArray()) {
      JsonDisablePostprocessing();
      AppendOutput(indexes.GetBegin());
   }

   for (Int_t j = 0; j < n; j++) {

      if (j > 0)
         AppendOutput(indexes.NextSeparator());

      if (!isPreAlloc) {
         res |= WriteObjectAny(start[j], cl);
      } else {
         if (!start[j])
            start[j] = ((TClass *)cl)->New();
         // ((TClass*)cl)->Streamer(start[j],*this);
         JsonWriteObject(start[j], cl, kFALSE);
      }

      if (indexes.IsArray() && (fValue.Length() > 0)) {
         AppendOutput(fValue.Data());
         fValue.Clear();
      }
   }

   if (indexes.IsArray())
      AppendOutput(indexes.GetEnd());

   if (Stack(0)->fIndx)
      AppendOutput(Stack(0)->fIndx->NextSeparator());

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// stream object to/from buffer

void TBufferJSON::StreamObject(void *obj, const std::type_info &typeinfo, const TClass * /* onFileClass */)
{
   StreamObject(obj, TClass::GetClass(typeinfo));
}

////////////////////////////////////////////////////////////////////////////////
/// stream object to/from buffer

void TBufferJSON::StreamObject(void *obj, const char *className, const TClass * /* onFileClass */)
{
   StreamObject(obj, TClass::GetClass(className));
}

void TBufferJSON::StreamObject(TObject *obj)
{
   // stream object to/from buffer

   StreamObject(obj, obj ? obj->IsA() : TObject::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// stream object to/from buffer

void TBufferJSON::StreamObject(void *obj, const TClass *cl, const TClass * /* onfileClass */)
{
   if (gDebug > 3)
      Info("StreamObject", "Class: %s", (cl ? cl->GetName() : "none"));

   if (IsWriting())
      JsonWriteObject(obj, cl);
   else
      JsonReadObject(obj, cl);
}


#define JsonReadBasic(arg, cast_type)                              \
   TJSONStackObj *stack = Stack();                                 \
   if (stack && stack->fNode)                                      \
      arg = ((nlohmann::json *)stack->fNode)->get<cast_type>();

// read basic, but first check if values prepend
#define JsonReadBasicMore(arg, cast_type)                          \
   TJSONStackObj *stack = Stack();                                 \
   if (stack && (stack->fValues.GetLast() >= 0)) {                 \
      TObject *str = stack->fValues.Last();                        \
      arg = nlohmann::json::parse(str->GetName()).get<cast_type>(); \
      stack->fValues.Remove(str);                                  \
      delete str;                                                  \
   } else if (stack && stack->fNode)                               \
      arg = ((nlohmann::json *)stack->fNode)->get<cast_type>();



#define JsonReadString(arg)                                        \
   TJSONStackObj *stack = Stack();                                 \
   if (stack && stack->fNode) {                                    \
      nlohmann::json *json = ((nlohmann::json *)stack->fNode);     \
      if (stack->fStlIndx >= 0)                                    \
         json = &(json->at(stack->fStlIndx++));                    \
      arg = json->get<std::string>();                              \
   }

////////////////////////////////////////////////////////////////////////////////
/// Reads Bool_t value from buffer

void TBufferJSON::ReadBool(Bool_t &val)
{
   JsonReadBasic(val, Bool_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Char_t value from buffer

void TBufferJSON::ReadChar(Char_t &val)
{
   TJSONStackObj *stack = Stack();
   if (stack && stack->fNode) {
      nlohmann::json *node = ((nlohmann::json *)stack->fNode);
      if (stack->fElem &&
          (stack->fElem->GetType() > TStreamerInfo::kOffsetP) &&
          (stack->fElem->GetType() < TStreamerInfo::kOffsetP + 20)) {
             val = node->is_array() || node->is_string() ? 1 : 0;
      } else {
         val = node->get<Char_t>();
      }
   } else {
      val = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UChar_t value from buffer

void TBufferJSON::ReadUChar(UChar_t &val)
{
   JsonReadBasic(val, UChar_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Short_t value from buffer

void TBufferJSON::ReadShort(Short_t &val)
{
   JsonReadBasic(val, Short_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UShort_t value from buffer

void TBufferJSON::ReadUShort(UShort_t &val)
{
   JsonReadBasic(val, UShort_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Int_t value from buffer

void TBufferJSON::ReadInt(Int_t &val)
{
   JsonReadBasicMore(val, Int_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UInt_t value from buffer

void TBufferJSON::ReadUInt(UInt_t &val)
{
   JsonReadBasic(val, UInt_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Long_t value from buffer

void TBufferJSON::ReadLong(Long_t &val)
{
   JsonReadBasic(val, Long_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads ULong_t value from buffer

void TBufferJSON::ReadULong(ULong_t &val)
{
   JsonReadBasic(val, ULong_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Long64_t value from buffer

void TBufferJSON::ReadLong64(Long64_t &val)
{
   JsonReadBasic(val, Long64_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads ULong64_t value from buffer

void TBufferJSON::ReadULong64(ULong64_t &val)
{
   JsonReadBasic(val, ULong64_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Float_t value from buffer

void TBufferJSON::ReadFloat(Float_t &val)
{
   JsonReadBasic(val, Float_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Double_t value from buffer

void TBufferJSON::ReadDouble(Double_t &val)
{
   JsonReadBasic(val, Double_t);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads array of characters from buffer

void TBufferJSON::ReadCharP(Char_t *)
{
   Error("ReadCharP", "Not implemented");
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a TString

void TBufferJSON::ReadTString(TString &val)
{
   std::string str;
   JsonReadString(str);
   val = str.c_str();
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a std::string

void TBufferJSON::ReadStdString(std::string *val)
{
   JsonReadString(*val);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads a char* string

void TBufferJSON::ReadCharStar(char *&s)
{
   std::string str;
   JsonReadString(str);

   if (s) {
     delete [] s;
     s = nullptr;
   }

   std::size_t nch = str.length();
   if (nch > 0) {
      s = new char[nch+1];
      memcpy(s, str.c_str(), nch);
      s[nch] = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Bool_t value to buffer

void TBufferJSON::WriteBool(Bool_t b)
{
   TJSONPushValue();

   JsonWriteBasic(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Char_t value to buffer

void TBufferJSON::WriteChar(Char_t c)
{
   TJSONPushValue();

   JsonWriteBasic(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UChar_t value to buffer

void TBufferJSON::WriteUChar(UChar_t c)
{
   TJSONPushValue();

   JsonWriteBasic(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Short_t value to buffer

void TBufferJSON::WriteShort(Short_t h)
{
   TJSONPushValue();

   JsonWriteBasic(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UShort_t value to buffer

void TBufferJSON::WriteUShort(UShort_t h)
{
   TJSONPushValue();

   JsonWriteBasic(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Int_t value to buffer

void TBufferJSON::WriteInt(Int_t i)
{
   TJSONPushValue();

   JsonWriteBasic(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UInt_t value to buffer

void TBufferJSON::WriteUInt(UInt_t i)
{
   TJSONPushValue();

   JsonWriteBasic(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Long_t value to buffer

void TBufferJSON::WriteLong(Long_t l)
{
   TJSONPushValue();

   JsonWriteBasic(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes ULong_t value to buffer

void TBufferJSON::WriteULong(ULong_t l)
{
   TJSONPushValue();

   JsonWriteBasic(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Long64_t value to buffer

void TBufferJSON::WriteLong64(Long64_t l)
{
   TJSONPushValue();

   JsonWriteBasic(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes ULong64_t value to buffer

void TBufferJSON::WriteULong64(ULong64_t l)
{
   TJSONPushValue();

   JsonWriteBasic(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Float_t value to buffer

void TBufferJSON::WriteFloat(Float_t f)
{
   TJSONPushValue();

   JsonWriteBasic(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Double_t value to buffer

void TBufferJSON::WriteDouble(Double_t d)
{
   TJSONPushValue();

   JsonWriteBasic(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes array of characters to buffer

void TBufferJSON::WriteCharP(const Char_t *c)
{
   TJSONPushValue();

   JsonWriteConstChar(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes a TString

void TBufferJSON::WriteTString(const TString &s)
{
   TJSONPushValue();

   JsonWriteConstChar(s.Data(), s.Length());
}

////////////////////////////////////////////////////////////////////////////////
/// Writes a std::string

void TBufferJSON::WriteStdString(const std::string *s)
{
   TJSONPushValue();

   if (s)
      JsonWriteConstChar(s->c_str(), s->length());
   else
      JsonWriteConstChar("", 0);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes a char*

void TBufferJSON::WriteCharStar(char *s)
{
   TJSONPushValue();

   JsonWriteConstChar(s);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Char_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Char_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Short_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Short_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%hd", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Int_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Int_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Long_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Long_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%ld", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Long64_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Long64_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), FLong64, value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// method compress float string, excluding exp and/or move float point
///  - 1.000000e-01 -> 0.1
///  - 3.750000e+00 -> 3.75
///  - 3.750000e-03 -> 0.00375
///  - 3.750000e-04 -> 3.75e-4
///  - 1.100000e-10 -> 1.1e-10

void TBufferJSON::CompactFloatString(char *sbuf, unsigned len)
{
   char *pnt = 0, *exp = 0, *lastdecimal = 0, *s = sbuf;
   bool negative_exp = false;
   int power = 0;
   while (*s && --len) {
      switch (*s) {
      case '.': pnt = s; break;
      case 'E':
      case 'e': exp = s; break;
      case '-':
         if (exp)
            negative_exp = true;
         break;
      case '+': break;
      default: // should be digits from '0' to '9'
         if ((*s < '0') || (*s > '9'))
            return;
         if (exp)
            power = power * 10 + (*s - '0');
         else if (pnt && *s != '0')
            lastdecimal = s;
         break;
      }
      ++s;
   }
   if (*s)
      return; // if end-of-string was not found

   if (!exp) {
      // value without exponent like 123.4569000
      if (pnt) {
         if (lastdecimal)
            *(lastdecimal + 1) = 0;
         else
            *pnt = 0;
      }
   } else if (power == 0) {
      if (lastdecimal)
         *(lastdecimal + 1) = 0;
      else if (pnt)
         *pnt = 0;
   } else if (!negative_exp && pnt && exp && (exp - pnt > power)) {
      // this is case of value 1.23000e+02
      // we can move point and exclude exponent easily
      for (int cnt = 0; cnt < power; ++cnt) {
         char tmp = *pnt;
         *pnt = *(pnt + 1);
         *(++pnt) = tmp;
      }
      if (lastdecimal && (pnt < lastdecimal))
         *(lastdecimal + 1) = 0;
      else
         *pnt = 0;
   } else if (negative_exp && pnt && exp && (power < (s - exp))) {
      // this is small negative exponent like 1.2300e-02
      if (!lastdecimal)
         lastdecimal = pnt;
      *(lastdecimal + 1) = 0;
      // copy most significant digit on the point place
      *pnt = *(pnt - 1);

      for (char *pos = lastdecimal + 1; pos >= pnt; --pos)
         *(pos + power) = *pos;
      *(pnt - 1) = '0';
      *pnt = '.';
      for (int cnt = 1; cnt < power; ++cnt)
         *(pnt + cnt) = '0';
   } else if (pnt && exp) {
      // keep exponent, but non-significant zeros
      if (lastdecimal)
         pnt = lastdecimal + 1;
      // copy exponent sign
      *pnt++ = *exp++;
      if (*exp == '+')
         ++exp;
      else if (*exp == '-')
         *pnt++ = *exp++;
      // exclude zeros in the begin of exponent
      while (*exp == '0')
         ++exp;
      while (*exp)
         *pnt++ = *exp++;
      *pnt = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// converts Float_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Float_t value)
{
   char buf[200];
   // this is just check if float value looks like integer and can be stored in more compact form
   // default should be storage with fgFloatFmt, which will be optimized afterwards anyway
   if ((value == std::nearbyint(value)) && (std::abs(value) < 1e15)) {
      snprintf(buf, sizeof(buf), "%1.0f", value);
   } else {
      snprintf(buf, sizeof(buf), fgFloatFmt, value);
      CompactFloatString(buf, sizeof(buf));
   }
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Double_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Double_t value)
{
   char buf[200];
   // this is just check if float value looks like integer and can be stored in more compact form
   // default should be storage with fgDoubleFmt, which will be optimized afterwards anyway
   if ((value == std::nearbyint(value)) && (std::abs(value) < 1e25)) {
      snprintf(buf, sizeof(buf), "%1.0f", value);
   } else {
      snprintf(buf, sizeof(buf), fgDoubleFmt, value);
      CompactFloatString(buf, sizeof(buf));
   }
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Bool_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(Bool_t value)
{
   fValue.Append(value ? "true" : "false");
}

////////////////////////////////////////////////////////////////////////////////
/// converts UChar_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(UChar_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts UShort_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(UShort_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%hu", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts UInt_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(UInt_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts ULong_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(ULong_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%lu", value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// converts ULong64_t to string and add to json value buffer

void TBufferJSON::JsonWriteBasic(ULong64_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), FULong64, value);
   fValue.Append(buf);
}

////////////////////////////////////////////////////////////////////////////////
/// writes string value, processing all kind of special characters

void TBufferJSON::JsonWriteConstChar(const char *value, Int_t len)
{
   if (value == 0) {

      fValue.Append("\"\"");

   } else {

      fValue.Append("\"");

      if (len < 0)
         len = strlen(value);

      for (Int_t n = 0; n < len; n++) {
         char c = value[n];
         if (c == 0)
            break;
         switch (c) {
         case '\n': fValue.Append("\\n"); break;
         case '\t': fValue.Append("\\t"); break;
         case '\"': fValue.Append("\\\""); break;
         case '\\': fValue.Append("\\\\"); break;
         case '\b': fValue.Append("\\b"); break;
         case '\f': fValue.Append("\\f"); break;
         case '\r': fValue.Append("\\r"); break;
         case '/': fValue.Append("\\/"); break;
         default:
            if ((c > 31) && (c < 127))
               fValue.Append(c);
            else
               fValue.Append(TString::Format("\\u%04x", (unsigned)c));
         }
      }

      fValue.Append("\"");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// set printf format for float/double members, default "%e"
/// to change format only for doubles, use SetDoubleFormat

void TBufferJSON::SetFloatFormat(const char *fmt)
{
   if (fmt == 0)
      fmt = "%e";
   fgFloatFmt = fmt;
   fgDoubleFmt = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// return current printf format for float members, default "%e"

const char *TBufferJSON::GetFloatFormat()
{
   return fgFloatFmt;
}

////////////////////////////////////////////////////////////////////////////////
/// set printf format for double members, default "%.14e"
/// use it after SetFloatFormat, which also overwrites format for doubles

void TBufferJSON::SetDoubleFormat(const char *fmt)
{
   if (fmt == 0)
      fmt = "%.14e";
   fgDoubleFmt = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// return current printf format for double members, default "%.14e"

const char *TBufferJSON::GetDoubleFormat()
{
   return fgDoubleFmt;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.
/// The collection needs to be a split TClonesArray or a split vector of pointers.

Int_t TBufferJSON::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, obj);
         (*iter)(*this, obj);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, obj);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.
/// The collection needs to be a split TClonesArray or a split vector of pointers.

Int_t TBufferJSON::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                       void *end_collection)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(
            *this, *(char **)start_collection); // Warning: This limits us to TClonesArray and vector of pointers.
         (*iter)(*this, start_collection, end_collection);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.

Int_t TBufferJSON::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                 void *end_collection)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   TStreamerInfoActions::TLoopConfiguration *loopconfig = sequence.fLoopConfig;
   if (gDebug) {

      // Get the address of the first item for the PrintDebug.
      // (Performance is not essential here since we are going to print to
      // the screen anyway).
      void *arr0 = loopconfig->GetFirstAddress(start_collection, end_collection);
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, arr0);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }
   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for JSON streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }
   }
   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to TStreamerInfo::WriteBufferClones.

Int_t TBufferJSON::WriteClones(TClonesArray *a, Int_t /*nobjects*/)
{
   Info("WriteClones", "Not yet tested");

   if (a != 0)
      JsonStreamCollection(a, a->IsA());

   return 0;
}

namespace {
struct DynamicType {
   // Helper class to enable typeid on any address
   // Used in code similar to:
   //    typeid( * (DynamicType*) void_ptr );
   virtual ~DynamicType() {}
};
}

////////////////////////////////////////////////////////////////////////////////
/// Write object to I/O buffer.
/// This function assumes that the value in 'obj' is the value stored in
/// a pointer to a "ptrClass". The actual type of the object pointed to
/// can be any class derived from "ptrClass".
/// Return:
///  - 0: failure
///  - 1: success
///  - 2: truncated success (i.e actual class is missing. Only ptrClass saved.)
///
/// If 'cacheReuse' is true (default) upon seeing an object address a second time,
/// we record the offset where its was written the first time rather than streaming
/// the object a second time.
/// If 'cacheReuse' is false, we always stream the object.  This allows the (re)use
/// of temporary object to store different data in the same buffer.

Int_t TBufferJSON::WriteObjectAny(const void *obj, const TClass *ptrClass, Bool_t cacheReuse /* = kTRUE */)
{
   if (!obj) {
      WriteObjectClass(0, 0, kTRUE);
      return 1;
   }

   if (!ptrClass) {
      Error("WriteObjectAny", "ptrClass argument may not be 0");
      return 0;
   }

   TClass *clActual = ptrClass->GetActualClass(obj);

   if (clActual == 0) {
      // The ptrClass is a class with a virtual table and we have no
      // TClass with the actual type_info in memory.

      DynamicType *d_ptr = (DynamicType *)obj;
      Warning("WriteObjectAny", "An object of type %s (from type_info) passed through a %s pointer was truncated (due "
                                "a missing dictionary)!!!",
              typeid(*d_ptr).name(), ptrClass->GetName());
      WriteObjectClass(obj, ptrClass, cacheReuse);
      return 2;
   } else if (clActual && (clActual != ptrClass)) {
      const char *temp = (const char *)obj;
      temp -= clActual->GetBaseClassOffset(ptrClass);
      WriteObjectClass(temp, clActual, cacheReuse);
      return 1;
   } else {
      WriteObjectClass(obj, ptrClass, cacheReuse);
      return 1;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Function called by the Streamer functions to serialize object at p
/// to buffer b. The optional argument info may be specified to give an
/// alternative StreamerInfo instead of using the default StreamerInfo
/// automatically built from the class definition.
/// For more information, see class TStreamerInfo.

Int_t TBufferJSON::WriteClassBuffer(const TClass *cl, void *pointer)
{

   // build the StreamerInfo if first time for the class
   TStreamerInfo *sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
   if (sinfo == 0) {
      // Have to be sure between the check and the taking of the lock if the current streamer has changed
      R__LOCKGUARD(gInterpreterMutex);
      sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
      if (sinfo == 0) {
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
         const_cast<TClass *>(cl)->SetCurrentStreamerInfo(sinfo);
         const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
         if (gDebug > 0)
            printf("Creating StreamerInfo for class: %s, version: %d\n", cl->GetName(), cl->GetClassVersion());
         sinfo->Build();
      }
   } else if (!sinfo->IsCompiled()) {
      R__LOCKGUARD(gInterpreterMutex);
      // Redo the test in case we have been victim of a data race on fIsCompiled.
      if (!sinfo->IsCompiled()) {
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   // write the class version number and reserve space for the byte count
   // UInt_t R__c = WriteVersion(cl, kTRUE);

   // NOTE: In the future Philippe wants this to happen via a custom action
   TagStreamerInfo(sinfo);
   ApplySequence(*(sinfo->GetWriteTextActions()), (char *)pointer);

   // write the byte count at the start of the buffer
   // SetByteCount(R__c, kTRUE);

   if (gDebug > 2)
      Info("WriteClassBuffer", "class: %s version %d done", cl->GetName(), cl->GetClassVersion());
   return 0;
}
