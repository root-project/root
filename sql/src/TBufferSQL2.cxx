// @(#)root/net:$Name:  $:$Id: TBufferSQL2.cxx,v 1.5 2005/12/01 16:30:43 pcanal Exp $
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//________________________________________________________________________
//
// Class for serializing/deserializing object to/from SQL data base.
// It redefines most of TBuffer class function to convert simple types,
// array of simple types and objects to/from TSQLStructure objects.
// TBufferSQL2 class uses streaming mechanism, provided by ROOT system,
// therefore most of ROOT and user classes can be stored. There are
// limitations for complex objects like TTree, TClonesArray, TDirectory and
// few other, which can not be converted to SQL (yet).
//________________________________________________________________________

#include "TBufferSQL2.h"

#include "TObjArray.h"
#include "TROOT.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TExMap.h"
#include "TMethodCall.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TMemberStreamer.h"
#include "TStreamer.h"
#include "Riostream.h"

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TSQLStructure.h"
#include "TSQLObjectData.h"
#include "TSQLFile.h"
#include "TSQLClassInfo.h"

ClassImp(TBufferSQL2);

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2() :
   TBuffer(),
   fSQL(0)
{
   // Default constructor
}

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2(TBuffer::EMode mode) :
   TBuffer(mode),
   fSQL(0),
   fStructure(0),
   fStk(0),
   fObjMap(0),
   fIdArray(0),
   fErrorFlag(0),
   fExpectedChain(kFALSE),
   fCompressLevel(0),
   fReadVersionBuffer(-1),
   fObjIdCounter(1),
   fIgnoreVerification(kFALSE)
{
   // Creates buffer object to serailize/deserialize data to/from sql.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   SetParent(0);
   SetBit(kCannotHandleMemberWiseStreaming);
}

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2(TBuffer::EMode mode, TSQLFile* file) :
   TBuffer(mode),
   fSQL(0),
   fStructure(0),
   fStk(0),
   fObjMap(0),
   fIdArray(0),
   fErrorFlag(0),
   fExpectedChain(kFALSE),
   fCompressLevel(0),
   fReadVersionBuffer(-1),
   fObjIdCounter(1)
{
   // Creates buffer object to serailize/deserialize data to/from sql.
   // This constructor should be used, if data from buffer supposed to be stored in file.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   fBufSize = 1000000000;

   // for TClonesArray recognize if this is special case
   SetBit(kCannotHandleMemberWiseStreaming);

   SetParent(file);
   fSQL = file;
   if (file!=0)
      SetCompressionLevel(file->GetCompressionLevel());
}

//______________________________________________________________________________
TBufferSQL2::~TBufferSQL2()
{
   // destroy sql buffer
   if (fObjMap) delete fObjMap;
   if (fIdArray) delete fIdArray;

   if (fStructure!=0) {
      delete fStructure;
      fStructure = 0;
   }
}

//______________________________________________________________________________
TSQLStructure* TBufferSQL2::SqlWrite(const TObject* obj, Int_t objid)
{
   // Convert object, derived from TObject class to sql structures
   // Return pointer on created TSQLStructure
   // TSQLStructure object will be owned by TBufferSQL2

   if (obj==0) return SqlWrite(0,0, objid);
   else return SqlWrite(obj, obj->IsA(), objid);
}

//______________________________________________________________________________
TSQLStructure* TBufferSQL2::SqlWrite(const void* obj, const TClass* cl, Int_t objid)
{
   // Convert object of any class to sql structures
   // Return pointer on created TSQLStructure
   // TSQLStructure object will be owned by TBufferSQL2

   fErrorFlag = 0;

   fStructure = 0;

   fObjIdCounter = objid;

   SqlWriteObject(obj, cl);

   if (gDebug>3)
      if (fStructure!=0) {
         cout << "==== Printout of Sql structures ===== " << endl;
         fStructure->Print("*");
         cout << "=========== End printout ============ " << endl;
      }

   return fStructure;
}

//______________________________________________________________________________
TObject* TBufferSQL2::SqlRead(Int_t objid)
{
   // Recreate object from sql structure.
   // Return pointer to read object.
   // If object class is not inherited from TObject,
   // object is deleted and function return 0

   TClass* cl = 0;
   void* obj = SqlReadAny(objid, &cl);

   if ((cl!=0) && !cl->InheritsFrom(TObject::Class())) {
      cl->Destructor(obj);
      obj = 0;
   }

   return (TObject*) obj;
}

//______________________________________________________________________________
void* TBufferSQL2::SqlReadAny(Int_t objid, TClass** cl)
{
   // Recreate object from sql structure.
   // Return pointer to read object.
   // if (cl!=0) returns pointer to class of object

   if (cl) *cl = 0;
   if (fSQL==0) return 0;

   fCurrentData=0;
   fErrorFlag = 0;

   fReadVersionBuffer = -1;

   return SqlReadObjectDirect(0, cl, objid);
}

//______________________________________________________________________________
void TBufferSQL2::WriteObject(const TObject *obj)
{
   // Convert object into sql structures.
   // !!! Should be used only by TBufferSQL2 itself.
   // Use SqlWrite() functions to convert your object to sql
   // Redefined here to avoid gcc 3.x warning

   TBuffer::WriteObject(obj);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::ProcessPointer(const void* ptr, Int_t& objid)
{
   // Return SqlObjectId_Null, if ptr is null or
   // id of object which was already stored in buffer

   if (ptr==0) {
      objid = 0;
   } else {
      if (fObjMap==0) return kFALSE;
      ULong_t hash = TMath::Hash(&ptr, sizeof(void*));
      Long_t storedobjid = fObjMap->GetValue(hash, (Long_t) ptr);
      if (storedobjid==0) return kFALSE;

      objid = storedobjid;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TBufferSQL2::RegisterPointer(const void* ptr, Int_t objid)
{
   // Register pair of object id, in object map
   // Object id is used to indentify object in tables

   if ((ptr==0) || (objid==sqlio::Ids_NullPtr)) return;

   ULong_t hash = TMath::Hash(&ptr, sizeof(void*));

   if (fObjMap==0) fObjMap = new TExMap();

   if (fObjMap->GetValue(hash, (Long_t) ptr)==0)
      fObjMap->Add(hash, (Long_t) ptr, (Long_t) objid);
}

//______________________________________________________________________________
Int_t TBufferSQL2::SqlWriteObject(const void* obj, const TClass* cl, TMemberStreamer *streamer, Int_t streamer_index)
{
   // Write object to buffer
   // If object was written before, only pointer will be stored
   // Return id of saved object

   if (gDebug>1)
      cout << " SqlWriteObject " << obj << " : cl = " << (cl ? cl->GetName() : "null") << endl;

   PushStack();

   Int_t objid = 0;

   if (cl==0) obj = 0;

   if (ProcessPointer(obj, objid)) {
      Stack()->SetObjectPointer(objid);
      PopStack();
      return objid;
   }

   objid = fObjIdCounter++;

   Stack()->SetObjectRef(objid, cl);
   RegisterPointer(obj, objid);

   if (streamer!=0)
      (*streamer)(*this, (void*) obj, streamer_index);
   else  
      ((TClass*)cl)->Streamer((void*)obj, *this);

   if (gDebug>1)
      cout << "Done write of " << cl->GetName() << endl;

   PopStack();

   return objid;
}

//______________________________________________________________________________
void* TBufferSQL2::SqlReadObject(void* obj, TClass** cl, TMemberStreamer *streamer, Int_t streamer_index)
{
   // Read object from the buffer

   if (gDebug>2)
      cout << "TBufferSQL2::SqlReadObject " << fCurrentData->GetBlobName2() << endl;

   if (cl) *cl = 0;

   if (fErrorFlag>0) return obj;

   Bool_t findptr = kFALSE;

   const char* refid = fCurrentData->GetValue();
   if ((refid==0) || (strlen(refid)==0)) {
      Error("SqlReadObject","Invalid object reference value");
      fErrorFlag = 1;
      return obj;
   }

   if (!fCurrentData->IsBlobData() ||
       fCurrentData->VerifyDataType(sqlio::ObjectPtr,kFALSE))
      if (strcmp(refid,"0")==0) {
         obj = 0;
         findptr = kTRUE;
      } else
         if ((fIdArray!=0) && (fObjMap!=0)) {
            TNamed* identry = (TNamed*) fIdArray->FindObject(refid);
            if (identry!=0) {
               obj = (void*) fObjMap->GetValue((Long_t) fIdArray->IndexOf(identry));
               if (cl) *cl = gROOT->GetClass(identry->GetTitle());
               findptr = kTRUE;
            }
         }

   if ((gDebug>3) && findptr)
      cout << "    Found pointer " << (obj ? obj : 0)
           << " class = " << ((cl && *cl) ? (*cl)->GetName() : "null") << endl;

   if (findptr) {
      fCurrentData->ShiftToNextValue();
      return obj;
   }

   if (fCurrentData->IsBlobData())
      if (!fCurrentData->VerifyDataType(sqlio::ObjectRef)) {
         Error("SqlReadObject","Object reference or pointer is not found in blob data");
         fErrorFlag = 1;
         return obj;
      }

   Int_t objid = atoi(refid);

   fCurrentData->ShiftToNextValue();

   if (gDebug>2)
      cout << "Found object reference " << objid << endl;

   return SqlReadObjectDirect(obj, cl, objid, streamer, streamer_index);
}

//______________________________________________________________________________
void* TBufferSQL2::SqlReadObjectDirect(void* obj, TClass** cl, Int_t objid, TMemberStreamer *streamer, Int_t streamer_index)
{
   // Read object data.
   // Class name and version are taken from special objects table.

   TString clname;
   Version_t version;

   if (!fSQL->GetObjectData(objid, clname, version)) return obj;

   TSQLClassInfo* sqlinfo = fSQL->RequestSQLClassInfo(clname.Data(), version);

   TClass* objClass = gROOT->GetClass(clname);
   if ((objClass==0) || (sqlinfo==0)) {
      Error("SqlReadObjectDirect","Class %s is not known", clname.Data());
      return obj;
   }

   if (obj==0) obj = objClass->New();

   TString strid;
   strid.Form("%d", objid);

   if (fIdArray==0) {
      fIdArray = new TObjArray;
      fIdArray->SetOwner(kTRUE);
   }
   TNamed* nid = new TNamed(strid.Data(), objClass->GetName());
   fIdArray->Add(nid);

   if (fObjMap==0) fObjMap = new TExMap();

   fObjMap->Add((Long_t) fIdArray->IndexOf(nid), (Long_t) obj);

   PushStack()->SetObjectRef(objid, objClass);

   TSQLObjectData* olddata = fCurrentData;

   if (sqlinfo->IsClassTableExist()) {
      // TObject and TString classes treated differently
      if ((objClass==TObject::Class()) || (objClass==TString::Class())) {

         TSQLObjectData* objdata = new TSQLObjectData;
         if (objClass==TObject::Class())
            TSQLStructure::UnpackTObject(fSQL, objdata, objid, version);
         else
            if (objClass==TString::Class())
               TSQLStructure::UnpackTString(fSQL, objdata, objid, version);

         Stack()->AddObjectData(objdata);
         fCurrentData = objdata;
      } else
         // before normal streamer first version will be read and
         // then streamer functions of TStreamerInfo class
         fReadVersionBuffer = version;
   } else {
      TSQLObjectData* objdata = fSQL->GetObjectClassData(objid, sqlinfo);
      if ((objdata==0) || !objdata->PrepareForRawData()) {
         Error("SqlReadObjectDirect","No found raw data for obj %d in class %s version %d table", objid, clname.Data(), version);
         fErrorFlag = 1;
         return obj;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }

   if (streamer!=0)
      (*streamer)(*this, (void*)obj, streamer_index);
   else  
      objClass->Streamer((void*)obj, *this);

   PopStack();

   if (gDebug>1)
      cout << "Read object of class " << objClass->GetName() << " done" << endl << endl;

   if (cl!=0) *cl = objClass;

   fCurrentData = olddata;

   return obj;
}

//______________________________________________________________________________
void  TBufferSQL2::IncrementLevel(TStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and indent new level in data structure.
   // This call indicates, that TStreamerInfo functions starts streaming
   // object data of correspondent class

   if (info==0) return;

   PushStack()->SetStreamerInfo(info);

   fExpectedChain = kFALSE;

   if (gDebug>2)
      cout << " IncrementLevel " << info->GetClass()->GetName() << endl;

   if (IsReading()) {
      Int_t objid = 0;

      Bool_t isclonesarray = Stack()->IsClonesArray();

      if (isclonesarray) {
         // nothing to do, data will be read from raw data as before
         return;
      }

      if ((fCurrentData!=0) && fCurrentData->IsBlobData() &&
          fCurrentData->VerifyDataType(sqlio::ObjectInst, kFALSE)) {
         objid = atoi(fCurrentData->GetValue());
         fCurrentData->ShiftToNextValue();
         TString sobjid;
         sobjid.Form("%d",objid);
         Stack()->ChangeValueOnly(sobjid.Data());
      } else
         objid = Stack()->DefineObjectId();
      if (objid<0) {
         Error("IncrementLevel","cannot define object id");
         fErrorFlag = 1;
         return;
      }

      TSQLClassInfo* sqlinfo = fSQL->RequestSQLClassInfo(info->GetName(), info->GetClassVersion());
      if (info==0) {
         Error("IncrementLevel","Can not find table for class %s version %d",info->GetName(), info->GetClassVersion());
         fErrorFlag = 1;
         return;
      }

      TSQLObjectData* objdata = fSQL->GetObjectClassData(objid, sqlinfo);
      if (objdata==0) {
         Error("IncrementLevel","Request error for data of object %d for class %s version %d", objid, info->GetName(), info->GetClassVersion());
         fErrorFlag = 1;
         return;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }
}

//______________________________________________________________________________
void  TBufferSQL2::DecrementLevel(TStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and decrease level in sql structure.

   TSQLStructure* curr = Stack();
   if (curr->GetType()==TSQLStructure::kSqlElement) PopStack(); // for element
   PopStack();  // for streamerinfo

   // restore value of object data
   fCurrentData = Stack()->GetObjectData(kTRUE);

   fExpectedChain = kFALSE;

   if (gDebug>2)
      cout << " DecrementLevel " << info->GetClass()->GetName() << endl;
}

//______________________________________________________________________________
void TBufferSQL2::SetStreamerElementNumber(Int_t number)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and add/verify next element in sql tables
   // This calls allows separate data, correspondent to one class member, from another

   if (number>0) PopStack();
   TSQLStructure* curr = Stack();

   TStreamerInfo* info = curr->GetStreamerInfo();
   if (info==0) {
      Error("SetStreamerElementNumber","Error in structures stack");
      return;
   }
   TStreamerElement* elem = info->GetStreamerElementReal(number, 0);

   Int_t comp_type = info->GetTypes()[number];

   Int_t elem_type = elem->GetType();

   fExpectedChain = ((elem_type>0) && (elem_type<20)) &&
      (comp_type - elem_type == TStreamerInfo::kOffsetL);


   WorkWithElement(elem, number);
}

//______________________________________________________________________________
void TBufferSQL2::WorkWithElement(TStreamerElement* elem, Int_t number)
{
   // This function is a part of SetStreamerElementNumber function.
   // It is introduced for reading of data for specified data memeber of class.
   // Used also in ReadFastArray methods to resolve problem of compressed data,
   // when several data memebers of the same basic type streamed with single ...FastArray call

   if (gDebug>2)
      cout << " TBufferSQL2::WorkWithElement " << elem->GetName() << endl;

   PushStack()->SetStreamerElement(elem, number);

   if (IsReading()) {

      if (fCurrentData==0) {
         Error("WorkWithElement","Object data is lost");
         fErrorFlag = 1;
         return;
      }

      fCurrentData = Stack()->GetObjectData(kTRUE);

      Int_t located = Stack()->LocateElementColumn(fSQL, fCurrentData);

      if (located==TSQLStructure::kColUnknown) {
         Error("WorkWithElement","Cannot locate correct column in the table");
         fErrorFlag = 1;
         return;
      } else
         if ((located==TSQLStructure::kColObject) ||
             (located==TSQLStructure::kColObjectArray) ||
             (located==TSQLStructure::kColParent)) {
            // search again for object data while for BLOB it should be already assign
            fCurrentData = Stack()->GetObjectData(kTRUE);
         }
   }
}

//______________________________________________________________________________
TClass* TBufferSQL2::ReadClass(const TClass*, UInt_t*)
{
   // suppressed function of TBuffer

   return 0;
}

//______________________________________________________________________________
void TBufferSQL2::WriteClass(const TClass*)
{
   // suppressed function of TBuffer
}

//______________________________________________________________________________
Int_t TBufferSQL2::CheckByteCount(UInt_t /*r_s */, UInt_t /*r_c*/, const TClass* /*cl*/)
{
   // suppressed function of TBuffer

   return 0;
}

//______________________________________________________________________________
Int_t  TBufferSQL2::CheckByteCount(UInt_t, UInt_t, const char*)
{
   // suppressed function of TBuffer

   return 0;
}

//______________________________________________________________________________
void TBufferSQL2::SetByteCount(UInt_t, Bool_t)
{
   // suppressed function of TBuffer
}

//______________________________________________________________________________
Version_t TBufferSQL2::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass *)
{
   // read version value from buffer
   // actually version is normally defined by table name before
   // and kept in intermediate variable fReadVersionBuffer

   Version_t res = 0;

   if (start) *start = 0;
   if (bcnt) *bcnt = 0;

   if (fReadVersionBuffer>=0) {
      res = fReadVersionBuffer;
      fReadVersionBuffer = -1;
      if (gDebug>3)
         cout << "TBufferSQL2::ReadVersion from buffer = " << res << endl;
   } else
      if ((fCurrentData!=0) && fCurrentData->IsBlobData() &&
          fCurrentData->VerifyDataType(sqlio::Version)) {
         TString value = fCurrentData->GetValue();
         res = value.Atoi();
         if (gDebug>3)
            cout << "TBufferSQL2::ReadVersion from blob " << fCurrentData->GetBlobName1() << " = " << res << endl;
         fCurrentData->ShiftToNextValue();
      } else {
         Error("ReadVersion", "No correspondent tags to read version");
         fErrorFlag = 1;
      }

   return res;
}

//______________________________________________________________________________
UInt_t TBufferSQL2::WriteVersion(const TClass *cl, Bool_t /* useBcnt */)
{
   // Copies class version to buffer, but not writes it to sql immidiately
   // Version will be used to produce complete table
   // name, which will include class version

   if (gDebug>2)
      cout << "TBufferSQL2::WriteVersion " << (cl ? cl->GetName() : "null") << "   ver = " << cl->GetClassVersion() << endl;

   Stack()->AddVersion(cl);

   return 0;
}

//______________________________________________________________________________
void* TBufferSQL2::ReadObjectAny(const TClass*)
{
   // Read object from buffer. Only used from TBuffer
   return SqlReadObject(0);
}

//______________________________________________________________________________
void TBufferSQL2::SkipObjectAny()
{
   // ?????? Skip any kind of object from buffer
   // !!!!!! fix me, not yet implemented
   // Should be just skip of current column later
}

//______________________________________________________________________________
void TBufferSQL2::WriteObject(const void *actualObjStart, const TClass *actualClass)
{
   // Write object to buffer. Only used from TBuffer

   if (gDebug>2)
      cout << "TBufferSQL2::WriteObject of class " << (actualClass ? actualClass->GetName() : " null") << endl;
   SqlWriteObject(actualObjStart, actualClass);
}

#define SQLReadArrayUncompress(vname, arrsize)  \
   {                                            \
      while(indx<arrsize)                       \
         SqlReadBasic(vname[indx++]);           \
   }


#define SQLReadArrayCompress(vname, arrsize)                            \
   {                                                                    \
      while(indx<arrsize) {                                             \
         const char* name = fCurrentData->GetBlobName1();               \
         Int_t first, last, res;                                        \
         if (strstr(name,sqlio::IndexSepar)==0) {                       \
            res = sscanf(name,"[%d]", &first); last = first;            \
         } else res = sscanf(name,"[%d..%d]", &first, &last);           \
         if (gDebug>5) cout << name << " first = " << first << " last = " << last << " res = " << res << endl; \
         if ((first!=indx) || (last<first) || (last>=arrsize)) {        \
            Error("SQLReadArrayCompress","Error reading array content %s", name); \
            fErrorFlag = 1;                                             \
            break;                                                      \
         }                                                              \
         SqlReadBasic(vname[indx]); indx++;                             \
         while(indx<=last)                                              \
            vname[indx++] = vname[first];                               \
      }                                                                 \
   }


// macro to read content of array with compression
#define SQLReadArrayContent(vname, arrsize, withsize)                   \
   {                                                                    \
      if (gDebug>3) cout << "SQLReadArrayContent  " << (arrsize) << endl; \
      PushStack()->SetArray(withsize ? arrsize : -1);                   \
      Int_t indx = 0;                                                   \
      if (fCurrentData->IsBlobData())                                   \
         SQLReadArrayCompress(vname, arrsize)                           \
         else                                                           \
            SQLReadArrayUncompress(vname, arrsize)                      \
               PopStack();                                              \
      if (gDebug>3) cout << "SQLReadArrayContent done " << endl;        \
   }

// macro to read array, which include size attribute
#define TBufferSQL2_ReadArray(tname, vname)     \
   {                                            \
      Int_t n = SqlReadArraySize();             \
      if (n<=0) return 0;                       \
      if (!vname) vname = new tname[n];         \
      SQLReadArrayContent(vname, n, kTRUE);     \
      return n;                                 \
   }

//______________________________________________________________________________
void TBufferSQL2::ReadDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // Read Double_32 value

   SqlReadBasic(*d);
}

//______________________________________________________________________________
void TBufferSQL2::WriteDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // Write Double_32 value

   SqlWriteBasic(*d);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Bool_t    *&b)
{
   // Read array of Bool_t from buffer

   TBufferSQL2_ReadArray(Bool_t,b);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Char_t    *&c)
{
   // Read array of Char_t from buffer

   TBufferSQL2_ReadArray(Char_t,c);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(UChar_t   *&c)
{
   // Read array of UChar_t from buffer

   TBufferSQL2_ReadArray(UChar_t,c);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Short_t   *&h)
{
   // Read array of Short_t from buffer

   TBufferSQL2_ReadArray(Short_t,h);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(UShort_t  *&h)
{
   // Read array of UShort_t from buffer

   TBufferSQL2_ReadArray(UShort_t,h);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Int_t     *&i)
{
   // Read array of Int_t from buffer

   TBufferSQL2_ReadArray(Int_t,i);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(UInt_t    *&i)
{
   // Read array of UInt_t from buffer

   TBufferSQL2_ReadArray(UInt_t,i);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Long_t    *&l)
{
   // Read array of Long_t from buffer

   TBufferSQL2_ReadArray(Long_t,l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(ULong_t   *&l)
{
   // Read array of ULong_t from buffer

   TBufferSQL2_ReadArray(ULong_t,l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Long64_t  *&l)
{
   // Read array of Long64_t from buffer

   TBufferSQL2_ReadArray(Long64_t,l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(ULong64_t *&l)
{
   // Read array of ULong64_t from buffer

   TBufferSQL2_ReadArray(ULong64_t,l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Float_t   *&f)
{
   // Read array of Float_t from buffer

   TBufferSQL2_ReadArray(Float_t,f);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArray(Double_t  *&d)
{
   // Read array of Double_t from buffer

   TBufferSQL2_ReadArray(Double_t,d);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadArrayDouble32(Double_t  *&d, TStreamerElement * /*ele*/)
{
   // Read array of Double32_t from buffer

   TBufferSQL2_ReadArray(Double_t,d);
}

// macro to read static array, which include size attribute
#define TBufferSQL2_ReadStaticArray(vname)      \
   {                                            \
      Int_t n = SqlReadArraySize();             \
      if (n<=0) return 0;                       \
      if (!vname) return 0;                     \
      SQLReadArrayContent(vname, n, kTRUE);     \
      return n;                                 \
   }

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Bool_t    *b)
{
   // Read array of Bool_t from buffer

   TBufferSQL2_ReadStaticArray(b);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Char_t    *c)
{
   // Read array of Char_t from buffer

   TBufferSQL2_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(UChar_t   *c)
{
   // Read array of UChar_t from buffer

   TBufferSQL2_ReadStaticArray(c);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Short_t   *h)
{
   // Read array of Short_t from buffer

   TBufferSQL2_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(UShort_t  *h)
{
   // Read array of UShort_t from buffer

   TBufferSQL2_ReadStaticArray(h);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Int_t     *i)
{
   // Read array of Int_t from buffer

   TBufferSQL2_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(UInt_t    *i)
{
   // Read array of UInt_t from buffer

   TBufferSQL2_ReadStaticArray(i);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Long_t    *l)
{
   // Read array of Long_t from buffer

   TBufferSQL2_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(ULong_t   *l)
{
   // Read array of ULong_t from buffer

   TBufferSQL2_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Long64_t  *l)
{
   // Read array of Long64_t from buffer

   TBufferSQL2_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(ULong64_t *l)
{
   // Read array of ULong64_t from buffer

   TBufferSQL2_ReadStaticArray(l);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Float_t   *f)
{
   // Read array of Float_t from buffer

   TBufferSQL2_ReadStaticArray(f);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArray(Double_t  *d)
{
   // Read array of Double_t from buffer

   TBufferSQL2_ReadStaticArray(d);
}

//______________________________________________________________________________
Int_t TBufferSQL2::ReadStaticArrayDouble32(Double_t  *d, TStreamerElement * /*ele*/)
{
   // Read array of Double32_t from buffer

   TBufferSQL2_ReadStaticArray(d);
}

// macro to read content of array, which not include size of array
// macro also treat situation, when instead of one single array chain of several elements should be produced
#define TBufferSQL2_ReadFastArray(vname)                                \
   {                                                                    \
      if (n<=0) return;                                                 \
      TStreamerElement* elem = Stack(0)->GetElement();                  \
      if ((elem!=0) && (elem->GetType()>TStreamerInfo::kOffsetL) &&     \
          (elem->GetType()<TStreamerInfo::kOffsetP) &&                  \
          (elem->GetArrayLength()!=n)) fExpectedChain = kTRUE;          \
      if (fExpectedChain) {                                             \
         fExpectedChain = kFALSE;                                       \
         Int_t startnumber = Stack(0)->GetElementNumber();              \
         TStreamerInfo* info = Stack(1)->GetStreamerInfo();             \
         Int_t number = 0;                                              \
         Int_t index = 0;                                               \
         while (index<n) {                                              \
            elem = info->GetStreamerElementReal(startnumber, number++); \
            if (number>1) { PopStack(); WorkWithElement(elem, startnumber); } \
            if (elem->GetType()<TStreamerInfo::kOffsetL) {              \
               SqlReadBasic(vname[index]);                              \
               index++;                                                 \
            } else {                                                    \
               Int_t elemlen = elem->GetArrayLength();                  \
               SQLReadArrayContent((vname+index), elemlen, kFALSE);     \
               index+=elemlen;                                          \
            }                                                           \
         }                                                              \
      } else {                                                          \
         SQLReadArrayContent(vname, n, kFALSE);                         \
      }                                                                 \
   }
//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Bool_t    *b, Int_t n)
{
   // read array of Bool_t from buffer

   TBufferSQL2_ReadFastArray(b);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Char_t    *c, Int_t n)
{
   // read array of Char_t from buffer
   // if nodename==CharStar, read all array as string

   if ((n>0) && fCurrentData->IsBlobData() &&
       fCurrentData->VerifyDataType(sqlio::CharStar, kFALSE)) {
      const char* buf = SqlReadCharStarValue();
      if ((buf==0) || (n<=0)) return;
      Int_t size = strlen(buf);
      if (size<n) size = n;
      memcpy(c, buf, size);
   } else {
      //     cout << "call standard macro TBufferSQL2_ReadFastArray" << endl;
      TBufferSQL2_ReadFastArray(c);
   }
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(UChar_t   *c, Int_t n)
{
   // read array of UChar_t from buffer

   TBufferSQL2_ReadFastArray(c);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Short_t   *h, Int_t n)
{
   // read array of Short_t from buffer

   TBufferSQL2_ReadFastArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(UShort_t  *h, Int_t n)
{
   // read array of UShort_t from buffer

   TBufferSQL2_ReadFastArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Int_t     *i, Int_t n)
{
   // read array of Int_t from buffer

   TBufferSQL2_ReadFastArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(UInt_t    *i, Int_t n)
{
   // read array of UInt_t from buffer

   TBufferSQL2_ReadFastArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Long_t    *l, Int_t n)
{
   // read array of Long_t from buffer

   TBufferSQL2_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(ULong_t   *l, Int_t n)
{
   // read array of ULong_t from buffer

   TBufferSQL2_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Long64_t  *l, Int_t n)
{
   // read array of Long64_t from buffer

   TBufferSQL2_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(ULong64_t *l, Int_t n)
{
   // read array of ULong64_t from buffer

   TBufferSQL2_ReadFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Float_t   *f, Int_t n)
{
   // read array of Float_t from buffer

   TBufferSQL2_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(Double_t  *d, Int_t n)
{
   // read array of Double_t from buffer

   TBufferSQL2_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // read array of Double32_t from buffer

   TBufferSQL2_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(void  *start, const TClass *cl, Int_t n, TMemberStreamer *streamer)
{
   // Same functionality as TBuffer::ReadFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished 

   if (gDebug>2)
      cout << "TBufferSQL2::ReadFastArray(* " << endl; 

   if (streamer) {
      StreamObject(start, streamer, cl, 0); 
//      (*streamer)(*this,start,0);
      return;
   }

   int objectSize = cl->Size();
   char *obj = (char*)start;
   char *end = obj + n*objectSize;

   for(; obj<end; obj+=objectSize) 
      StreamObject(obj, cl);

   //   TBuffer::ReadFastArray(start, cl, n, s);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer)
{
   // Same functionality as TBuffer::ReadFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished 

   if (gDebug>2)
      cout << "TBufferSQL2::ReadFastArray(** " << endl; 
  
   if (streamer) {
      if (isPreAlloc) {
         for (Int_t j=0;j<n;j++) {
            if (!start[j]) start[j] = ((TClass*)cl)->New();
         }
      }
      StreamObject((void*)start, streamer, cl, 0); 
//      (*streamer)(*this,(void*)start,0);
      return;
   }

   if (!isPreAlloc) {

      for (Int_t j=0; j<n; j++){
         //delete the object or collection
         if (start[j] && TStreamerInfo::CanDelete()) ((TClass*)cl)->Destructor(start[j],kFALSE); // call delete and desctructor
         start[j] = ReadObjectAny(cl);
      }

   } else {   //case //-> in comment

      for (Int_t j=0; j<n; j++) {
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         StreamObject(start[j], cl);
      }
   }

   //   TBuffer::ReadFastArray(startp, cl, n, isPreAlloc, s);
}

//______________________________________________________________________________
Int_t TBufferSQL2::SqlReadArraySize()
{
   // Reads array size, written in raw data table.
   // Used in ReadArray methods, where TBuffer need to read array size first.   
    
   const char* value = SqlReadValue(sqlio::Array);
   if ((value==0) || (strlen(value)==0)) return 0;
   Int_t sz = atoi(value);
   return sz;
}

// macro to write content of noncompressed array, not used
#define SQLWriteArrayNoncompress(vname, arrsize)        \
   {                                                    \
      for(Int_t indx=0;indx<arrsize;indx++) {           \
         SqlWriteBasic(vname[indx]);                    \
         Stack()->ChildArrayIndex(indx, 1);             \
      }                                                 \
   }

// macro to write content of compressed array
#define SQLWriteArrayCompress(vname, arrsize)                           \
   {                                                                    \
      Int_t indx = 0;                                                   \
      while(indx<arrsize) {                                             \
         Int_t curr = indx; indx++;                                     \
         while ((indx<arrsize) && (vname[indx]==vname[curr])) indx++;   \
         SqlWriteBasic(vname[curr]);                                    \
         Stack()->ChildArrayIndex(curr, indx-curr);                     \
      }                                                                 \
   }

#define SQLWriteArrayContent(vname, arrsize, withsize)  \
   {                                                    \
      PushStack()->SetArray(withsize ? arrsize : -1);   \
      if (fCompressLevel>0) {                           \
         SQLWriteArrayCompress(vname, arrsize)          \
            } else {                                    \
         SQLWriteArrayNoncompress(vname, arrsize)       \
            }                                           \
      PopStack();                                       \
   }

// macro to write array, which include size
#define TBufferSQL2_WriteArray(vname)           \
   {                                            \
      SQLWriteArrayContent(vname, n, kTRUE);    \
   }

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Bool_t    *b, Int_t n)
{
   // Write array of Bool_t to buffer

   TBufferSQL2_WriteArray(b);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Char_t    *c, Int_t n)
{
   // Write array of Char_t to buffer

   TBufferSQL2_WriteArray(c);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const UChar_t   *c, Int_t n)
{
   // Write array of UChar_t to buffer

   TBufferSQL2_WriteArray(c);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Short_t   *h, Int_t n)
{
   // Write array of Short_t to buffer

   TBufferSQL2_WriteArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const UShort_t  *h, Int_t n)
{
   // Write array of UShort_t to buffer

   TBufferSQL2_WriteArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Int_t     *i, Int_t n)
{
   // Write array of Int_ to buffer

   TBufferSQL2_WriteArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const UInt_t    *i, Int_t n)
{
   // Write array of UInt_t to buffer

   TBufferSQL2_WriteArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Long_t    *l, Int_t n)
{
   // Write array of Long_t to buffer

   TBufferSQL2_WriteArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const ULong_t   *l, Int_t n)
{
   // Write array of ULong_t to buffer

   TBufferSQL2_WriteArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Long64_t  *l, Int_t n)
{
   // Write array of Long64_t to buffer

   TBufferSQL2_WriteArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const ULong64_t *l, Int_t n)
{
   // Write array of ULong64_t to buffer

   TBufferSQL2_WriteArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Float_t   *f, Int_t n)
{
   // Write array of Float_t to buffer

   TBufferSQL2_WriteArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArray(const Double_t  *d, Int_t n)
{
   // Write array of Double_t to buffer

   TBufferSQL2_WriteArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::WriteArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Double32_t to buffer

   TBufferSQL2_WriteArray(d);
}

// write array without size attribute
// macro also treat situation, when instead of one single array chain of several elements should be produced
#define TBufferSQL2_WriteFastArray(vname)                               \
   {                                                                    \
      if (n<=0) return;                                                 \
      TStreamerElement* elem = Stack(0)->GetElement();                  \
      if ((elem!=0) && (elem->GetType()>TStreamerInfo::kOffsetL) &&     \
          (elem->GetType()<TStreamerInfo::kOffsetP) &&                  \
          (elem->GetArrayLength()!=n)) fExpectedChain = kTRUE;          \
      if (fExpectedChain) {                                             \
         TStreamerInfo* info = Stack(1)->GetStreamerInfo();             \
         Int_t startnumber = Stack(0)->GetElementNumber();              \
         Int_t number = 0;                                              \
         Int_t index = 0;                                               \
         while (index<n) {                                              \
            elem = info->GetStreamerElementReal(startnumber, number++); \
            if (number>1) { PopStack(); WorkWithElement(elem, startnumber + number); } \
            if (elem->GetType()<TStreamerInfo::kOffsetL) {              \
               SqlWriteBasic(vname[index]);                             \
               index++;                                                 \
            } else {                                                    \
               Int_t elemlen = elem->GetArrayLength();                  \
               SQLWriteArrayContent((vname+index), elemlen, kFALSE);    \
               index+=elemlen;                                          \
            }                                                           \
            fExpectedChain = kFALSE;                                    \
         }                                                              \
      } else {                                                          \
         SQLWriteArrayContent(vname, n, kFALSE);                        \
      }                                                                 \
   }

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Bool_t    *b, Int_t n)
{
   // Write array of Bool_t to buffer

   TBufferSQL2_WriteFastArray(b);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Char_t    *c, Int_t n)
{
   // Write array of Char_t to buffer
   // it will be reproduced as CharStar node with string as attribute

   Bool_t usedefault = (n==0) || fExpectedChain;

   const Char_t* ccc = c;
   // check if no zeros in the array
   if (!usedefault)
      for (int i=0;i<n;i++)
         if (*ccc++==0) { usedefault = kTRUE; break; }

   if (usedefault) {
      TBufferSQL2_WriteFastArray(c);
   } else {
      Char_t* buf = new Char_t[n+1];
      memcpy(buf, c, n);
      buf[n] = 0;
      SqlWriteValue(buf, sqlio::CharStar);
      delete[] buf;
   }
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const UChar_t   *c, Int_t n)
{
   // Write array of UChar_t to buffer

   TBufferSQL2_WriteFastArray(c);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Short_t   *h, Int_t n)
{
   // Write array of Short_t to buffer

   TBufferSQL2_WriteFastArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const UShort_t  *h, Int_t n)
{
   // Write array of UShort_t to buffer

   TBufferSQL2_WriteFastArray(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Int_t     *i, Int_t n)
{
   // Write array of Int_t to buffer

   TBufferSQL2_WriteFastArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const UInt_t    *i, Int_t n)
{
   // Write array of UInt_t to buffer

   TBufferSQL2_WriteFastArray(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Long_t    *l, Int_t n)
{
   // Write array of Long_t to buffer

   TBufferSQL2_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const ULong_t   *l, Int_t n)
{
   // Write array of ULong_t to buffer

   TBufferSQL2_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Long64_t  *l, Int_t n)
{
   // Write array of Long64_t to buffer

   TBufferSQL2_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const ULong64_t *l, Int_t n)
{
   // Write array of ULong64_t to buffer

   TBufferSQL2_WriteFastArray(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Float_t   *f, Int_t n)
{
   // Write array of Float_t to buffer

   TBufferSQL2_WriteFastArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArray(const Double_t  *d, Int_t n)
{
   // Write array of Double_t to buffer

   TBufferSQL2_WriteFastArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFastArrayDouble32(const Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Double32_t to buffer

   TBufferSQL2_WriteFastArray(d);
}

//______________________________________________________________________________
void  TBufferSQL2::WriteFastArray(void  *start,  const TClass *cl, Int_t n, TMemberStreamer *streamer)
{
   // Same functionality as TBuffer::WriteFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished 

   if (streamer) {
      StreamObject(start, streamer, cl, 0); 
//      (*streamer)(*this, start, 0);
      return;
   }

   char *obj = (char*)start;
   if (!n) n=1;
   int size = cl->Size();

   for(Int_t j=0; j<n; j++,obj+=size) 
     StreamObject(obj, cl);

   //   TBuffer::WriteFastArray(start, cl, n, s);
}

//______________________________________________________________________________
Int_t TBufferSQL2::WriteFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer)
{
   // Same functionality as TBuffer::WriteFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished 
   
   if (streamer) {
      StreamObject((void*) start, streamer, cl, 0); 
//      (*streamer)(*this,(void*)start,0);
      return 0;
   }

   int strInfo = 0;

   Int_t res = 0;

   if (!isPreAlloc) {

      for (Int_t j=0;j<n;j++) {
         //must write StreamerInfo if pointer is null
         if (!strInfo && !start[j] ) ((TClass*)cl)->GetStreamerInfo()->ForceWriteInfo((TFile *)GetParent());
         strInfo = 2003;
         res |= WriteObjectAny(start[j],cl);
      }

   } else {	//case //-> in comment

      for (Int_t j=0;j<n;j++) {
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         StreamObject(start[j], cl);
      }

   }
   return res;

//   return TBuffer::WriteFastArray(startp, cl, n, isPreAlloc, s);
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const type_info &typeinfo)
{
   // steram object to/from buffer

   StreamObject(obj, gROOT->GetClass(typeinfo));
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const char *className)
{
   // steram object to/from buffer

   StreamObject(obj, gROOT->GetClass(className));
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const TClass *cl)
{
   // steram object to/from buffer

   if (gDebug>1)
      cout << " TBufferSQL2::StreamObject class = " << (cl ? cl->GetName() : "none") << endl;
   if (IsReading())
      SqlReadObject(obj);
   else
      SqlWriteObject(obj, cl);
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(TObject *obj)
{
   // steram object to/from buffer

   StreamObject(obj, obj ? obj->IsA() : TObject::Class());
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, TMemberStreamer *streamer, const TClass *cl, Int_t n)
{
   // steram object to/from buffer

   if (streamer==0) return;
   
   if (gDebug>1)
      cout << "Stream object of class = " << cl->GetName() << endl;
//   (*streamer)(*this, obj, n);
   
   if (IsReading())
      SqlReadObject(obj, 0, streamer, n);
   else
      SqlWriteObject(obj, cl, streamer, n);
}

// macro for right shift operator for basic type
#define TBufferSQL2_operatorin(vname)           \
   {                                            \
      SqlReadBasic(vname);                      \
      return *this;                             \
   }

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Bool_t    &b)
{
   // Reads Bool_t value from buffer

   TBufferSQL2_operatorin(b);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Char_t    &c)
{
   // Reads Char_t value from buffer

   TBufferSQL2_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(UChar_t   &c)
{
   // Reads UChar_t value from buffer

   TBufferSQL2_operatorin(c);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Short_t   &h)
{
   // Reads Short_t value from buffer

   TBufferSQL2_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(UShort_t  &h)
{
   // Reads UShort_t value from buffer

   TBufferSQL2_operatorin(h);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Int_t     &i)
{
   // Reads Int_t value from buffer

   TBufferSQL2_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(UInt_t    &i)
{
   // Reads UInt_t value from buffer

   TBufferSQL2_operatorin(i);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Long_t    &l)
{
   // Reads Long_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(ULong_t   &l)
{
   // Reads ULong_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Long64_t  &l)
{
   // Reads Long64_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(ULong64_t &l)
{
   // Reads ULong64_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Float_t   &f)
{
   // Reads Float_t value from buffer

   TBufferSQL2_operatorin(f);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Double_t  &d)
{
   // Reads Double_t value from buffer

   TBufferSQL2_operatorin(d);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator>>(Char_t    *c)
{
   // Reads array of characters from buffer

   const char* buf = SqlReadCharStarValue();
   strcpy(c, buf);
   return *this;
}

// macro for right shift operator for basic types
#define TBufferSQL2_operatorout(vname)          \
   {                                            \
      SqlWriteBasic(vname);                     \
      return *this;                             \
   }

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Bool_t    b)
{
   // Writes Bool_t value to buffer

   TBufferSQL2_operatorout(b);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Char_t    c)
{
   // Writes Char_t value to buffer

   TBufferSQL2_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(UChar_t   c)
{
   // Writes UChar_t value to buffer

   TBufferSQL2_operatorout(c);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Short_t   h)
{
   // Writes Short_t value to buffer

   TBufferSQL2_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(UShort_t  h)
{
   // Writes UShort_t value to buffer

   TBufferSQL2_operatorout(h);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Int_t     i)
{
   // Writes Int_t value to buffer

   TBufferSQL2_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(UInt_t    i)
{
   // Writes UInt_t value to buffer

   TBufferSQL2_operatorout(i);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Long_t    l)
{
   // Writes Long_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(ULong_t   l)
{
   // Writes ULong_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Long64_t  l)
{
   // Writes Long64_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(ULong64_t l)
{
   // Writes ULong64_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Float_t   f)
{
   // Writes Float_t value to buffer

   TBufferSQL2_operatorout(f);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(Double_t  d)
{
   // Writes Double_t value to buffer

   TBufferSQL2_operatorout(d);
}

//______________________________________________________________________________
TBuffer& TBufferSQL2::operator<<(const Char_t *c)
{
   // Writes array of characters to buffer

   SqlWriteValue(c, sqlio::CharStar);
   return *this;
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Char_t value)
{
   // converts Char_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%d",value);
   return SqlWriteValue(buf, sqlio::Char);
}

//______________________________________________________________________________
Bool_t  TBufferSQL2::SqlWriteBasic(Short_t value)
{
   // converts Short_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%hd", value);
   return SqlWriteValue(buf, sqlio::Short);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Int_t value)
{
   // converts Int_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%d", value);
   return SqlWriteValue(buf, sqlio::Int);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Long_t value)
{
   // converts Long_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%ld", value);
   return SqlWriteValue(buf, sqlio::Long);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Long64_t value)
{
   // converts Long64_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%lld", value);
   return SqlWriteValue(buf, sqlio::Long64);
}

//______________________________________________________________________________
Bool_t  TBufferSQL2::SqlWriteBasic(Float_t value)
{
   // converts Float_t to string and creates correspondent sql structure

   char buf[200];
   sprintf(buf,"%f", value);
   return SqlWriteValue(buf, sqlio::Float);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Double_t value)
{
   // converts Double_t to string and creates correspondent sql structure

   char buf[1000];
   sprintf(buf,"%f", value);
   return SqlWriteValue(buf, sqlio::Double);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Bool_t value)
{
   // converts Bool_t to string and creates correspondent sql structure

   return SqlWriteValue(value ? sqlio::True : sqlio::False, sqlio::Bool);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(UChar_t value)
{
   // converts UChar_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%u", value);
   return SqlWriteValue(buf, sqlio::UChar);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(UShort_t value)
{
   // converts UShort_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%hu", value);
   return SqlWriteValue(buf, sqlio::UShort);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(UInt_t value)
{
   // converts UInt_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%u", value);
   return SqlWriteValue(buf, sqlio::UInt);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(ULong_t value)
{
   // converts ULong_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%lu", value);
   return SqlWriteValue(buf, sqlio::ULong);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(ULong64_t value)
{
   // converts ULong64_t to string and creates correspondent sql structure

   char buf[50];
   sprintf(buf,"%llu", value);
   return SqlWriteValue(buf, sqlio::ULong64);
}

//______________________________________________________________________________

Bool_t TBufferSQL2::SqlWriteValue(const char* value, const char* tname)
{
   // create structure in stack, which holds specified value

   Stack()->AddValue(value, tname);

   return kTRUE;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Char_t& value)
{
   // read current value from table and convert it to Char_t value

   const char* res = SqlReadValue(sqlio::Char);
   if (res) {
      int n;
      sscanf(res,"%d", &n);
      value = n;
   } else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Short_t& value)
{
   // read current value from table and convert it to Short_t value

   const char* res = SqlReadValue(sqlio::Short);
   if (res)
      sscanf(res,"%hd", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Int_t& value)
{
   // read current value from table and convert it to Int_t value

   const char* res = SqlReadValue(sqlio::Int);
   if (res)
      sscanf(res,"%d", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Long_t& value)
{
   // read current value from table and convert it to Long_t value

   const char* res = SqlReadValue(sqlio::Long);
   if (res)
      sscanf(res,"%ld", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Long64_t& value)
{
   // read current value from table and convert it to Long64_t value

   const char* res = SqlReadValue(sqlio::Long64);
   if (res)
      sscanf(res,"%lld", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Float_t& value)
{
   // read current value from table and convert it to Float_t value

   const char* res = SqlReadValue(sqlio::Float);
   if (res)
      sscanf(res,"%f", &value);
   else
      value = 0.;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Double_t& value)
{
   // read current value from table and convert it to Double_t value

   const char* res = SqlReadValue(sqlio::Double);
   if (res)
      sscanf(res,"%lf", &value);
   else
      value = 0.;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Bool_t& value)
{
   // read current value from table and convert it to Bool_t value

   const char* res = SqlReadValue(sqlio::Bool);
   if (res)
      value = (strcmp(res, sqlio::True)==0);
   else
      value = kFALSE;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(UChar_t& value)
{
   // read current value from table and convert it to UChar_t value

   const char* res = SqlReadValue(sqlio::UChar);
   if (res) {
      unsigned int n;
      sscanf(res,"%ud", &n);
      value = n;
   } else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(UShort_t& value)
{
   // read current value from table and convert it to UShort_t value

   const char* res = SqlReadValue(sqlio::UShort);
   if (res)
      sscanf(res,"%hud", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(UInt_t& value)
{
   // read current value from table and convert it to UInt_t value

   const char* res = SqlReadValue(sqlio::UInt);
   if (res)
      sscanf(res,"%u", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(ULong_t& value)
{
   // read current value from table and convert it to ULong_t value

   const char* res = SqlReadValue(sqlio::ULong);
   if (res)
      sscanf(res,"%lu", &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(ULong64_t& value)
{
   // read current value from table and convert it to ULong64_t value

   const char* res = SqlReadValue(sqlio::ULong64);
   if (res)
      sscanf(res,"%llu", &value);
   else
      value = 0;
}

//______________________________________________________________________________
const char* TBufferSQL2::SqlReadValue(const char* tname)
{
   // read string value from current stack node

   if (fErrorFlag>0) return 0;

   if (fCurrentData==0) {
      Error("SqlReadValue","No object data to read from");
      fErrorFlag = 1;
      return 0;
   }

   if (!fIgnoreVerification)
      if (!fCurrentData->VerifyDataType(tname)) {
         fErrorFlag = 1;
         return 0;
      }

   fReadBuffer = fCurrentData->GetValue();

   fCurrentData->ShiftToNextValue();

   if (gDebug>4)
      cout << "   SqlReadValue " << tname << " = " << fReadBuffer << endl;

   return fReadBuffer.Data();
}

//______________________________________________________________________________
const char* TBufferSQL2::SqlReadCharStarValue()
{
   // read CharStar value, if it has special code, request it from large table
   const char* res = SqlReadValue(sqlio::CharStar);
   if ((res==0) || (fSQL==0)) return 0;

   Int_t objid = Stack()->DefineObjectId();

   Int_t strid = fSQL->IsLongStringCode(res, objid);
   if (strid<=0) return res;

   fSQL->GetLongString(objid, strid, fReadBuffer);

   return fReadBuffer.Data();
}


//______________________________________________________________________________
TSQLStructure* TBufferSQL2::PushStack()
{
   // Push stack with structurual information about streamed object

   TSQLStructure* res = new TSQLStructure;
   if (fStk==0) {
      fStructure = res;
   } else {
      fStk->Add(res);
   }

   fStk = res;   // add in the stack
   return fStk;
}

//______________________________________________________________________________
TSQLStructure* TBufferSQL2::PopStack()
{
   // Pop stack

   if (fStk==0) return 0;
   fStk = fStk->GetParent();
   return fStk;
}

//______________________________________________________________________________
TSQLStructure* TBufferSQL2::Stack(Int_t depth)
{
   // returns head of stack

   TSQLStructure* curr = fStk;
   while ((depth-->0) && (curr!=0)) curr = curr->GetParent();
   return curr;
}
