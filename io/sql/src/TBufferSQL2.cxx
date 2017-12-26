// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
\class TBufferSQL2
\ingroup IO

Converts data to SQL statements or read data from SQL tables.

Class for serializing/deserializing object to/from SQL data base.
It redefines most of TBuffer class function to convert simple types,
array of simple types and objects to/from TSQLStructure objects.
TBufferSQL2 class uses streaming mechanism, provided by ROOT system,
therefore most of ROOT and user classes can be stored. There are
limitations for complex objects like TTree, TClonesArray, TDirectory and
few other, which can not be converted to SQL (yet).
*/

#include "TBufferSQL2.h"

#include "TObjArray.h"
#include "TROOT.h"
#include "TDataType.h"
#include "TClass.h"
#include "TClassTable.h"
#include "TMap.h"
#include "TExMap.h"
#include "TMethodCall.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TFile.h"
#include "TMemberStreamer.h"
#include "TStreamer.h"
#include "Riostream.h"
#include <stdlib.h>
#include "TStreamerInfoActions.h"

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TSQLStructure.h"
#include "TSQLObjectData.h"
#include "TSQLFile.h"
#include "TSQLClassInfo.h"

#ifdef R__VISUAL_CPLUSPLUS
#define FLong64 "%I64d"
#define FULong64 "%I64u"
#else
#define FLong64 "%lld"
#define FULong64 "%llu"
#endif

ClassImp(TBufferSQL2);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor, should not be used

TBufferSQL2::TBufferSQL2()
   : TBufferText(), fSQL(nullptr), fIOVersion(1), fStructure(0), fStk(0), fObjMap(nullptr), fReadBuffer(),
     fErrorFlag(0), fCompressLevel(0), fReadVersionBuffer(-1), fObjIdCounter(1), fIgnoreVerification(kFALSE),
     fCurrentData(0), fObjectsInfos(nullptr), fFirstObjId(0), fLastObjId(0), fPoolsMap(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Creates buffer object to serialize/deserialize data to/from sql.
/// Mode should be either TBuffer::kRead or TBuffer::kWrite.

TBufferSQL2::TBufferSQL2(TBuffer::EMode mode)
   : TBufferText(mode), fSQL(nullptr), fIOVersion(1), fStructure(0), fStk(0), fObjMap(nullptr), fReadBuffer(),
     fErrorFlag(0), fCompressLevel(0), fReadVersionBuffer(-1), fObjIdCounter(1), fIgnoreVerification(kFALSE),
     fCurrentData(0), fObjectsInfos(nullptr), fFirstObjId(0), fLastObjId(0), fPoolsMap(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Creates buffer object to serialize/deserialize data to/from sql.
/// This constructor should be used, if data from buffer supposed to be stored in file.
/// Mode should be either TBuffer::kRead or TBuffer::kWrite.

TBufferSQL2::TBufferSQL2(TBuffer::EMode mode, TSQLFile *file)
   : TBufferText(mode), fSQL(nullptr), fIOVersion(1), fStructure(0), fStk(0), fObjMap(nullptr), fReadBuffer(),
     fErrorFlag(0), fCompressLevel(0), fReadVersionBuffer(-1), fObjIdCounter(1), fIgnoreVerification(kFALSE),
     fCurrentData(0), fObjectsInfos(nullptr), fFirstObjId(0), fLastObjId(0), fPoolsMap(0)
{
   SetParent(file);
   fSQL = file;
   if (file) {
      SetCompressionLevel(file->GetCompressionLevel());
      fIOVersion = file->GetIOVersion();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Destroy sql buffer.

TBufferSQL2::~TBufferSQL2()
{
   if (fObjMap)
      delete fObjMap;

   if (fStructure)
      delete fStructure;

   if (fObjectsInfos) {
      fObjectsInfos->Delete();
      delete fObjectsInfos;
   }

   if (fPoolsMap) {
      fPoolsMap->DeleteValues();
      delete fPoolsMap;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Convert object of any class to sql structures
/// Return pointer on created TSQLStructure
/// TSQLStructure object will be owned by TBufferSQL2

TSQLStructure *TBufferSQL2::SqlWriteAny(const void *obj, const TClass *cl, Long64_t objid)
{
   fErrorFlag = 0;

   fStructure = 0;

   fFirstObjId = objid;
   fObjIdCounter = objid;

   SqlWriteObject(obj, cl, kTRUE);

   if (gDebug > 3)
      if (fStructure != 0) {
         std::cout << "==== Printout of Sql structures ===== " << std::endl;
         fStructure->Print("*");
         std::cout << "=========== End printout ============ " << std::endl;
      }

   return fStructure;
}

////////////////////////////////////////////////////////////////////////////////
/// Recreate object from sql structure.
/// Return pointer to read object.
/// if (cl!=0) returns pointer to class of object

void *TBufferSQL2::SqlReadAny(Long64_t keyid, Long64_t objid, TClass **cl, void *obj)
{
   if (cl)
      *cl = nullptr;
   if (!fSQL)
      return nullptr;

   fCurrentData = 0;
   fErrorFlag = 0;

   fReadVersionBuffer = -1;

   fObjectsInfos = fSQL->SQLObjectsInfo(keyid);
   fFirstObjId = objid;
   fLastObjId = objid;
   if (fObjectsInfos) {
      TSQLObjectInfo *objinfo = (TSQLObjectInfo *)fObjectsInfos->Last();
      if (objinfo)
         fLastObjId = objinfo->GetObjId();
   }

   return SqlReadObjectDirect(obj, cl, objid);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns object info like classname and version
/// Should be taken from buffer, which is produced in the beginning

Bool_t TBufferSQL2::SqlObjectInfo(Long64_t objid, TString &clname, Version_t &version)
{
   if ((objid < 0) || !fObjectsInfos)
      return kFALSE;

   // suppose that objects info are sorted out

   Long64_t shift = objid - fFirstObjId;

   TSQLObjectInfo *info = 0;
   if ((shift >= 0) && (shift <= fObjectsInfos->GetLast())) {
      info = (TSQLObjectInfo *)fObjectsInfos->At(shift);
      if (info->GetObjId() != objid)
         info = 0;
   }

   if (info == 0) {
      // I hope, i will never get inside it
      Info("SqlObjectInfo", "Standard not works %lld", objid);
      for (Int_t n = 0; n <= fObjectsInfos->GetLast(); n++) {
         info = (TSQLObjectInfo *)fObjectsInfos->At(n);
         if (info->GetObjId() == objid)
            break;
         info = 0;
      }
   }

   if (info == 0)
      return kFALSE;

   clname = info->GetObjClassName();
   version = info->GetObjVersion();
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Creates TSQLObjectData for specified object id and specified class
///
/// Object data for each class can be stored in two different tables.
/// First table contains data in column-wise form for simple types like integer,
/// strings and so on when second table contains any other data which cannot
/// be converted into column-wise representation.
/// TSQLObjectData will contain results of the requests to both such tables for
/// concrete object id.

TSQLObjectData *TBufferSQL2::SqlObjectData(Long64_t objid, TSQLClassInfo *sqlinfo)
{
   TSQLResult *classdata = 0;
   TSQLRow *classrow = 0;

   if (sqlinfo->IsClassTableExist()) {

      TSQLObjectDataPool *pool = 0;

      if (fPoolsMap != 0)
         pool = (TSQLObjectDataPool *)fPoolsMap->GetValue(sqlinfo);

      if ((pool == 0) && (fLastObjId >= fFirstObjId)) {
         if (gDebug > 4)
            Info("SqlObjectData", "Before request to %s", sqlinfo->GetClassTableName());
         TSQLResult *alldata = fSQL->GetNormalClassDataAll(fFirstObjId, fLastObjId, sqlinfo);
         if (gDebug > 4)
            Info("SqlObjectData", "After request res = 0x%lx", (Long_t)alldata);
         if (alldata == 0) {
            Error("SqlObjectData", "Cannot get data from table %s", sqlinfo->GetClassTableName());
            return 0;
         }

         if (fPoolsMap == 0)
            fPoolsMap = new TMap();
         pool = new TSQLObjectDataPool(sqlinfo, alldata);
         fPoolsMap->Add(sqlinfo, pool);
      }

      if (pool == 0)
         return 0;

      if (pool->GetSqlInfo() != sqlinfo) {
         Error("SqlObjectData", "Missmatch in pools map !!! CANNOT BE !!!");
         return 0;
      }

      classdata = pool->GetClassData();

      classrow = pool->GetObjectRow(objid);
      if (classrow == 0) {
         Error("SqlObjectData", "Can not find row for objid = %lld in table %s", objid, sqlinfo->GetClassTableName());
         return 0;
      }
   }

   TSQLResult *blobdata = 0;
   TSQLStatement *blobstmt = fSQL->GetBlobClassDataStmt(objid, sqlinfo);

   if (blobstmt == 0)
      blobdata = fSQL->GetBlobClassData(objid, sqlinfo);

   return new TSQLObjectData(sqlinfo, objid, classdata, classrow, blobdata, blobstmt);
}

////////////////////////////////////////////////////////////////////////////////
/// Write object to buffer.
/// If object was written before, only pointer will be stored
/// Return id of saved object

Int_t TBufferSQL2::SqlWriteObject(const void *obj, const TClass *cl, Bool_t cacheReuse, TMemberStreamer *streamer,
                                  Int_t streamer_index)
{
   if (gDebug > 1)
      Info("SqlWriteObject", "Object: %p Class: %s", obj, (cl ? cl->GetName() : "null"));

   PushStack();

   Long64_t objid = -1;

   if (!cl)
      obj = nullptr;

   if (!obj) {
      objid = 0;
   } else if (fObjMap != 0) {
      ULong_t hash = TString::Hash(&obj, sizeof(void *));
      Long_t value = fObjMap->GetValue(hash, (Long_t)obj);
      if (value > 0)
         objid = fFirstObjId + value - 1;
   }

   if (gDebug > 1)
      Info("SqlWriteObject", "Find objectid %ld", (long)objid);

   if (objid >= 0) {
      Stack()->SetObjectPointer(objid);
      PopStack();
      return objid;
   }

   objid = fObjIdCounter++;

   Stack()->SetObjectRef(objid, cl);

   if (cacheReuse) {
      ULong_t hash = TString::Hash(&obj, sizeof(void *));
      if (fObjMap == 0)
         fObjMap = new TExMap();
      if (fObjMap->GetValue(hash, (Long_t)obj) == 0)
         fObjMap->Add(hash, (Long_t)obj, (Long_t)objid - fFirstObjId + 1);
   }

   if (streamer != 0)
      (*streamer)(*this, (void *)obj, streamer_index);
   else
      ((TClass *)cl)->Streamer((void *)obj, *this);

   if (gDebug > 1)
      Info("SqlWriteObject", "Done write of %s", cl->GetName());

   PopStack();

   return objid;
}

////////////////////////////////////////////////////////////////////////////////
/// Read object from the buffer

void *TBufferSQL2::SqlReadObject(void *obj, TClass **cl, TMemberStreamer *streamer, Int_t streamer_index,
                                 const TClass *onFileClass)
{
   if (cl)
      *cl = nullptr;

   if (fErrorFlag > 0)
      return obj;

   Bool_t findptr = kFALSE;

   const char *refid = fCurrentData->GetValue();
   if ((refid == 0) || (strlen(refid) == 0)) {
      Error("SqlReadObject", "Invalid object reference value");
      fErrorFlag = 1;
      return obj;
   }

   Long64_t objid = -1;
   sscanf(refid, FLong64, &objid);

   if (gDebug > 2)
      Info("SqlReadObject", "Starting objid = %lld column=%s", objid, fCurrentData->GetLocatedField());

   if (!fCurrentData->IsBlobData() || fCurrentData->VerifyDataType(sqlio::ObjectPtr, kFALSE)) {
      if (objid == 0) {
         obj = nullptr;
         findptr = kTRUE;
      } else {
         if (objid == -1) {
            findptr = kTRUE;
         } else {
            if ((fObjMap != 0) && (objid >= fFirstObjId)) {
               void *obj1 = (void *)(Long_t)fObjMap->GetValue((Long_t)objid - fFirstObjId);
               if (obj1) {
                  obj = obj1;
                  findptr = kTRUE;
                  TString clname;
                  Version_t version;
                  if (cl && SqlObjectInfo(objid, clname, version))
                     *cl = TClass::GetClass(clname);
               }
            }
         }
      }
   }

   if ((gDebug > 3) && findptr)
      Info("SqlReadObject", "Found pointer %p cl %s", obj, ((cl && *cl) ? (*cl)->GetName() : "null"));

   if (findptr) {
      fCurrentData->ShiftToNextValue();
      return obj;
   }

   if (fCurrentData->IsBlobData())
      if (!fCurrentData->VerifyDataType(sqlio::ObjectRef)) {
         Error("SqlReadObject", "Object reference or pointer is not found in blob data");
         fErrorFlag = 1;
         return obj;
      }

   fCurrentData->ShiftToNextValue();

   if ((gDebug > 2) || (objid < 0))
      Info("SqlReadObject", "Found object reference %ld", (long)objid);

   return SqlReadObjectDirect(obj, cl, objid, streamer, streamer_index, onFileClass);
}

////////////////////////////////////////////////////////////////////////////////
/// Read object data.
/// Class name and version are taken from special objects table.

void *TBufferSQL2::SqlReadObjectDirect(void *obj, TClass **cl, Long64_t objid, TMemberStreamer *streamer,
                                       Int_t streamer_index, const TClass *onFileClass)
{
   TString clname;
   Version_t version;

   if (!SqlObjectInfo(objid, clname, version))
      return obj;

   if (gDebug > 2)
      Info("SqlReadObjectDirect", "objid = %lld clname = %s ver = %d", objid, clname.Data(), version);

   TSQLClassInfo *sqlinfo = fSQL->FindSQLClassInfo(clname.Data(), version);

   TClass *objClass = TClass::GetClass(clname);
   if (objClass == TDirectory::Class())
      objClass = TDirectoryFile::Class();

   if (!objClass || !sqlinfo) {
      Error("SqlReadObjectDirect", "Class %s is not known", clname.Data());
      return obj;
   }

   if (obj == 0)
      obj = objClass->New();

   if (fObjMap == 0)
      fObjMap = new TExMap();

   fObjMap->Add((Long_t)objid - fFirstObjId, (Long_t)obj);

   PushStack()->SetObjectRef(objid, objClass);

   TSQLObjectData *olddata = fCurrentData;

   if (sqlinfo->IsClassTableExist()) {
      // TObject and TString classes treated differently
      if ((objClass == TObject::Class()) || (objClass == TString::Class())) {

         TSQLObjectData *objdata = new TSQLObjectData;
         if (objClass == TObject::Class())
            TSQLStructure::UnpackTObject(fSQL, this, objdata, objid, version);
         else if (objClass == TString::Class())
            TSQLStructure::UnpackTString(fSQL, this, objdata, objid, version);

         Stack()->AddObjectData(objdata);
         fCurrentData = objdata;
      } else
         // before normal streamer first version will be read and
         // then streamer functions of TStreamerInfo class
         fReadVersionBuffer = version;
   } else {
      TSQLObjectData *objdata = SqlObjectData(objid, sqlinfo);
      if ((objdata == 0) || !objdata->PrepareForRawData()) {
         Error("SqlReadObjectDirect", "No found raw data for obj %lld in class %s version %d table", objid,
               clname.Data(), version);
         fErrorFlag = 1;
         return obj;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }

   if (streamer != 0) {
      streamer->SetOnFileClass(onFileClass);
      (*streamer)(*this, (void *)obj, streamer_index);
   } else {
      objClass->Streamer((void *)obj, *this, onFileClass);
   }

   PopStack();

   if (gDebug > 1)
      Info("SqlReadObjectDirect", "Read object of class %s done", objClass->GetName());

   if (cl != 0)
      *cl = objClass;

   fCurrentData = olddata;

   return obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
/// and indent new level in data structure.
/// This call indicates, that TStreamerInfo functions starts streaming
/// object data of correspondent class

void TBufferSQL2::IncrementLevel(TVirtualStreamerInfo *info)
{
   if (!info)
      return;

   PushStack()->SetStreamerInfo((TStreamerInfo *)info);

   if (gDebug > 2)
      Info("IncrementLevel", "Info: %s", info->GetName());

   WorkWithClass(info->GetName(), info->GetClassVersion());
}

////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
/// and decrease level in sql structure.

void TBufferSQL2::DecrementLevel(TVirtualStreamerInfo *info)
{
   if (Stack()->GetElement())
      PopStack(); // for element
   PopStack();    // for streamerinfo

   // restore value of object data
   fCurrentData = Stack()->GetObjectData(kTRUE);

   if (gDebug > 2)
      Info("DecrementLevel", "Info: %s", info->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
/// and add/verify next element in sql tables
/// This calls allows separate data, correspondent to one class member, from another

void TBufferSQL2::SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type)
{
   if (Stack()->GetElement())
      PopStack(); // was with if (number > 0), i.e. not first element.
   TSQLStructure *curr = Stack();

   TStreamerInfo *info = curr->GetStreamerInfo();
   if (!info) {
      Error("SetStreamerElementNumber", "Error in structures stack");
      return;
   }

   WorkWithElement(elem, comp_type);
}

////////////////////////////////////////////////////////////////////////////////
/// This method inform buffer data of which class now
/// will be streamed. When reading, classversion should be specified
/// as was read by TBuffer::ReadVersion().
///
/// ClassBegin(), ClassEnd() & ClassMemeber() should be used in
/// custom class streamers to specify which kind of data are
/// now streamed to/from buffer. That information is used to correctly
/// convert class data to/from "normal" sql tables with meaningfull names
/// and correct datatypes. Without that functions data from custom streamer
/// will be saved as "raw" data in special _streamer_ table one value after another
/// Such MUST be used when object is written with standard ROOT streaming
/// procedure, but should be read back in custom streamer.
/// For example, custom streamer of TNamed class may look like:

void TBufferSQL2::ClassBegin(const TClass *cl, Version_t classversion)
{
   // void TNamed::Streamer(TBuffer &b)
   //   UInt_t R__s, R__c;
   //   if (b.IsReading()) {
   //      Version_t R__v = b.ReadVersion(&R__s, &R__c);
   //      b.ClassBegin(TNamed::Class(), R__v);
   //      b.ClassMember("TObject");
   //      TObject::Streamer(b);
   //      b.ClassMember("fName","TString");
   //      fName.Streamer(b);
   //      b.ClassMember("fTitle","TString");
   //      fTitle.Streamer(b);
   //      b.ClassEnd(TNamed::Class());
   //      b.SetBufferOffset(R__s+R__c+sizeof(UInt_t));
   //   } else {
   //      TNamed::Class()->WriteBuffer(b,this);
   //   }

   if (classversion < 0)
      classversion = cl->GetClassVersion();

   PushStack()->SetCustomClass(cl, classversion);

   if (gDebug > 2)
      Info("ClassBegin", "Class: %s", cl->GetName());

   WorkWithClass(cl->GetName(), classversion);
}

////////////////////////////////////////////////////////////////////////////////
/// Method indicates end of streaming of classdata in custom streamer.
/// See ClassBegin() method for more details.

void TBufferSQL2::ClassEnd(const TClass *cl)
{
   if (Stack()->GetType() == TSQLStructure::kSqlCustomElement)
      PopStack(); // for element
   PopStack();    // for streamerinfo

   // restore value of object data
   fCurrentData = Stack()->GetObjectData(kTRUE);

   if (gDebug > 2)
      Info("ClassEnd", "Class: %s", cl->GetName());
}

////////////////////////////////////////////////////////////////////////////////
/// Method indicates name and typename of class memeber,
/// which should be now streamed in custom streamer
/// Following combinations are supported:
/// see TBufferXML::ClassMember for the details.

void TBufferSQL2::ClassMember(const char *name, const char *typeName, Int_t arrsize1, Int_t arrsize2)
{
   if (!typeName)
      typeName = name;

   if (!name || (strlen(name) == 0)) {
      Error("ClassMember", "Invalid member name");
      fErrorFlag = 1;
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
         fErrorFlag = 1;
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
   } else

      if (typ_id == TStreamerInfo::kBase) {
      TClass *cl = TClass::GetClass(tname.Data());
      if (cl != 0) {
         TStreamerBase *b = new TStreamerBase(tname.Data(), "title", 0);
         b->SetBaseVersion(cl->GetClassVersion());
         elem = b;
      }
   } else

      if ((typ_id > 0) && (typ_id < 20)) {
      elem = new TStreamerBasicType(name, "title", 0, typ_id, typeName);
   } else

      if ((typ_id == TStreamerInfo::kObject) || (typ_id == TStreamerInfo::kTObject) ||
          (typ_id == TStreamerInfo::kTNamed)) {
      elem = new TStreamerObject(name, "title", 0, tname.Data());
   } else

      if (typ_id == TStreamerInfo::kObjectp) {
      elem = new TStreamerObjectPointer(name, "title", 0, tname.Data());
   } else

      if (typ_id == TStreamerInfo::kAny) {
      elem = new TStreamerObjectAny(name, "title", 0, tname.Data());
   } else

      if (typ_id == TStreamerInfo::kAnyp) {
      elem = new TStreamerObjectAnyPointer(name, "title", 0, tname.Data());
   } else

      if (typ_id == TStreamerInfo::kTString) {
      elem = new TStreamerString(name, "title", 0);
   }

   if (elem == 0) {
      Error("ClassMember", "Invalid combination name = %s type = %s", name, typeName);
      fErrorFlag = 1;
      return;
   }

   if (arrsize1 > 0) {
      elem->SetArrayDim(arrsize2 > 0 ? 2 : 1);
      elem->SetMaxIndex(0, arrsize1);
      if (arrsize2 > 0)
         elem->SetMaxIndex(1, arrsize2);
   }

   // return stack to CustomClass node
   if (Stack()->GetType() == TSQLStructure::kSqlCustomElement)
      PopStack();

   // we indicate that there is no streamerinfo
   WorkWithElement(elem, -1);
}

////////////////////////////////////////////////////////////////////////////////
/// This function is a part of IncrementLevel method.
/// Also used in StartClass method

void TBufferSQL2::WorkWithClass(const char *classname, Version_t classversion)
{
   if (IsReading()) {
      Long64_t objid = 0;

      //      if ((fCurrentData!=0) && fCurrentData->VerifyDataType(sqlio::ObjectInst, kFALSE))
      //        if (!fCurrentData->IsBlobData()) Info("WorkWithClass","Big problem %s", fCurrentData->GetValue());

      if ((fCurrentData != 0) && fCurrentData->IsBlobData() &&
          fCurrentData->VerifyDataType(sqlio::ObjectInst, kFALSE)) {
         objid = atoi(fCurrentData->GetValue());
         fCurrentData->ShiftToNextValue();
         TString sobjid;
         sobjid.Form("%lld", objid);
         Stack()->ChangeValueOnly(sobjid.Data());
      } else
         objid = Stack()->DefineObjectId(kTRUE);
      if (objid < 0) {
         Error("WorkWithClass", "cannot define object id");
         fErrorFlag = 1;
         return;
      }

      TSQLClassInfo *sqlinfo = fSQL->FindSQLClassInfo(classname, classversion);
      if (sqlinfo == 0) {
         Error("WorkWithClass", "Can not find table for class %s version %d", classname, classversion);
         fErrorFlag = 1;
         return;
      }

      TSQLObjectData *objdata = SqlObjectData(objid, sqlinfo);
      if (objdata == 0) {
         Error("WorkWithClass", "Request error for data of object %lld for class %s version %d", objid, classname,
               classversion);
         fErrorFlag = 1;
         return;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// This function is a part of SetStreamerElementNumber method.
/// It is introduced for reading of data for specified data memeber of class.
/// Used also in ReadFastArray methods to resolve problem of compressed data,
/// when several data memebers of the same basic type streamed with single ...FastArray call

void TBufferSQL2::WorkWithElement(TStreamerElement *elem, Int_t /* comp_type */)
{
   if (gDebug > 2)
      Info("WorkWithElement", "elem = %s", elem->GetName());

   TSQLStructure *stack = Stack(1);
   TStreamerInfo *info = stack->GetStreamerInfo();
   Int_t number = info ? info->GetElements()->IndexOf(elem) : -1;

   if (number >= 0)
      PushStack()->SetStreamerElement(elem, number);
   else
      PushStack()->SetCustomElement(elem);

   if (IsReading()) {

      if (fCurrentData == 0) {
         Error("WorkWithElement", "Object data is lost");
         fErrorFlag = 1;
         return;
      }

      fCurrentData = Stack()->GetObjectData(kTRUE);

      Int_t located = Stack()->LocateElementColumn(fSQL, this, fCurrentData);

      if (located == TSQLStructure::kColUnknown) {
         Error("WorkWithElement", "Cannot locate correct column in the table");
         fErrorFlag = 1;
         return;
      } else if ((located == TSQLStructure::kColObject) || (located == TSQLStructure::kColObjectArray) ||
                 (located == TSQLStructure::kColParent)) {
         // search again for object data while for BLOB it should be already assign
         fCurrentData = Stack()->GetObjectData(kTRUE);
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Suppressed function of TBuffer

TClass *TBufferSQL2::ReadClass(const TClass *, UInt_t *)
{
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Suppressed function of TBuffer

void TBufferSQL2::WriteClass(const TClass *)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Read version value from buffer
/// actually version is normally defined by table name
/// and kept in intermediate variable fReadVersionBuffer

Version_t TBufferSQL2::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass *)
{
   Version_t res = 0;

   if (start)
      *start = 0;
   if (bcnt)
      *bcnt = 0;

   if (fReadVersionBuffer >= 0) {
      res = fReadVersionBuffer;
      fReadVersionBuffer = -1;
      if (gDebug > 3)
         Info("ReadVersion", "from buffer = %d", (int)res);
   } else if ((fCurrentData != 0) && fCurrentData->IsBlobData() && fCurrentData->VerifyDataType(sqlio::Version)) {
      TString value = fCurrentData->GetValue();
      res = value.Atoi();
      if (gDebug > 3)
         Info("ReadVersion", "from blob %s = %d", fCurrentData->GetBlobPrefixName(), (int)res);
      fCurrentData->ShiftToNextValue();
   } else {
      Error("ReadVersion", "No correspondent tags to read version");
      fErrorFlag = 1;
   }

   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Copies class version to buffer, but not writes it to sql immidiately
/// Version will be used to produce complete table
/// name, which will include class version

UInt_t TBufferSQL2::WriteVersion(const TClass *cl, Bool_t /* useBcnt */)
{
   if (gDebug > 2)
      Info("WriteVersion", "cl:%s ver:%d", (cl ? cl->GetName() : "null"), (int)(cl ? cl->GetClassVersion() : 0));

   if (cl)
      Stack()->AddVersion(cl);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read object from buffer. Only used from TBuffer.

void *TBufferSQL2::ReadObjectAny(const TClass *)
{
   return SqlReadObject(0);
}

////////////////////////////////////////////////////////////////////////////////
/// ?????? Skip any kind of object from buffer
/// !!!!!! fix me, not yet implemented
/// Should be just skip of current column later

void TBufferSQL2::SkipObjectAny()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Write object to buffer. Only used from TBuffer

void TBufferSQL2::WriteObjectClass(const void *actualObjStart, const TClass *actualClass, Bool_t cacheReuse)
{
   if (gDebug > 2)
      Info("WriteObjectClass", "class %s", (actualClass ? actualClass->GetName() : " null"));
   SqlWriteObject(actualObjStart, actualClass, cacheReuse);
}

#define SQLReadArrayUncompress(vname, arrsize) \
   {                                           \
      while (indx < arrsize)                   \
         SqlReadBasic(vname[indx++]);          \
   }

#define SQLReadArrayCompress(vname, arrsize)                                       \
   {                                                                               \
      while (indx < arrsize) {                                                     \
         const char *name = fCurrentData->GetBlobPrefixName();                     \
         Int_t first, last;                                                        \
         if (strstr(name, sqlio::IndexSepar) == 0) {                               \
            sscanf(name, "[%d", &first);                                           \
            last = first;                                                          \
         } else {                                                                  \
            sscanf(name, "[%d..%d", &first, &last);                                \
         }                                                                         \
         if ((first != indx) || (last < first) || (last >= arrsize)) {             \
            Error("SQLReadArrayCompress", "Error reading array content %s", name); \
            fErrorFlag = 1;                                                        \
            break;                                                                 \
         }                                                                         \
         SqlReadBasic(vname[indx]);                                                \
         indx++;                                                                   \
         while (indx <= last)                                                      \
            vname[indx++] = vname[first];                                          \
      }                                                                            \
   }

// macro to read content of array with compression
#define SQLReadArrayContent(vname, arrsize, withsize)                                                 \
   {                                                                                                  \
      if (gDebug > 3)                                                                                 \
         Info("SQLReadArrayContent", "size %d", (int)(arrsize));                                      \
      PushStack()->SetArray(withsize ? arrsize : -1);                                                 \
      Int_t indx = 0;                                                                                 \
      if (fCurrentData->IsBlobData())                                                                 \
         SQLReadArrayCompress(vname, arrsize) else SQLReadArrayUncompress(vname, arrsize) PopStack(); \
      if (gDebug > 3)                                                                                 \
         Info("SQLReadArrayContent", "done");                                                         \
   }

// macro to read array, which include size attribute
#define TBufferSQL2_ReadArray(tname, vname) \
   {                                        \
      Int_t n = SqlReadArraySize();         \
      if (n <= 0)                           \
         return 0;                          \
      if (!vname)                           \
         vname = new tname[n];              \
      SQLReadArrayContent(vname, n, kTRUE); \
      return n;                             \
   }

////////////////////////////////////////////////////////////////////////////////
/// Read array of Bool_t from buffer

Int_t TBufferSQL2::ReadArray(Bool_t *&b)
{
   TBufferSQL2_ReadArray(Bool_t, b);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Char_t from buffer

Int_t TBufferSQL2::ReadArray(Char_t *&c)
{
   TBufferSQL2_ReadArray(Char_t, c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UChar_t from buffer

Int_t TBufferSQL2::ReadArray(UChar_t *&c)
{
   TBufferSQL2_ReadArray(UChar_t, c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Short_t from buffer

Int_t TBufferSQL2::ReadArray(Short_t *&h)
{
   TBufferSQL2_ReadArray(Short_t, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UShort_t from buffer

Int_t TBufferSQL2::ReadArray(UShort_t *&h)
{
   TBufferSQL2_ReadArray(UShort_t, h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Int_t from buffer

Int_t TBufferSQL2::ReadArray(Int_t *&i)
{
   TBufferSQL2_ReadArray(Int_t, i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UInt_t from buffer

Int_t TBufferSQL2::ReadArray(UInt_t *&i)
{
   TBufferSQL2_ReadArray(UInt_t, i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long_t from buffer

Int_t TBufferSQL2::ReadArray(Long_t *&l)
{
   TBufferSQL2_ReadArray(Long_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong_t from buffer

Int_t TBufferSQL2::ReadArray(ULong_t *&l)
{
   TBufferSQL2_ReadArray(ULong_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long64_t from buffer

Int_t TBufferSQL2::ReadArray(Long64_t *&l)
{
   TBufferSQL2_ReadArray(Long64_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong64_t from buffer

Int_t TBufferSQL2::ReadArray(ULong64_t *&l)
{
   TBufferSQL2_ReadArray(ULong64_t, l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float_t from buffer

Int_t TBufferSQL2::ReadArray(Float_t *&f)
{
   TBufferSQL2_ReadArray(Float_t, f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double_t from buffer

Int_t TBufferSQL2::ReadArray(Double_t *&d)
{
   TBufferSQL2_ReadArray(Double_t, d);
}

// macro to read static array, which include size attribute
#define TBufferSQL2_ReadStaticArray(vname)  \
   {                                        \
      Int_t n = SqlReadArraySize();         \
      if (n <= 0)                           \
         return 0;                          \
      if (!vname)                           \
         return 0;                          \
      SQLReadArrayContent(vname, n, kTRUE); \
      return n;                             \
   }

////////////////////////////////////////////////////////////////////////////////
/// Read array of Bool_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Bool_t *b)
{
   TBufferSQL2_ReadStaticArray(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Char_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Char_t *c)
{
   TBufferSQL2_ReadStaticArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UChar_t from buffer

Int_t TBufferSQL2::ReadStaticArray(UChar_t *c)
{
   TBufferSQL2_ReadStaticArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Short_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Short_t *h)
{
   TBufferSQL2_ReadStaticArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UShort_t from buffer

Int_t TBufferSQL2::ReadStaticArray(UShort_t *h)
{
   TBufferSQL2_ReadStaticArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Int_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Int_t *i)
{
   TBufferSQL2_ReadStaticArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UInt_t from buffer

Int_t TBufferSQL2::ReadStaticArray(UInt_t *i)
{
   TBufferSQL2_ReadStaticArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Long_t *l)
{
   TBufferSQL2_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong_t from buffer

Int_t TBufferSQL2::ReadStaticArray(ULong_t *l)
{
   TBufferSQL2_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long64_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Long64_t *l)
{
   TBufferSQL2_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong64_t from buffer

Int_t TBufferSQL2::ReadStaticArray(ULong64_t *l)
{
   TBufferSQL2_ReadStaticArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Float_t *f)
{
   TBufferSQL2_ReadStaticArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double_t from buffer

Int_t TBufferSQL2::ReadStaticArray(Double_t *d)
{
   TBufferSQL2_ReadStaticArray(d);
}

// macro to read content of array, which not include size of array
// macro also treat situation, when instead of one single array chain of several elements should be produced
#define TBufferSQL2_ReadFastArray(vname)     \
   {                                         \
      if (n <= 0)                            \
         return;                             \
      SQLReadArrayContent(vname, n, kFALSE); \
   }
////////////////////////////////////////////////////////////////////////////////
/// Read array of Bool_t from buffer

void TBufferSQL2::ReadFastArray(Bool_t *b, Int_t n)
{
   TBufferSQL2_ReadFastArray(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Char_t from buffer
/// if nodename==CharStar, read all array as string

void TBufferSQL2::ReadFastArray(Char_t *c, Int_t n)
{
   if ((n > 0) && fCurrentData->IsBlobData() && fCurrentData->VerifyDataType(sqlio::CharStar, kFALSE)) {
      const char *buf = SqlReadCharStarValue();
      if ((buf == 0) || (n <= 0))
         return;
      Int_t size = strlen(buf);
      if (size < n)
         size = n;
      memcpy(c, buf, size);
   } else {
      TBufferSQL2_ReadFastArray(c);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UChar_t from buffer

void TBufferSQL2::ReadFastArray(UChar_t *c, Int_t n)
{
   TBufferSQL2_ReadFastArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Short_t from buffer

void TBufferSQL2::ReadFastArray(Short_t *h, Int_t n)
{
   TBufferSQL2_ReadFastArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UShort_t from buffer

void TBufferSQL2::ReadFastArray(UShort_t *h, Int_t n)
{
   TBufferSQL2_ReadFastArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Int_t from buffer

void TBufferSQL2::ReadFastArray(Int_t *i, Int_t n)
{
   TBufferSQL2_ReadFastArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of UInt_t from buffer

void TBufferSQL2::ReadFastArray(UInt_t *i, Int_t n)
{
   TBufferSQL2_ReadFastArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long_t from buffer

void TBufferSQL2::ReadFastArray(Long_t *l, Int_t n)
{
   TBufferSQL2_ReadFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong_t from buffer

void TBufferSQL2::ReadFastArray(ULong_t *l, Int_t n)
{
   TBufferSQL2_ReadFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Long64_t from buffer

void TBufferSQL2::ReadFastArray(Long64_t *l, Int_t n)
{
   TBufferSQL2_ReadFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of ULong64_t from buffer

void TBufferSQL2::ReadFastArray(ULong64_t *l, Int_t n)
{
   TBufferSQL2_ReadFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Float_t from buffer

void TBufferSQL2::ReadFastArray(Float_t *f, Int_t n)
{
   TBufferSQL2_ReadFastArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of n characters from the I/O buffer.
/// Used only from TLeafC, dummy implementation here

void TBufferSQL2::ReadFastArrayString(Char_t *c, Int_t n)
{
   TBufferSQL2_ReadFastArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Read array of Double_t from buffer

void TBufferSQL2::ReadFastArray(Double_t *d, Int_t n)
{
   TBufferSQL2_ReadFastArray(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TBuffer::ReadFastArray(...) but
/// instead of calling cl->Streamer(obj,buf) call here
/// buf.StreamObject(obj, cl). In that case it is easy to understand where
/// object data is started and finished

void TBufferSQL2::ReadFastArray(void *start, const TClass *cl, Int_t n, TMemberStreamer *streamer,
                                const TClass *onFileClass)
{
   if (gDebug > 2)
      Info("ReadFastArray", "(void *");

   if (streamer) {
      StreamObjectExtra(start, streamer, cl, 0, onFileClass);
      //      (*streamer)(*this,start,0);
      return;
   }

   int objectSize = cl->Size();
   char *obj = (char *)start;
   char *end = obj + n * objectSize;

   for (; obj < end; obj += objectSize) {
      StreamObject(obj, cl, onFileClass);
   }
   //   TBuffer::ReadFastArray(start, cl, n, s);
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TBuffer::ReadFastArray(...) but
/// instead of calling cl->Streamer(obj,buf) call here
/// buf.StreamObject(obj, cl). In that case it is easy to understand where
/// object data is started and finished

void TBufferSQL2::ReadFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer,
                                const TClass *onFileClass)
{
   if (gDebug > 2)
      Info("ReadFastArray", "(void **  pre = %d  n = %d", isPreAlloc, n);

   Bool_t oldStyle = kFALSE; // flag used to reproduce old-style I/O actions for kSTLp

   if ((fIOVersion < 2) && !isPreAlloc) {
      TStreamerElement *elem = Stack(0)->GetElement();
      if (elem && ((elem->GetType() == TStreamerInfo::kSTLp) ||
                   (elem->GetType() == TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL)))
         oldStyle = kTRUE;
   }

   if (streamer) {
      if (isPreAlloc) {
         for (Int_t j = 0; j < n; j++) {
            if (!start[j])
               start[j] = ((TClass *)cl)->New();
         }
      }
      if (oldStyle)
         (*streamer)(*this, (void *)start, n);
      else
         StreamObjectExtra((void *)start, streamer, cl, 0, onFileClass);
      return;
   }

   if (!isPreAlloc) {

      for (Int_t j = 0; j < n; j++) {
         if (oldStyle) {
            if (!start[j])
               start[j] = ((TClass *)cl)->New();
            ((TClass *)cl)->Streamer(start[j], *this);
            continue;
         }

         // delete the object or collection
         if (start[j] && TStreamerInfo::CanDelete())
            ((TClass *)cl)->Destructor(start[j], kFALSE); // call delete and desctructor
         start[j] = ReadObjectAny(cl);
      }

   } else { // case //-> in comment

      for (Int_t j = 0; j < n; j++) {
         if (!start[j])
            start[j] = ((TClass *)cl)->New();
         StreamObject(start[j], cl, onFileClass);
      }
   }

   if (gDebug > 2)
      Info("ReadFastArray", "(void ** Done");

   //   TBuffer::ReadFastArray(startp, cl, n, isPreAlloc, s);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads array size, written in raw data table.
/// Used in ReadArray methods, where TBuffer need to read array size first.

Int_t TBufferSQL2::SqlReadArraySize()
{
   const char *value = SqlReadValue(sqlio::Array);
   if ((value == 0) || (strlen(value) == 0))
      return 0;
   Int_t sz = atoi(value);
   return sz;
}

// macro to write content of noncompressed array, not used
#define SQLWriteArrayNoncompress(vname, arrsize)     \
   {                                                 \
      for (Int_t indx = 0; indx < arrsize; indx++) { \
         SqlWriteBasic(vname[indx]);                 \
         Stack()->ChildArrayIndex(indx, 1);          \
      }                                              \
   }

// macro to write content of compressed array
#define SQLWriteArrayCompress(vname, arrsize)                     \
   {                                                              \
      Int_t indx = 0;                                             \
      while (indx < arrsize) {                                    \
         Int_t curr = indx;                                       \
         indx++;                                                  \
         while ((indx < arrsize) && (vname[indx] == vname[curr])) \
            indx++;                                               \
         SqlWriteBasic(vname[curr]);                              \
         Stack()->ChildArrayIndex(curr, indx - curr);             \
      }                                                           \
   }

#define SQLWriteArrayContent(vname, arrsize, withsize) \
   {                                                   \
      PushStack()->SetArray(withsize ? arrsize : -1);  \
      if (fCompressLevel > 0) {                        \
         SQLWriteArrayCompress(vname, arrsize)         \
      } else {                                         \
         SQLWriteArrayNoncompress(vname, arrsize)      \
      }                                                \
      PopStack();                                      \
   }

// macro to write array, which include size
#define TBufferSQL2_WriteArray(vname)        \
   {                                         \
      SQLWriteArrayContent(vname, n, kTRUE); \
   }

////////////////////////////////////////////////////////////////////////////////
/// Write array of Bool_t to buffer

void TBufferSQL2::WriteArray(const Bool_t *b, Int_t n)
{
   TBufferSQL2_WriteArray(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Char_t to buffer

void TBufferSQL2::WriteArray(const Char_t *c, Int_t n)
{
   TBufferSQL2_WriteArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UChar_t to buffer

void TBufferSQL2::WriteArray(const UChar_t *c, Int_t n)
{
   TBufferSQL2_WriteArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Short_t to buffer

void TBufferSQL2::WriteArray(const Short_t *h, Int_t n)
{
   TBufferSQL2_WriteArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UShort_t to buffer

void TBufferSQL2::WriteArray(const UShort_t *h, Int_t n)
{
   TBufferSQL2_WriteArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Int_ to buffer

void TBufferSQL2::WriteArray(const Int_t *i, Int_t n)
{
   TBufferSQL2_WriteArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UInt_t to buffer

void TBufferSQL2::WriteArray(const UInt_t *i, Int_t n)
{
   TBufferSQL2_WriteArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long_t to buffer

void TBufferSQL2::WriteArray(const Long_t *l, Int_t n)
{
   TBufferSQL2_WriteArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong_t to buffer

void TBufferSQL2::WriteArray(const ULong_t *l, Int_t n)
{
   TBufferSQL2_WriteArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long64_t to buffer

void TBufferSQL2::WriteArray(const Long64_t *l, Int_t n)
{
   TBufferSQL2_WriteArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong64_t to buffer

void TBufferSQL2::WriteArray(const ULong64_t *l, Int_t n)
{
   TBufferSQL2_WriteArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float_t to buffer

void TBufferSQL2::WriteArray(const Float_t *f, Int_t n)
{
   TBufferSQL2_WriteArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double_t to buffer

void TBufferSQL2::WriteArray(const Double_t *d, Int_t n)
{
   TBufferSQL2_WriteArray(d);
}

// write array without size attribute
// macro also treat situation, when instead of one single array chain of several elements should be produced
#define TBufferSQL2_WriteFastArray(vname)     \
   {                                          \
      if (n <= 0)                             \
         return;                              \
      SQLWriteArrayContent(vname, n, kFALSE); \
   }

////////////////////////////////////////////////////////////////////////////////
/// Write array of Bool_t to buffer

void TBufferSQL2::WriteFastArray(const Bool_t *b, Int_t n)
{
   TBufferSQL2_WriteFastArray(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Char_t to buffer
/// it will be reproduced as CharStar node with string as attribute

void TBufferSQL2::WriteFastArray(const Char_t *c, Int_t n)
{
   Bool_t usedefault = (n == 0);

   const Char_t *ccc = c;
   // check if no zeros in the array
   if (!usedefault)
      for (int i = 0; i < n; i++)
         if (*ccc++ == 0) {
            usedefault = kTRUE;
            break;
         }

   if (usedefault) {
      TBufferSQL2_WriteFastArray(c);
   } else {
      Char_t *buf = new Char_t[n + 1];
      memcpy(buf, c, n);
      buf[n] = 0;
      SqlWriteValue(buf, sqlio::CharStar);
      delete[] buf;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UChar_t to buffer

void TBufferSQL2::WriteFastArray(const UChar_t *c, Int_t n)
{
   TBufferSQL2_WriteFastArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Short_t to buffer

void TBufferSQL2::WriteFastArray(const Short_t *h, Int_t n)
{
   TBufferSQL2_WriteFastArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UShort_t to buffer

void TBufferSQL2::WriteFastArray(const UShort_t *h, Int_t n)
{
   TBufferSQL2_WriteFastArray(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Int_t to buffer

void TBufferSQL2::WriteFastArray(const Int_t *i, Int_t n)
{
   TBufferSQL2_WriteFastArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of UInt_t to buffer

void TBufferSQL2::WriteFastArray(const UInt_t *i, Int_t n)
{
   TBufferSQL2_WriteFastArray(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long_t to buffer

void TBufferSQL2::WriteFastArray(const Long_t *l, Int_t n)
{
   TBufferSQL2_WriteFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong_t to buffer

void TBufferSQL2::WriteFastArray(const ULong_t *l, Int_t n)
{
   TBufferSQL2_WriteFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Long64_t to buffer

void TBufferSQL2::WriteFastArray(const Long64_t *l, Int_t n)
{
   TBufferSQL2_WriteFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of ULong64_t to buffer

void TBufferSQL2::WriteFastArray(const ULong64_t *l, Int_t n)
{
   TBufferSQL2_WriteFastArray(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Float_t to buffer

void TBufferSQL2::WriteFastArray(const Float_t *f, Int_t n)
{
   TBufferSQL2_WriteFastArray(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of Double_t to buffer

void TBufferSQL2::WriteFastArray(const Double_t *d, Int_t n)
{
   TBufferSQL2_WriteFastArray(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Write array of n characters into the I/O buffer.
/// Used only by TLeafC, just dummy implementation here

void TBufferSQL2::WriteFastArrayString(const Char_t *c, Int_t n)
{
   TBufferSQL2_WriteFastArray(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TBuffer::WriteFastArray(...) but
/// instead of calling cl->Streamer(obj,buf) call here
/// buf.StreamObject(obj, cl). In that case it is easy to understand where
/// object data is started and finished

void TBufferSQL2::WriteFastArray(void *start, const TClass *cl, Int_t n, TMemberStreamer *streamer)
{
   if (streamer) {
      StreamObjectExtra(start, streamer, cl, 0);
      //      (*streamer)(*this, start, 0);
      return;
   }

   char *obj = (char *)start;
   if (!n)
      n = 1;
   int size = cl->Size();

   for (Int_t j = 0; j < n; j++, obj += size)
      StreamObject(obj, cl);

   //   TBuffer::WriteFastArray(start, cl, n, s);
}

////////////////////////////////////////////////////////////////////////////////
/// Same functionality as TBuffer::WriteFastArray(...) but
/// instead of calling cl->Streamer(obj,buf) call here
/// buf.StreamObject(obj, cl). In that case it is easy to understand where
/// object data is started and finished

Int_t TBufferSQL2::WriteFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer)
{

   Bool_t oldStyle = kFALSE; // flag used to reproduce old-style I/O actions for kSTLp

   if ((fIOVersion < 2) && !isPreAlloc) {
      TStreamerElement *elem = Stack(0)->GetElement();
      if (elem && ((elem->GetType() == TStreamerInfo::kSTLp) ||
                   (elem->GetType() == TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL)))
         oldStyle = kTRUE;
   }

   if (streamer) {
      if (oldStyle)
         (*streamer)(*this, (void *)start, n);
      else
         StreamObjectExtra((void *)start, streamer, cl, 0);
      return 0;
   }

   int strInfo = 0;

   Int_t res = 0;

   if (!isPreAlloc) {

      for (Int_t j = 0; j < n; j++) {
         // must write StreamerInfo if pointer is null
         if (!strInfo && !start[j] && !oldStyle)
            ForceWriteInfo(((TClass *)cl)->GetStreamerInfo(), kFALSE);
         strInfo = 2003;
         if (oldStyle)
            ((TClass *)cl)->Streamer(start[j], *this);
         else
            res |= WriteObjectAny(start[j], cl);
      }

   } else {
      // case //-> in comment

      for (Int_t j = 0; j < n; j++) {
         if (!start[j])
            start[j] = ((TClass *)cl)->New();
         StreamObject(start[j], cl);
      }
   }
   return res;

   //   return TBuffer::WriteFastArray(startp, cl, n, isPreAlloc, s);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream object to/from buffer

void TBufferSQL2::StreamObject(void *obj, const TClass *cl, const TClass *onFileClass)
{
   if (fIOVersion < 2) {
      TStreamerElement *elem = Stack(0)->GetElement();
      if (elem && (elem->GetType() == TStreamerInfo::kTObject)) {
         ((TObject *)obj)->TObject::Streamer(*this);
         return;
      } else if (elem && (elem->GetType() == TStreamerInfo::kTNamed)) {
         ((TNamed *)obj)->TNamed::Streamer(*this);
         return;
      }
   }

   if (gDebug > 1)
      Info("StreamObject", "class  %s", (cl ? cl->GetName() : "none"));
   if (IsReading())
      SqlReadObject(obj, 0, nullptr, 0, onFileClass);
   else
      SqlWriteObject(obj, cl, kTRUE);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream object to/from buffer

void TBufferSQL2::StreamObjectExtra(void *obj, TMemberStreamer *streamer, const TClass *cl, Int_t n,
                                    const TClass *onFileClass)
{
   if (!streamer)
      return;

   if (gDebug > 1)
      Info("StreamObjectExtra", "class = %s", cl->GetName());
   //   (*streamer)(*this, obj, n);

   if (IsReading())
      SqlReadObject(obj, 0, streamer, n, onFileClass);
   else
      SqlWriteObject(obj, cl, kTRUE, streamer, n);
}

// macro for right shift operator for basic type
#define TBufferSQL2_operatorin(vname) \
   {                                  \
      SqlReadBasic(vname);            \
   }

////////////////////////////////////////////////////////////////////////////////
/// Reads Bool_t value from buffer

void TBufferSQL2::ReadBool(Bool_t &b)
{
   TBufferSQL2_operatorin(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Char_t value from buffer

void TBufferSQL2::ReadChar(Char_t &c)
{
   TBufferSQL2_operatorin(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UChar_t value from buffer

void TBufferSQL2::ReadUChar(UChar_t &c)
{
   TBufferSQL2_operatorin(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Short_t value from buffer

void TBufferSQL2::ReadShort(Short_t &h)
{
   TBufferSQL2_operatorin(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UShort_t value from buffer

void TBufferSQL2::ReadUShort(UShort_t &h)
{
   TBufferSQL2_operatorin(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Int_t value from buffer

void TBufferSQL2::ReadInt(Int_t &i)
{
   TBufferSQL2_operatorin(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads UInt_t value from buffer

void TBufferSQL2::ReadUInt(UInt_t &i)
{
   TBufferSQL2_operatorin(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Long_t value from buffer

void TBufferSQL2::ReadLong(Long_t &l)
{
   TBufferSQL2_operatorin(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads ULong_t value from buffer

void TBufferSQL2::ReadULong(ULong_t &l)
{
   TBufferSQL2_operatorin(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Long64_t value from buffer

void TBufferSQL2::ReadLong64(Long64_t &l)
{
   TBufferSQL2_operatorin(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads ULong64_t value from buffer

void TBufferSQL2::ReadULong64(ULong64_t &l)
{
   TBufferSQL2_operatorin(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Float_t value from buffer

void TBufferSQL2::ReadFloat(Float_t &f)
{
   TBufferSQL2_operatorin(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads Double_t value from buffer

void TBufferSQL2::ReadDouble(Double_t &d)
{
   TBufferSQL2_operatorin(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Reads array of characters from buffer

void TBufferSQL2::ReadCharP(Char_t *c)
{
   const char *buf = SqlReadCharStarValue();
   if (buf)
      strcpy(c, buf);
}

////////////////////////////////////////////////////////////////////////////////
/// Read a TString

void TBufferSQL2::ReadTString(TString &s)
{
   if (fIOVersion < 2) {
      // original TBufferFile method can not be used, while used TString methods are private
      // try to reimplement close to the original
      Int_t nbig;
      UChar_t nwh;
      *this >> nwh;
      if (nwh == 0) {
         s.Resize(0);
      } else {
         if (nwh == 255)
            *this >> nbig;
         else
            nbig = nwh;

         char *data = new char[nbig];
         data[nbig] = 0;
         ReadFastArray(data, nbig);
         s = data;
         delete[] data;
      }
   } else {
      // TODO: new code - direct reading of string
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Write a TString

void TBufferSQL2::WriteTString(const TString &s)
{
   if (fIOVersion < 2) {
      // original TBufferFile method, keep for compatibility
      Int_t nbig = s.Length();
      UChar_t nwh;
      if (nbig > 254) {
         nwh = 255;
         *this << nwh;
         *this << nbig;
      } else {
         nwh = UChar_t(nbig);
         *this << nwh;
      }
      const char *data = s.Data();
      WriteFastArray(data, nbig);
   } else {
      // TODO: make writing of string directly
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read a std::string

void TBufferSQL2::ReadStdString(std::string *obj)
{
   if (fIOVersion < 2) {
      if (!obj) {
         Error("ReadStdString", "The std::string address is nullptr but should not");
         return;
      }
      Int_t nbig;
      UChar_t nwh;
      *this >> nwh;
      if (nwh == 0) {
         obj->clear();
      } else {
         if (obj->size()) {
            // Insure that the underlying data storage is not shared
            (*obj)[0] = '\0';
         }
         if (nwh == 255) {
            *this >> nbig;
            obj->resize(nbig, '\0');
            ReadFastArray((char *)obj->data(), nbig);
         } else {
            obj->resize(nwh, '\0');
            ReadFastArray((char *)obj->data(), nwh);
         }
      }
   } else {
      // TODO: direct reading of std string
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Write a std::string

void TBufferSQL2::WriteStdString(const std::string *obj)
{
   if (fIOVersion < 2) {
      if (!obj) {
         *this << (UChar_t)0;
         WriteFastArray("", 0);
         return;
      }

      UChar_t nwh;
      Int_t nbig = obj->length();
      if (nbig > 254) {
         nwh = 255;
         *this << nwh;
         *this << nbig;
      } else {
         nwh = UChar_t(nbig);
         *this << nwh;
      }
      WriteFastArray(obj->data(), nbig);
   } else {
      // TODO: make writing of string directly
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read a char* string

void TBufferSQL2::ReadCharStar(char *&s)
{
   delete[] s;
   s = nullptr;

   Int_t nch;
   *this >> nch;
   if (nch > 0) {
      s = new char[nch + 1];
      ReadFastArray(s, nch);
      s[nch] = 0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Write a char* string

void TBufferSQL2::WriteCharStar(char *s)
{
   Int_t nch = 0;
   if (s) {
      nch = strlen(s);
      *this << nch;
      WriteFastArray(s, nch);
   } else {
      *this << nch;
   }
}

// macro for right shift operator for basic types
#define TBufferSQL2_operatorout(vname) \
   {                                   \
      SqlWriteBasic(vname);            \
   }

////////////////////////////////////////////////////////////////////////////////
/// Writes Bool_t value to buffer

void TBufferSQL2::WriteBool(Bool_t b)
{
   TBufferSQL2_operatorout(b);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Char_t value to buffer

void TBufferSQL2::WriteChar(Char_t c)
{
   TBufferSQL2_operatorout(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UChar_t value to buffer

void TBufferSQL2::WriteUChar(UChar_t c)
{
   TBufferSQL2_operatorout(c);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Short_t value to buffer

void TBufferSQL2::WriteShort(Short_t h)
{
   TBufferSQL2_operatorout(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UShort_t value to buffer

void TBufferSQL2::WriteUShort(UShort_t h)
{
   TBufferSQL2_operatorout(h);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Int_t value to buffer

void TBufferSQL2::WriteInt(Int_t i)
{
   TBufferSQL2_operatorout(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes UInt_t value to buffer

void TBufferSQL2::WriteUInt(UInt_t i)
{
   TBufferSQL2_operatorout(i);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Long_t value to buffer

void TBufferSQL2::WriteLong(Long_t l)
{
   TBufferSQL2_operatorout(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes ULong_t value to buffer

void TBufferSQL2::WriteULong(ULong_t l)
{
   TBufferSQL2_operatorout(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Long64_t value to buffer

void TBufferSQL2::WriteLong64(Long64_t l)
{
   TBufferSQL2_operatorout(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes ULong64_t value to buffer

void TBufferSQL2::WriteULong64(ULong64_t l)
{
   TBufferSQL2_operatorout(l);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Float_t value to buffer

void TBufferSQL2::WriteFloat(Float_t f)
{
   TBufferSQL2_operatorout(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes Double_t value to buffer

void TBufferSQL2::WriteDouble(Double_t d)
{
   TBufferSQL2_operatorout(d);
}

////////////////////////////////////////////////////////////////////////////////
/// Writes array of characters to buffer

void TBufferSQL2::WriteCharP(const Char_t *c)
{
   SqlWriteValue(c, sqlio::CharStar);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Char_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Char_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   return SqlWriteValue(buf, sqlio::Char);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Short_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Short_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%hd", value);
   return SqlWriteValue(buf, sqlio::Short);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Int_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Int_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   return SqlWriteValue(buf, sqlio::Int);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Long_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Long_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%ld", value);
   return SqlWriteValue(buf, sqlio::Long);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Long64_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Long64_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%lld", value);
   return SqlWriteValue(buf, sqlio::Long64);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Float_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Float_t value)
{
   char buf[200];
   snprintf(buf, sizeof(buf), TSQLServer::GetFloatFormat(), value);
   return SqlWriteValue(buf, sqlio::Float);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Double_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Double_t value)
{
   char buf[128];
   snprintf(buf, sizeof(buf), TSQLServer::GetFloatFormat(), value);
   return SqlWriteValue(buf, sqlio::Double);
}

////////////////////////////////////////////////////////////////////////////////
/// converts Bool_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(Bool_t value)
{
   return SqlWriteValue(value ? sqlio::True : sqlio::False, sqlio::Bool);
}

////////////////////////////////////////////////////////////////////////////////
/// converts UChar_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(UChar_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   return SqlWriteValue(buf, sqlio::UChar);
}

////////////////////////////////////////////////////////////////////////////////
/// converts UShort_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(UShort_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%hu", value);
   return SqlWriteValue(buf, sqlio::UShort);
}

////////////////////////////////////////////////////////////////////////////////
/// converts UInt_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(UInt_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   return SqlWriteValue(buf, sqlio::UInt);
}

////////////////////////////////////////////////////////////////////////////////
/// converts ULong_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(ULong_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), "%lu", value);
   return SqlWriteValue(buf, sqlio::ULong);
}

////////////////////////////////////////////////////////////////////////////////
/// converts ULong64_t to string and creates correspondent sql structure

Bool_t TBufferSQL2::SqlWriteBasic(ULong64_t value)
{
   char buf[50];
   snprintf(buf, sizeof(buf), FULong64, value);
   return SqlWriteValue(buf, sqlio::ULong64);
}

//______________________________________________________________________________

Bool_t TBufferSQL2::SqlWriteValue(const char *value, const char *tname)
{
   // create structure in stack, which holds specified value

   Stack()->AddValue(value, tname);

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Char_t value

void TBufferSQL2::SqlReadBasic(Char_t &value)
{
   const char *res = SqlReadValue(sqlio::Char);
   if (res) {
      int n;
      sscanf(res, "%d", &n);
      value = n;
   } else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Short_t value

void TBufferSQL2::SqlReadBasic(Short_t &value)
{
   const char *res = SqlReadValue(sqlio::Short);
   if (res)
      sscanf(res, "%hd", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Int_t value

void TBufferSQL2::SqlReadBasic(Int_t &value)
{
   const char *res = SqlReadValue(sqlio::Int);
   if (res)
      sscanf(res, "%d", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Long_t value

void TBufferSQL2::SqlReadBasic(Long_t &value)
{
   const char *res = SqlReadValue(sqlio::Long);
   if (res)
      sscanf(res, "%ld", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Long64_t value

void TBufferSQL2::SqlReadBasic(Long64_t &value)
{
   const char *res = SqlReadValue(sqlio::Long64);
   if (res)
      sscanf(res, FLong64, &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Float_t value

void TBufferSQL2::SqlReadBasic(Float_t &value)
{
   const char *res = SqlReadValue(sqlio::Float);
   if (res)
      sscanf(res, "%f", &value);
   else
      value = 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Double_t value

void TBufferSQL2::SqlReadBasic(Double_t &value)
{
   const char *res = SqlReadValue(sqlio::Double);
   if (res)
      sscanf(res, "%lf", &value);
   else
      value = 0.;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to Bool_t value

void TBufferSQL2::SqlReadBasic(Bool_t &value)
{
   const char *res = SqlReadValue(sqlio::Bool);
   if (res)
      value = (strcmp(res, sqlio::True) == 0);
   else
      value = kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to UChar_t value

void TBufferSQL2::SqlReadBasic(UChar_t &value)
{
   const char *res = SqlReadValue(sqlio::UChar);
   if (res) {
      unsigned int n;
      sscanf(res, "%ud", &n);
      value = n;
   } else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to UShort_t value

void TBufferSQL2::SqlReadBasic(UShort_t &value)
{
   const char *res = SqlReadValue(sqlio::UShort);
   if (res)
      sscanf(res, "%hud", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to UInt_t value

void TBufferSQL2::SqlReadBasic(UInt_t &value)
{
   const char *res = SqlReadValue(sqlio::UInt);
   if (res)
      sscanf(res, "%u", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to ULong_t value

void TBufferSQL2::SqlReadBasic(ULong_t &value)
{
   const char *res = SqlReadValue(sqlio::ULong);
   if (res)
      sscanf(res, "%lu", &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read current value from table and convert it to ULong64_t value

void TBufferSQL2::SqlReadBasic(ULong64_t &value)
{
   const char *res = SqlReadValue(sqlio::ULong64);
   if (res)
      sscanf(res, FULong64, &value);
   else
      value = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read string value from current stack node

const char *TBufferSQL2::SqlReadValue(const char *tname)
{
   if (fErrorFlag > 0)
      return 0;

   if (!fCurrentData) {
      Error("SqlReadValue", "No object data to read from");
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

   if (gDebug > 4)
      Info("SqlReadValue", "%s = %s", tname, fReadBuffer.Data());

   return fReadBuffer.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Read CharStar value, if it has special code, request it from large table

const char *TBufferSQL2::SqlReadCharStarValue()
{
   const char *res = SqlReadValue(sqlio::CharStar);
   if ((res == 0) || (fSQL == 0))
      return 0;

   Long64_t objid = Stack()->DefineObjectId(kTRUE);

   Int_t strid = fSQL->IsLongStringCode(objid, res);
   if (strid <= 0)
      return res;

   fSQL->GetLongString(objid, strid, fReadBuffer);

   return fReadBuffer.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// Push stack with structurual information about streamed object

TSQLStructure *TBufferSQL2::PushStack()
{
   TSQLStructure *res = new TSQLStructure;
   if (fStk == 0) {
      fStructure = res;
   } else {
      fStk->Add(res);
   }

   fStk = res; // add in the stack
   return fStk;
}

////////////////////////////////////////////////////////////////////////////////
/// Pop stack

TSQLStructure *TBufferSQL2::PopStack()
{
   if (fStk == 0)
      return 0;
   fStk = fStk->GetParent();
   return fStk;
}

////////////////////////////////////////////////////////////////////////////////
/// returns head of stack

TSQLStructure *TBufferSQL2::Stack(Int_t depth)
{
   TSQLStructure *curr = fStk;
   while ((depth-- > 0) && (curr != 0))
      curr = curr->GetParent();
   return curr;
}

////////////////////////////////////////////////////////////////////////////////
/// set printf format for float/double members, default "%e"
/// changes global TSQLServer variable

void TBufferSQL2::SetFloatFormat(const char *fmt)
{
   TSQLServer::SetFloatFormat(fmt);
}

////////////////////////////////////////////////////////////////////////////////
/// return current printf format for float/double members, default "%e"
/// return format, hold by TSQLServer

const char *TBufferSQL2::GetFloatFormat()
{
   return TSQLServer::GetFloatFormat();
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object of the specified class is already in
/// the buffer. Returns kTRUE if object already in the buffer,
/// kFALSE otherwise (also if obj is 0 ).

Bool_t TBufferSQL2::CheckObject(const TObject *obj)
{
   return CheckObject(obj, TObject::Class());
}

////////////////////////////////////////////////////////////////////////////////
/// Check if the specified object of the specified class is already in
/// the buffer. Returns kTRUE if object already in the buffer,
/// kFALSE otherwise (also if obj is 0 ).

Bool_t TBufferSQL2::CheckObject(const void *obj, const TClass *ptrClass)
{
   if (!obj || !ptrClass || !fObjMap)
      return kFALSE;

   TClass *clActual = ptrClass->GetActualClass(obj);

   const char *temp = (const char *)obj;

   if (clActual && (ptrClass != clActual))
      temp -= clActual->GetBaseClassOffset(ptrClass);

   return fObjMap->GetValue(TString::Hash(&temp, sizeof(void *)), (Long_t)temp) != 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return current streamer info element

TVirtualStreamerInfo *TBufferSQL2::GetInfo()
{
   return Stack()->GetStreamerInfo();
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.
/// The collection needs to be a split TClonesArray or a split vector of pointers.

Int_t TBufferSQL2::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, obj);
         (*iter)(*this, obj);
      }

   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
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

Int_t TBufferSQL2::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
                                       void *end_collection)
{
   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
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
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection);
      }
   }

   DecrementLevel(info);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Read one collection of objects from the buffer using the StreamerInfoLoopAction.

Int_t TBufferSQL2::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection,
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
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this, arr0);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }

   } else {
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for (TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin(); iter != end;
           ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem, (*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this, start_collection, end_collection, loopconfig);
      }
   }

   DecrementLevel(info);
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

Int_t TBufferSQL2::ReadClassBuffer(const TClass *cl, void *pointer, Int_t version, UInt_t start, UInt_t count,
                                   const TClass *onFileClass)
{

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   /////////////////////////////////////////////////////////////////////////////

   TStreamerInfo *sinfo = nullptr;
   if (onFileClass) {
      sinfo = (TStreamerInfo *)cl->GetConversionStreamerInfo(onFileClass, version);
      if (!sinfo) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length());
         CheckByteCount(start, count, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   /////////////////////////////////////////////////////////////////////////////
   /// The StreamerInfo should exist at this point.

   else {
      R__LOCKGUARD(gInterpreterMutex);
      auto infos = cl->GetStreamerInfos();
      auto ninfos = infos->GetSize();
      if (version < -1 || version >= ninfos) {
         Error("ReadBuffer1", "class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
               cl->GetName(), version, Length());
         CheckByteCount(start, count, cl);
         return 0;
      }
      sinfo = (TStreamerInfo *)infos->At(version);
      if (!sinfo) {
         // Unless the data is coming via a socket connection from with schema evolution
         // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
         // one for the current version, otherwise let's complain ...
         // We could also get here if there old class version was '1' and the new class version is higher than 1
         // AND the checksum is the same.
         if (version == cl->GetClassVersion() || version == 1) {
            const_cast<TClass *>(cl)->BuildRealData(pointer);
            // This creation is alright since we just checked within the
            // current 'locked' section.
            sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
            const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
            if (gDebug > 0)
               printf("Creating StreamerInfo for class: %s, version: %d\n", cl->GetName(), version);
            sinfo->Build();
         } else if (version == 0) {
            // When the object was written the class was version zero, so
            // there is no StreamerInfo to be found.
            // Check that the buffer position corresponds to the byte count.
            CheckByteCount(start, count, cl);
            return 0;
         } else {
            Error("ReadClassBuffer",
                  "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                  version, cl->GetName(), Length());
            CheckByteCount(start, count, cl);
            return 0;
         }
      } else if (!sinfo->IsCompiled()) { // Note this read is protected by the above lock.
         // Streamer info has not been compiled, but exists.
         // Therefore it was read in from a file and we have to do schema evolution.
         const_cast<TClass *>(cl)->BuildRealData(pointer);
         sinfo->BuildOld();
      }
   }

   // Deserialize the object.
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)pointer);
   if (sinfo->IsRecovered())
      count = 0;

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(start, count, cl);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Deserialize information from a buffer into an object.
///
/// Note: This function is called by the xxx::Streamer()
/// functions in rootcint-generated dictionaries.
///

Int_t TBufferSQL2::ReadClassBuffer(const TClass *cl, void *pointer, const TClass *onFileClass)
{
   // Read the class version from the buffer.
   UInt_t R__s = 0; // Start of object.
   UInt_t R__c = 0; // Count of bytes.
   Version_t version;

   if (onFileClass)
      version = ReadVersion(&R__s, &R__c, onFileClass);
   else
      version = ReadVersion(&R__s, &R__c, cl);

   Bool_t v2file = kFALSE;
   TFile *file = (TFile *)GetParent();
   if (file && file->GetVersion() < 30000) {
      version = -1; // This is old file
      v2file = kTRUE;
   }

   //---------------------------------------------------------------------------
   // The ondisk class has been specified so get foreign streamer info
   /////////////////////////////////////////////////////////////////////////////

   TStreamerInfo *sinfo = nullptr;
   if (onFileClass) {
      sinfo = (TStreamerInfo *)cl->GetConversionStreamerInfo(onFileClass, version);
      if (!sinfo) {
         Error("ReadClassBuffer",
               "Could not find the right streamer info to convert %s version %d into a %s, object skipped at offset %d",
               onFileClass->GetName(), version, cl->GetName(), Length());
         CheckByteCount(R__s, R__c, onFileClass);
         return 0;
      }
   }
   //---------------------------------------------------------------------------
   // Get local streamer info
   /////////////////////////////////////////////////////////////////////////////
   /// The StreamerInfo should exist at this point.

   else {
      TStreamerInfo *guess = (TStreamerInfo *)cl->GetLastReadInfo();
      if (guess && guess->GetClassVersion() == version) {
         sinfo = guess;
      } else {
         // The last one is not the one we are looking for.
         {
            R__LOCKGUARD(gInterpreterMutex);

            const TObjArray *infos = cl->GetStreamerInfos();
            Int_t infocapacity = infos->Capacity();
            if (infocapacity) {
               if (version < -1 || version >= infocapacity) {
                  Error("ReadClassBuffer",
                        "class: %s, attempting to access a wrong version: %d, object skipped at offset %d",
                        cl->GetName(), version, Length());
                  CheckByteCount(R__s, R__c, cl);
                  return 0;
               }
               sinfo = (TStreamerInfo *)infos->UncheckedAt(version);
               if (sinfo) {
                  if (!sinfo->IsCompiled()) {
                     // Streamer info has not been compiled, but exists.
                     // Therefore it was read in from a file and we have to do schema evolution?
                     R__LOCKGUARD(gInterpreterMutex);
                     const_cast<TClass *>(cl)->BuildRealData(pointer);
                     sinfo->BuildOld();
                  }
                  // If the compilation succeeded, remember this StreamerInfo.
                  // const_cast okay because of the lock on gInterpreterMutex.
                  if (sinfo->IsCompiled())
                     const_cast<TClass *>(cl)->SetLastReadInfo(sinfo);
               }
            }
         }

         if (!sinfo) {
            // Unless the data is coming via a socket connection from with schema evolution
            // (tracking) was not enabled.  So let's create the StreamerInfo if it is the
            // one for the current version, otherwise let's complain ...
            // We could also get here when reading a file prior to the introduction of StreamerInfo.
            // We could also get here if there old class version was '1' and the new class version is higher than 1
            // AND the checksum is the same.
            if (v2file || version == cl->GetClassVersion() || version == 1) {
               R__LOCKGUARD(gInterpreterMutex);

               // We need to check if another thread did not get here first
               // and did the StreamerInfo creation already.
               auto infos = cl->GetStreamerInfos();
               auto ninfos = infos->GetSize();
               if (!(version < -1 || version >= ninfos)) {
                  sinfo = (TStreamerInfo *)infos->At(version);
               }
               if (!sinfo) {
                  const_cast<TClass *>(cl)->BuildRealData(pointer);
                  sinfo = new TStreamerInfo(const_cast<TClass *>(cl));
                  sinfo->SetClassVersion(version);
                  const_cast<TClass *>(cl)->RegisterStreamerInfo(sinfo);
                  if (gDebug > 0)
                     printf("Creating StreamerInfo for class: %s, version: %d\n", cl->GetName(), version);
                  if (v2file) {
                     sinfo->Build();             // Get the elements.
                     sinfo->Clear("build");      // Undo compilation.
                     sinfo->BuildEmulated(file); // Fix the types and redo compilation.
                  } else {
                     sinfo->Build();
                  }
               }
            } else if (version == 0) {
               // When the object was written the class was version zero, so
               // there is no StreamerInfo to be found.
               // Check that the buffer position corresponds to the byte count.
               CheckByteCount(R__s, R__c, cl);
               return 0;
            } else {
               Error("ReadClassBuffer",
                     "Could not find the StreamerInfo for version %d of the class %s, object skipped at offset %d",
                     version, cl->GetName(), Length());
               CheckByteCount(R__s, R__c, cl);
               return 0;
            }
         }
      }
   }

   // deserialize the object
   ApplySequence(*(sinfo->GetReadTextActions()), (char *)pointer);
   if (sinfo->TStreamerInfo::IsRecovered())
      R__c = 0; // 'TStreamerInfo::' avoids going via a virtual function.

   // Check that the buffer position corresponds to the byte count.
   CheckByteCount(R__s, R__c, cl);

   if (gDebug > 2)
      printf(" ReadBuffer for class: %s has read %d bytes\n", cl->GetName(), R__c);

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Function called by the Streamer functions to serialize object at p
/// to buffer b. The optional argument info may be specified to give an
/// alternative StreamerInfo instead of using the default StreamerInfo
/// automatically built from the class definition.
/// For more information, see class TStreamerInfo.

Int_t TBufferSQL2::WriteClassBuffer(const TClass *cl, void *pointer)
{
   // build the StreamerInfo if first time for the class
   TStreamerInfo *sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
   if (!sinfo) {
      // Have to be sure between the check and the taking of the lock if the current streamer has changed
      R__LOCKGUARD(gInterpreterMutex);
      sinfo = (TStreamerInfo *)const_cast<TClass *>(cl)->GetCurrentStreamerInfo();
      if (!sinfo) {
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
   UInt_t R__c = WriteVersion(cl, kTRUE);

   // NOTE: In the future Philippe wants this to happen via a custom action
   TagStreamerInfo(sinfo);
   ApplySequence(*(sinfo->GetWriteTextActions()), (char *)pointer);

   // write the byte count at the start of the buffer
   SetByteCount(R__c, kTRUE);

   if (gDebug > 2)
      printf(" WriteBuffer for class: %s version %d has written %d bytes\n", cl->GetName(), cl->GetClassVersion(),
             UInt_t(fBufCur - fBuffer) - R__c - (UInt_t)sizeof(UInt_t));
   return 0;
}
