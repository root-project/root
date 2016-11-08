// @(#)root/sql:$Id$
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
#define FLong64    "%I64d"
#define FULong64   "%I64u"
#else
#define FLong64    "%lld"
#define FULong64   "%llu"
#endif

ClassImp(TBufferSQL2);

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2() :
   TBufferFile(),
   fSQL(0),
   fStructure(0),
   fStk(0),
   fObjMap(0),
   fReadBuffer(),
   fErrorFlag(0),
   fExpectedChain(kFALSE),
   fCompressLevel(0),
   fReadVersionBuffer(-1),
   fObjIdCounter(1),
   fIgnoreVerification(kFALSE),
   fCurrentData(0),
   fObjectsInfos(0),
   fFirstObjId(0),
   fLastObjId(0),
   fPoolsMap(0)
{
   // Default constructor, should not be used
}

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2(TBuffer::EMode mode) :
   TBufferFile(mode),
   fSQL(0),
   fStructure(0),
   fStk(0),
   fObjMap(0),
   fReadBuffer(),
   fErrorFlag(0),
   fExpectedChain(kFALSE),
   fCompressLevel(0),
   fReadVersionBuffer(-1),
   fObjIdCounter(1),
   fIgnoreVerification(kFALSE),
   fCurrentData(0),
   fObjectsInfos(0),
   fFirstObjId(0),
   fLastObjId(0),
   fPoolsMap(0)
{
   // Creates buffer object to serailize/deserialize data to/from sql.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   SetParent(0);
   SetBit(kCannotHandleMemberWiseStreaming);
   SetBit(kTextBasedStreaming);
}

//______________________________________________________________________________
TBufferSQL2::TBufferSQL2(TBuffer::EMode mode, TSQLFile* file) :
   TBufferFile(mode),
   fSQL(0),
   fStructure(0),
   fStk(0),
   fObjMap(0),
   fReadBuffer(),
   fErrorFlag(0),
   fExpectedChain(kFALSE),
   fCompressLevel(0),
   fReadVersionBuffer(-1),
   fObjIdCounter(1),
   fIgnoreVerification(kFALSE),
   fCurrentData(0),
   fObjectsInfos(0),
   fFirstObjId(0),
   fLastObjId(0),
   fPoolsMap(0)
{
   // Creates buffer object to serailize/deserialize data to/from sql.
   // This constructor should be used, if data from buffer supposed to be stored in file.
   // Mode should be either TBuffer::kRead or TBuffer::kWrite.

   fBufSize = 1000000000;

   // for TClonesArray recognize if this is special case
   SetBit(kCannotHandleMemberWiseStreaming);
   SetBit(kTextBasedStreaming);

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

   if (fStructure!=0) {
      delete fStructure;
      fStructure = 0;
   }

   if (fObjectsInfos!=0) {
      fObjectsInfos->Delete();
      delete fObjectsInfos;
   }

   if (fPoolsMap!=0) {
      fPoolsMap->DeleteValues();
      delete fPoolsMap;
   }
}

//______________________________________________________________________________
TSQLStructure* TBufferSQL2::SqlWriteAny(const void* obj, const TClass* cl, Long64_t objid)
{
   // Convert object of any class to sql structures
   // Return pointer on created TSQLStructure
   // TSQLStructure object will be owned by TBufferSQL2

   fErrorFlag = 0;

   fStructure = 0;

   fFirstObjId = objid;
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
void* TBufferSQL2::SqlReadAny(Long64_t keyid, Long64_t objid, TClass** cl, void* obj)
{
   // Recreate object from sql structure.
   // Return pointer to read object.
   // if (cl!=0) returns pointer to class of object

   if (cl) *cl = 0;
   if (fSQL==0) return 0;

   fCurrentData = 0;
   fErrorFlag = 0;

   fReadVersionBuffer = -1;

   fObjectsInfos = fSQL->SQLObjectsInfo(keyid);
//   fObjectsInfos = 0;
   fFirstObjId = objid;
   fLastObjId = objid;
   if (fObjectsInfos!=0) {
      TSQLObjectInfo* objinfo = (TSQLObjectInfo*) fObjectsInfos->Last();
      if (objinfo!=0) fLastObjId = objinfo->GetObjId();
   }

   return SqlReadObjectDirect(obj, cl, objid);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlObjectInfo(Long64_t objid, TString& clname, Version_t& version)
{
// Returns object info like classname and version
// Should be taken from buffer, which is produced in the begginnig

   if ((objid<0) || (fObjectsInfos==0)) return kFALSE;

 //  if (fObjectsInfos==0) return fSQL->SQLObjectInfo(objid, clname, version);

   // suppose that objects info are sorted out

   Long64_t shift = objid - fFirstObjId;

   TSQLObjectInfo* info = 0;
   if ((shift>=0) && (shift<=fObjectsInfos->GetLast())) {
      info = (TSQLObjectInfo*) fObjectsInfos->At(shift);
      if (info->GetObjId()!=objid) info = 0;
   }

   if (info==0) {
      // I hope, i will never get inside it
      Info("SqlObjectInfo", "Standard not works %lld", objid);
      for (Int_t n=0;n<=fObjectsInfos->GetLast();n++) {
         info = (TSQLObjectInfo*) fObjectsInfos->At(n);
         if (info->GetObjId()==objid) break;
         info = 0;
      }
   }

   if (info==0) return kFALSE;

   clname = info->GetObjClassName();
   version = info->GetObjVersion();
   return kTRUE;
}


//______________________________________________________________________________
TSQLObjectData* TBufferSQL2::SqlObjectData(Long64_t objid, TSQLClassInfo* sqlinfo)
{
   // creates TSQLObjectData for specifed object id and specified class
   // Object data for each class can be stored in two different tables.
   // First table contains data in column-wise form for simple types like integer,
   // strings and so on when second table contains any other data which cannot
   // be converted into column-wise representation.
   // TSQLObjectData will contain results of the requests to both such tables for
   // concrete object id.

   TSQLResult *classdata = 0;
   TSQLRow *classrow = 0;

   if (sqlinfo->IsClassTableExist()) {

      TSQLObjectDataPool* pool = 0;

      if (fPoolsMap!=0)
        pool = (TSQLObjectDataPool*) fPoolsMap->GetValue(sqlinfo);

      if ((pool==0) && (fLastObjId>=fFirstObjId)) {
         if (gDebug>4) Info("SqlObjectData","Before request to %s",sqlinfo->GetClassTableName());
         TSQLResult *alldata = fSQL->GetNormalClassDataAll(fFirstObjId, fLastObjId, sqlinfo);
         if (gDebug>4) Info("SqlObjectData","After request res = 0x%lx",(Long_t)alldata);
         if (alldata==0) {
            Error("SqlObjectData","Cannot get data from table %s",sqlinfo->GetClassTableName());
            return 0;
         }

         if (fPoolsMap==0) fPoolsMap = new TMap();
         pool = new TSQLObjectDataPool(sqlinfo, alldata);
         fPoolsMap->Add(sqlinfo, pool);
      }

      if (pool==0) return 0;

      if (pool->GetSqlInfo()!=sqlinfo) {
         Error("SqlObjectData","Missmatch in pools map !!! CANNOT BE !!!");
         return 0;
      }

      classdata = pool->GetClassData();

      classrow = pool->GetObjectRow(objid);
      if (classrow==0) {
         Error("SqlObjectData","Can not find row for objid = %lld in table %s", objid, sqlinfo->GetClassTableName());
         return 0;
      }
   }

   TSQLResult *blobdata = 0;
   TSQLStatement* blobstmt = fSQL->GetBlobClassDataStmt(objid, sqlinfo);

   if (blobstmt==0) blobdata = fSQL->GetBlobClassData(objid, sqlinfo);

   return new TSQLObjectData(sqlinfo, objid, classdata, classrow, blobdata, blobstmt);
}

//______________________________________________________________________________
void TBufferSQL2::WriteObject(const TObject *obj)
{
   // Convert object into sql structures.
   // !!! Should be used only by TBufferSQL2 itself.
   // Use SqlWrite() functions to convert your object to sql
   // Redefined here to avoid gcc 3.x warning

   TBufferFile::WriteObject(obj);
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

   Long64_t objid = -1;

   if (cl==0) obj = 0;

   if (obj==0)
      objid = 0;
   else
   if (fObjMap!=0) {
      ULong_t hash = TString::Hash(&obj, sizeof(void*));
      Long_t value = fObjMap->GetValue(hash, (Long_t) obj);
      if (value>0)
         objid = fFirstObjId + value - 1;
   }

   if (gDebug>1)
      cout << "    Find objectid = " << objid << endl;

   if (objid>=0) {
      Stack()->SetObjectPointer(objid);
      PopStack();
      return objid;
   }

   objid = fObjIdCounter++;

   Stack()->SetObjectRef(objid, cl);

   ULong_t hash = TString::Hash(&obj, sizeof(void*));
   if (fObjMap==0) fObjMap = new TExMap();
   if (fObjMap->GetValue(hash, (Long_t) obj)==0)
      fObjMap->Add(hash, (Long_t) obj, (Long_t) objid - fFirstObjId + 1);

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
void* TBufferSQL2::SqlReadObject(void* obj, TClass** cl, TMemberStreamer *streamer, Int_t streamer_index, const TClass *onFileClass)
{
   // Read object from the buffer

   if (cl) *cl = 0;

   if (fErrorFlag>0) return obj;

   Bool_t findptr = kFALSE;

   const char* refid = fCurrentData->GetValue();
   if ((refid==0) || (strlen(refid)==0)) {
      Error("SqlReadObject","Invalid object reference value");
      fErrorFlag = 1;
      return obj;
   }

   Long64_t objid = -1;
   sscanf(refid, FLong64, &objid);

   if (gDebug>2)
      Info("SqlReadObject","Starting objid = %lld column=%s", objid, fCurrentData->GetLocatedField());

   if (!fCurrentData->IsBlobData() ||
       fCurrentData->VerifyDataType(sqlio::ObjectPtr,kFALSE)) {
      if (objid==0) {
         obj = 0;
         findptr = kTRUE;
      } else {
         if (objid==-1) {
            findptr = kTRUE;
         } else {
            if ((fObjMap!=0) && (objid>=fFirstObjId)) {
               void* obj1 = (void*) (Long_t)fObjMap->GetValue((Long_t) objid - fFirstObjId);
               if (obj1!=0) {
                  obj = obj1;
                  findptr = kTRUE;
                  TString clname;
                  Version_t version;
                  if ((cl!=0) && SqlObjectInfo(objid, clname, version))
                     *cl = TClass::GetClass(clname);
               }
            }
         }
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

   fCurrentData->ShiftToNextValue();

   if ((gDebug>2) || (objid<0))
      cout << "Found object reference " << objid << endl;

   return SqlReadObjectDirect(obj, cl, objid, streamer, streamer_index, onFileClass);
}

//______________________________________________________________________________
void* TBufferSQL2::SqlReadObjectDirect(void* obj, TClass** cl, Long64_t objid, TMemberStreamer *streamer, Int_t streamer_index, const TClass *onFileClass)
{
   // Read object data.
   // Class name and version are taken from special objects table.

   TString clname;
   Version_t version;

   if (!SqlObjectInfo(objid, clname, version)) return obj;

   if (gDebug>2)
      Info("SqlReadObjectDirect","objid = %lld clname = %s ver = %d",objid, clname.Data(), version);

   TSQLClassInfo* sqlinfo = fSQL->FindSQLClassInfo(clname.Data(), version);

   TClass* objClass = TClass::GetClass(clname);
   if (objClass == TDirectory::Class()) objClass = TDirectoryFile::Class();

   if ((objClass==0) || (sqlinfo==0)) {
      Error("SqlReadObjectDirect","Class %s is not known", clname.Data());
      return obj;
   }

   if (obj==0) obj = objClass->New();

   if (fObjMap==0) fObjMap = new TExMap();

   fObjMap->Add((Long_t) objid - fFirstObjId, (Long_t) obj);

   PushStack()->SetObjectRef(objid, objClass);

   TSQLObjectData* olddata = fCurrentData;

   if (sqlinfo->IsClassTableExist()) {
      // TObject and TString classes treated differently
      if ((objClass==TObject::Class()) || (objClass==TString::Class())) {

         TSQLObjectData* objdata = new TSQLObjectData;
         if (objClass==TObject::Class())
            TSQLStructure::UnpackTObject(fSQL, this, objdata, objid, version);
         else
            if (objClass==TString::Class())
               TSQLStructure::UnpackTString(fSQL, this, objdata, objid, version);

         Stack()->AddObjectData(objdata);
         fCurrentData = objdata;
      } else
         // before normal streamer first version will be read and
         // then streamer functions of TStreamerInfo class
         fReadVersionBuffer = version;
   } else {
      TSQLObjectData* objdata = SqlObjectData(objid, sqlinfo);
      if ((objdata==0) || !objdata->PrepareForRawData()) {
         Error("SqlReadObjectDirect","No found raw data for obj %lld in class %s version %d table", objid, clname.Data(), version);
         fErrorFlag = 1;
         return obj;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }

   if (streamer!=0) {
      streamer->SetOnFileClass( onFileClass );
      (*streamer)(*this, (void*)obj, streamer_index);
   } else {
      objClass->Streamer((void*)obj, *this, onFileClass);
   }

   PopStack();

   if (gDebug>1)
      cout << "Read object of class " << objClass->GetName() << " done" << endl << endl;

   if (cl!=0) *cl = objClass;

   fCurrentData = olddata;

   return obj;
}

//______________________________________________________________________________
void  TBufferSQL2::IncrementLevel(TVirtualStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and indent new level in data structure.
   // This call indicates, that TStreamerInfo functions starts streaming
   // object data of correspondent class

   if (info==0) return;

   PushStack()->SetStreamerInfo((TStreamerInfo*)info);

   if (gDebug>2)
      cout << " IncrementLevel " << info->GetName() << endl;

   WorkWithClass(info->GetName(), info->GetClassVersion());
}

//______________________________________________________________________________
void  TBufferSQL2::DecrementLevel(TVirtualStreamerInfo* info)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and decrease level in sql structure.

   TSQLStructure* curr = Stack();
   if (curr->GetElement()) PopStack(); // for element
   PopStack();  // for streamerinfo

   // restore value of object data
   fCurrentData = Stack()->GetObjectData(kTRUE);

   fExpectedChain = kFALSE;

   if (gDebug>2)
      cout << " DecrementLevel " << info->GetClass()->GetName() << endl;
}

//______________________________________________________________________________
void TBufferSQL2::SetStreamerElementNumber(TStreamerElement *elem, Int_t comp_type)
{
   // Function is called from TStreamerInfo WriteBuffer and Readbuffer functions
   // and add/verify next element in sql tables
   // This calls allows separate data, correspondent to one class member, from another

   if (Stack()->GetElement()) PopStack(); // was with if (number > 0), i.e. not first element.
   TSQLStructure* curr = Stack();

   TStreamerInfo* info = curr->GetStreamerInfo();
   if (info==0) {
      Error("SetStreamerElementNumber","Error in structures stack");
      return;
   }

   Int_t elem_type = elem->GetType();

   fExpectedChain = ((elem_type>0) && (elem_type<20)) &&
      (comp_type - elem_type == TStreamerInfo::kOffsetL);

   WorkWithElement(elem, comp_type);
}

//______________________________________________________________________________
void TBufferSQL2::ClassBegin(const TClass* cl, Version_t classversion)
{
   // This method inform buffer data of which class now
   // will be streamed. When reading, classversion should be specified
   // as was read by TBuffer::ReadVersion() call
   //
   // ClassBegin(), ClassEnd() & ClassMemeber() should be used in
   // custom class streamers to specify which kind of data are
   // now streamed to/from buffer. That information is used to correctly
   // convert class data to/from "normal" sql tables with meaningfull names
   // and correct datatypes. Without that functions data from custom streamer
   // will be saved as "raw" data in special _streamer_ table one value after another
   // Such MUST be used when object is written with standard ROOT streaming
   // procedure, but should be read back in custom streamer.
   // For example, custom streamer of TNamed class may look like:

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

   if (classversion<0) classversion = cl->GetClassVersion();

   PushStack()->SetCustomClass(cl, classversion);

   if (gDebug>2) Info("ClassBegin", "%s", cl->GetName());

   WorkWithClass(cl->GetName(), classversion);
}

//______________________________________________________________________________
void TBufferSQL2::ClassEnd(const TClass* cl)
{
   // Method indicates end of streaming of classdata in custom streamer.
   // See ClassBegin() method for more details.

   TSQLStructure* curr = Stack();
   if (curr->GetType()==TSQLStructure::kSqlCustomElement) PopStack(); // for element
   PopStack();  // for streamerinfo

   // restore value of object data
   fCurrentData = Stack()->GetObjectData(kTRUE);

   fExpectedChain = kFALSE;

   if (gDebug>2) Info("ClassEnd","%s",cl->GetName());
}

//______________________________________________________________________________
void TBufferSQL2::ClassMember(const char* name, const char* typeName, Int_t arrsize1, Int_t arrsize2)
{
   // Method indicates name and typename of class memeber,
   // which should be now streamed in custom streamer
   // Following combinations are supported:
   // 1. name = "ClassName", typeName = 0 or typename==ClassName
   //    This is a case, when data of parent class "ClassName" should be streamed.
   //    For instance, if class directly inherited from TObject, custom
   //    streamer should include following code:
   //    b.ClassMember("TObject");
   //    TObject::Streamer(b);
   // 2. Basic data type
   //      b.ClassMember("fInt","Int_t");
   //      b >> fInt;
   // 3. Array of basic data types
   //      b.ClassMember("fArr","Int_t", 5);
   //      b.ReadFastArray(fArr, 5);
   // 4. Object as data member
   //      b.ClassMemeber("fName","TString");
   //      fName.Streamer(b);
   // 5. Pointer on object as datamember
   //      b.ClassMemeber("fObj","TObject*");
   //      b.StreamObject(b);
   // arrsize1 and arrsize2 arguments (when specified) indicate first and
   // second dimension of array. Can be used for array of basic types.
   // For more details see ClassBegin() method description.

   if (typeName==0) typeName = name;

   if ((name==0) || (strlen(name)==0)) {
      Error("ClassMember","Invalid member name");
      fErrorFlag = 1;
      return;
   }

   TString tname = typeName;

   Int_t typ_id = -1;

   if (strcmp(typeName,"raw:data")==0)
      typ_id = TStreamerInfo::kMissing;

   if (typ_id<0) {
      TDataType *dt = gROOT->GetType(typeName);
      if (dt!=0)
         if ((dt->GetType()>0) && (dt->GetType()<20))
            typ_id = dt->GetType();
   }

   if (typ_id<0)
      if (strcmp(name, typeName)==0) {
         TClass* cl = TClass::GetClass(tname.Data());
         if (cl!=0) typ_id = TStreamerInfo::kBase;
      }

   if (typ_id<0) {
      Bool_t isptr = kFALSE;
      if (tname[tname.Length()-1]=='*') {
         tname.Resize(tname.Length()-1);
         isptr = kTRUE;
      }
      TClass* cl = TClass::GetClass(tname.Data());
      if (cl==0) {
         Error("ClassMember","Invalid class specifier %s", typeName);
         fErrorFlag = 1;
         return;
      }

      if (cl->IsTObject())
         typ_id = isptr ? TStreamerInfo::kObjectp : TStreamerInfo::kObject;
      else
         typ_id = isptr ? TStreamerInfo::kAnyp : TStreamerInfo::kAny;

      if ((cl==TString::Class()) && !isptr)
         typ_id = TStreamerInfo::kTString;
   }

   TStreamerElement* elem = 0;

   if (typ_id == TStreamerInfo::kMissing) {
      elem = new TStreamerElement(name,"title",0, typ_id, "raw:data");
   } else

   if (typ_id==TStreamerInfo::kBase) {
      TClass* cl = TClass::GetClass(tname.Data());
      if (cl!=0) {
         TStreamerBase* b = new TStreamerBase(tname.Data(), "title", 0);
         b->SetBaseVersion(cl->GetClassVersion());
         elem = b;
      }
   } else

   if ((typ_id>0) && (typ_id<20)) {
      elem = new TStreamerBasicType(name, "title", 0, typ_id, typeName);
   } else

   if ((typ_id==TStreamerInfo::kObject) ||
       (typ_id==TStreamerInfo::kTObject) ||
       (typ_id==TStreamerInfo::kTNamed)) {
      elem = new TStreamerObject(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kObjectp) {
      elem = new TStreamerObjectPointer(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kAny) {
      elem = new TStreamerObjectAny(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kAnyp) {
      elem = new TStreamerObjectAnyPointer(name, "title", 0, tname.Data());
   } else

   if (typ_id==TStreamerInfo::kTString) {
      elem = new TStreamerString(name, "title", 0);
   }

   if (elem==0) {
      Error("ClassMember","Invalid combination name = %s type = %s", name, typeName);
      fErrorFlag = 1;
      return;
   }

   if (arrsize1>0) {
      elem->SetArrayDim(arrsize2>0 ? 2 : 1);
      elem->SetMaxIndex(0, arrsize1);
      if (arrsize2>0)
         elem->SetMaxIndex(1, arrsize2);
   }

   // return stack to CustomClass node
   if (Stack()->GetType()==TSQLStructure::kSqlCustomElement) PopStack();

   fExpectedChain = kFALSE;

   // we indicate that there is no streamerinfo
   WorkWithElement(elem, -1);
}

//______________________________________________________________________________
void TBufferSQL2::WorkWithClass(const char* classname, Version_t classversion)
{
   // This function is a part of IncrementLevel method.
   // Also used in StartClass method

   fExpectedChain = kFALSE;

   if (IsReading()) {
      Long64_t objid = 0;

//      if ((fCurrentData!=0) && fCurrentData->VerifyDataType(sqlio::ObjectInst, kFALSE))
//        if (!fCurrentData->IsBlobData()) Info("WorkWithClass","Big problem %s", fCurrentData->GetValue());

      if ((fCurrentData!=0) && fCurrentData->IsBlobData() &&
          fCurrentData->VerifyDataType(sqlio::ObjectInst, kFALSE)) {
         objid = atoi(fCurrentData->GetValue());
         fCurrentData->ShiftToNextValue();
         TString sobjid;
         sobjid.Form("%lld",objid);
         Stack()->ChangeValueOnly(sobjid.Data());
      } else
         objid = Stack()->DefineObjectId(kTRUE);
      if (objid<0) {
         Error("WorkWithClass","cannot define object id");
         fErrorFlag = 1;
         return;
      }

      TSQLClassInfo* sqlinfo = fSQL->FindSQLClassInfo(classname, classversion);
      if (sqlinfo==0) {
         Error("WorkWithClass","Can not find table for class %s version %d", classname, classversion);
         fErrorFlag = 1;
         return;
      }

      TSQLObjectData* objdata = SqlObjectData(objid, sqlinfo);
      if (objdata==0) {
         Error("WorkWithClass","Request error for data of object %lld for class %s version %d", objid, classname, classversion);
         fErrorFlag = 1;
         return;
      }

      Stack()->AddObjectData(objdata);

      fCurrentData = objdata;
   }
}

//______________________________________________________________________________
void TBufferSQL2::WorkWithElement(TStreamerElement* elem, Int_t /* comp_type */)
{
   // This function is a part of SetStreamerElementNumber method.
   // It is introduced for reading of data for specified data memeber of class.
   // Used also in ReadFastArray methods to resolve problem of compressed data,
   // when several data memebers of the same basic type streamed with single ...FastArray call

   if (gDebug>2)
      Info("WorkWithElement","elem = %s",elem->GetName());

   TSQLStructure* stack = Stack(1);
   TStreamerInfo* info = stack->GetStreamerInfo();
   Int_t number = info ? info->GetElements()->IndexOf(elem) : -1;

   if (number>=0)
      PushStack()->SetStreamerElement(elem, number);
   else
      PushStack()->SetCustomElement(elem);

   if (IsReading()) {

      if (fCurrentData==0) {
         Error("WorkWithElement","Object data is lost");
         fErrorFlag = 1;
         return;
      }

      fCurrentData = Stack()->GetObjectData(kTRUE);

      Int_t located = Stack()->LocateElementColumn(fSQL, this, fCurrentData);

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
void TBufferSQL2::SkipVersion(const TClass *cl)
{
   // Skip class version from I/O buffer.
   ReadVersion(0,0,cl);
}

//______________________________________________________________________________
Version_t TBufferSQL2::ReadVersion(UInt_t *start, UInt_t *bcnt, const TClass *)
{
   // read version value from buffer
   // actually version is normally defined by table name
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
         cout << "TBufferSQL2::ReadVersion from blob " << fCurrentData->GetBlobPrefixName() << " = " << res << endl;
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
      cout << "TBufferSQL2::WriteVersion " << (cl ? cl->GetName() : "null") << "   ver = " << (cl ? cl->GetClassVersion() : 0) << endl;

   if (cl)
      Stack()->AddVersion(cl);

   return 0;
}

//______________________________________________________________________________
void* TBufferSQL2::ReadObjectAny(const TClass*)
{
   // Read object from buffer. Only used from TBuffer.

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
void TBufferSQL2::WriteObjectClass(const void *actualObjStart, const TClass *actualClass)
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
         const char* name = fCurrentData->GetBlobPrefixName();          \
         Int_t first, last, res;                                        \
         if (strstr(name,sqlio::IndexSepar)==0) {                       \
            res = sscanf(name,"[%d", &first); last = first;             \
         } else res = sscanf(name,"[%d..%d", &first, &last);            \
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
void TBufferSQL2::ReadFloat16 (Float_t *f, TStreamerElement * /*ele*/)
{
   // Read Float16 value

   SqlReadBasic(*f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // Read Double32 value

   SqlReadBasic(*d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadWithFactor(Float_t *ptr, Double_t /* factor */, Double_t /* minvalue */)
{
   // Read a Double32_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   SqlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferSQL2::ReadWithNbits(Float_t *ptr, Int_t /* nbits */)
{
   // Read a Float16_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Float16_t encoding at TBufferFile::WriteFloat16().
   // Currently TBufferXML does not optimize space in this case.

   SqlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferSQL2::ReadWithFactor(Double_t *ptr, Double_t /* factor */, Double_t /* minvalue */)
{
   // Read a Double32_t from the buffer when the factor and minimun value have been specified
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   SqlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferSQL2::ReadWithNbits(Double_t *ptr, Int_t /* nbits */)
{
   // Read a Double32_t from the buffer when the number of bits is specified (explicitly or not)
   // see comments about Double32_t encoding at TBufferFile::WriteDouble32().
   // Currently TBufferXML does not optimize space in this case.

   SqlReadBasic(*ptr);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFloat16 (Float_t *f, TStreamerElement * /*ele*/)
{
   // Write Float16 value

   SqlWriteBasic(*f);
}

//______________________________________________________________________________
void TBufferSQL2::WriteDouble32 (Double_t *d, TStreamerElement * /*ele*/)
{
   // Write Double32 value

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
Int_t TBufferSQL2::ReadArrayFloat16(Float_t  *&f, TStreamerElement * /*ele*/)
{
   // Read array of Float16_t from buffer

   TBufferSQL2_ReadArray(Float_t,f);
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
Int_t TBufferSQL2::ReadStaticArrayFloat16(Float_t  *f, TStreamerElement * /*ele*/)
{
   // Read array of Float16_t from buffer

   TBufferSQL2_ReadStaticArray(f);
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
         Int_t index = 0;                                               \
         while (index<n) {                                              \
            elem = (TStreamerElement*)info->GetElements()->At(startnumber++); \
            if (index>1) { PopStack(); WorkWithElement(elem, elem->GetType()); } \
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
void TBufferSQL2::ReadFastArrayFloat16(Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // read array of Float16_t from buffer

   TBufferSQL2_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayWithFactor(Float_t  *f, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Float16_t from buffer

   TBufferSQL2_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayWithNbits(Float_t  *f, Int_t n, Int_t /*nbits*/)
{
   // read array of Float16_t from buffer

   TBufferSQL2_ReadFastArray(f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayDouble32(Double_t  *d, Int_t n, TStreamerElement * /*ele*/)
{
   // read array of Double32_t from buffer

   TBufferSQL2_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayWithFactor(Double_t  *d, Int_t n, Double_t /* factor */, Double_t /* minvalue */)
{
   // read array of Double32_t from buffer

   TBufferSQL2_ReadFastArray(d);
}
//______________________________________________________________________________
void TBufferSQL2::ReadFastArrayWithNbits(Double_t  *d, Int_t n, Int_t /*nbits*/)
{
   // read array of Double32_t from buffer

   TBufferSQL2_ReadFastArray(d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(void  *start, const TClass *cl, Int_t n, TMemberStreamer *streamer, const TClass* onFileClass )
{
   // Same functionality as TBuffer::ReadFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished

   if (gDebug>2)
      Info("ReadFastArray","(void *");

   if (streamer) {
      StreamObject(start, streamer, cl, 0, onFileClass);
//      (*streamer)(*this,start,0);
      return;
   }

   int objectSize = cl->Size();
   char *obj = (char*)start;
   char *end = obj + n*objectSize;

   for(; obj<end; obj+=objectSize) {
      StreamObject(obj, cl, onFileClass);
   }
   //   TBuffer::ReadFastArray(start, cl, n, s);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFastArray(void **start, const TClass *cl, Int_t n, Bool_t isPreAlloc, TMemberStreamer *streamer, const TClass* onFileClass )
{
   // Same functionality as TBuffer::ReadFastArray(...) but
   // instead of calling cl->Streamer(obj,buf) call here
   // buf.StreamObject(obj, cl). In that case it is easy to understand where
   // object data is started and finished

   if (gDebug>2)
      Info("ReadFastArray","(void **  pre = %d  n = %d", isPreAlloc, n);

   if (streamer) {
      if (isPreAlloc) {
         for (Int_t j=0;j<n;j++) {
            if (!start[j]) start[j] = ((TClass*)cl)->New();
         }
      }
      StreamObject((void*)start, streamer, cl, 0, onFileClass);
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
         StreamObject(start[j], cl, onFileClass);
      }
   }

   if (gDebug>2)
      Info("ReadFastArray","(void ** Done" );

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
void TBufferSQL2::WriteArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Float16_t to buffer

   TBufferSQL2_WriteArray(f);
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
         Int_t index = 0;                                               \
         while (index<n) {                                              \
            elem = (TStreamerElement*)info->GetElements()->At(startnumber++); \
            if (index>0) { PopStack(); WorkWithElement(elem, elem->GetType()); } \
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
void TBufferSQL2::WriteFastArrayFloat16(const Float_t  *f, Int_t n, TStreamerElement * /*ele*/)
{
   // Write array of Float16_t to buffer

   TBufferSQL2_WriteFastArray(f);
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
         if (!strInfo && !start[j] ) ForceWriteInfo(((TClass*)cl)->GetStreamerInfo(),kFALSE);
         strInfo = 2003;
         res |= WriteObjectAny(start[j],cl);
      }

   } else {
      //case //-> in comment

      for (Int_t j=0;j<n;j++) {
         if (!start[j]) start[j] = ((TClass*)cl)->New();
         StreamObject(start[j], cl);
      }

   }
   return res;

//   return TBuffer::WriteFastArray(startp, cl, n, isPreAlloc, s);
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const type_info &typeinfo, const TClass *onFileClass)
{
   // steram object to/from buffer

   StreamObject(obj, TClass::GetClass(typeinfo), onFileClass);
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const char *className, const TClass *onFileClass)
{
   // steram object to/from buffer

   StreamObject(obj, TClass::GetClass(className), onFileClass);
}

//______________________________________________________________________________
void TBufferSQL2::StreamObject(void *obj, const TClass *cl, const TClass *onFileClass)
{
   // steram object to/from buffer

   if (gDebug>1)
      cout << " TBufferSQL2::StreamObject class = " << (cl ? cl->GetName() : "none") << endl;
   if (IsReading())
      SqlReadObject(obj, 0, 0, 0, onFileClass);
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
void TBufferSQL2::StreamObject(void *obj, TMemberStreamer *streamer, const TClass *cl, Int_t n, const TClass *onFileClass)
{
   // steram object to/from buffer

   if (streamer==0) return;

   if (gDebug>1)
      cout << "Stream object of class = " << cl->GetName() << endl;
//   (*streamer)(*this, obj, n);

   if (IsReading())
      SqlReadObject(obj, 0, streamer, n, onFileClass);
   else
      SqlWriteObject(obj, cl, streamer, n);
}

// macro for right shift operator for basic type
#define TBufferSQL2_operatorin(vname)           \
   {                                            \
      SqlReadBasic(vname);                      \
   }

//______________________________________________________________________________
void TBufferSQL2::ReadBool(Bool_t    &b)
{
   // Reads Bool_t value from buffer

   TBufferSQL2_operatorin(b);
}

//______________________________________________________________________________
void TBufferSQL2::ReadChar(Char_t    &c)
{
   // Reads Char_t value from buffer

   TBufferSQL2_operatorin(c);
}

//______________________________________________________________________________
void TBufferSQL2::ReadUChar(UChar_t   &c)
{
   // Reads UChar_t value from buffer

   TBufferSQL2_operatorin(c);
}

//______________________________________________________________________________
void TBufferSQL2::ReadShort(Short_t   &h)
{
   // Reads Short_t value from buffer

   TBufferSQL2_operatorin(h);
}

//______________________________________________________________________________
void TBufferSQL2::ReadUShort(UShort_t  &h)
{
   // Reads UShort_t value from buffer

   TBufferSQL2_operatorin(h);
}

//______________________________________________________________________________
void TBufferSQL2::ReadInt(Int_t     &i)
{
   // Reads Int_t value from buffer

   TBufferSQL2_operatorin(i);
}

//______________________________________________________________________________
void TBufferSQL2::ReadUInt(UInt_t    &i)
{
   // Reads UInt_t value from buffer

   TBufferSQL2_operatorin(i);
}

//______________________________________________________________________________
void TBufferSQL2::ReadLong(Long_t    &l)
{
   // Reads Long_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadULong(ULong_t   &l)
{
   // Reads ULong_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadLong64(Long64_t  &l)
{
   // Reads Long64_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadULong64(ULong64_t &l)
{
   // Reads ULong64_t value from buffer

   TBufferSQL2_operatorin(l);
}

//______________________________________________________________________________
void TBufferSQL2::ReadFloat(Float_t   &f)
{
   // Reads Float_t value from buffer

   TBufferSQL2_operatorin(f);
}

//______________________________________________________________________________
void TBufferSQL2::ReadDouble(Double_t  &d)
{
   // Reads Double_t value from buffer

   TBufferSQL2_operatorin(d);
}

//______________________________________________________________________________
void TBufferSQL2::ReadCharP(Char_t    *c)
{
   // Reads array of characters from buffer

   const char* buf = SqlReadCharStarValue();
   if (buf) strcpy(c, buf);
}

//________________________________________________________________________
void TBufferSQL2::ReadTString(TString   &s)
{
   // Read a TString

   TBufferFile::ReadTString(s);
}

//________________________________________________________________________
void TBufferSQL2::WriteTString(const TString  &s)
{
   // Write a TString

   TBufferFile::WriteTString(s);
}

//________________________________________________________________________
void TBufferSQL2::ReadStdString(std::string *s)
{
   // Read a std::string

   TBufferFile::ReadStdString(s);
}

//________________________________________________________________________
void TBufferSQL2::WriteStdString(const std::string *s)
{
   // Write a std::string

   TBufferFile::WriteStdString(s);
}

//______________________________________________________________________________
void TBufferSQL2::ReadCharStar(char* &s)
{
   // Read a char* string

   TBufferFile::ReadCharStar(s);
}

//______________________________________________________________________________
void TBufferSQL2::WriteCharStar(char *s)
{
   // Write a char* string

   TBufferFile::WriteCharStar(s);
}

// macro for right shift operator for basic types
#define TBufferSQL2_operatorout(vname)          \
   {                                            \
      SqlWriteBasic(vname);                     \
   }

//______________________________________________________________________________
void TBufferSQL2::WriteBool(Bool_t    b)
{
   // Writes Bool_t value to buffer

   TBufferSQL2_operatorout(b);
}

//______________________________________________________________________________
void TBufferSQL2::WriteChar(Char_t    c)
{
   // Writes Char_t value to buffer

   TBufferSQL2_operatorout(c);
}

//______________________________________________________________________________
void TBufferSQL2::WriteUChar(UChar_t   c)
{
   // Writes UChar_t value to buffer

   TBufferSQL2_operatorout(c);
}

//______________________________________________________________________________
void TBufferSQL2::WriteShort(Short_t   h)
{
   // Writes Short_t value to buffer

   TBufferSQL2_operatorout(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteUShort(UShort_t  h)
{
   // Writes UShort_t value to buffer

   TBufferSQL2_operatorout(h);
}

//______________________________________________________________________________
void TBufferSQL2::WriteInt(Int_t     i)
{
   // Writes Int_t value to buffer

   TBufferSQL2_operatorout(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteUInt(UInt_t    i)
{
   // Writes UInt_t value to buffer

   TBufferSQL2_operatorout(i);
}

//______________________________________________________________________________
void TBufferSQL2::WriteLong(Long_t    l)
{
   // Writes Long_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteULong(ULong_t   l)
{
   // Writes ULong_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteLong64(Long64_t  l)
{
   // Writes Long64_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteULong64(ULong64_t l)
{
   // Writes ULong64_t value to buffer

   TBufferSQL2_operatorout(l);
}

//______________________________________________________________________________
void TBufferSQL2::WriteFloat(Float_t   f)
{
   // Writes Float_t value to buffer

   TBufferSQL2_operatorout(f);
}

//______________________________________________________________________________
void TBufferSQL2::WriteDouble(Double_t  d)
{
   // Writes Double_t value to buffer

   TBufferSQL2_operatorout(d);
}

//______________________________________________________________________________
void TBufferSQL2::WriteCharP(const Char_t *c)
{
   // Writes array of characters to buffer

   SqlWriteValue(c, sqlio::CharStar);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Char_t value)
{
   // converts Char_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   return SqlWriteValue(buf, sqlio::Char);
}

//______________________________________________________________________________
Bool_t  TBufferSQL2::SqlWriteBasic(Short_t value)
{
   // converts Short_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%hd", value);
   return SqlWriteValue(buf, sqlio::Short);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Int_t value)
{
   // converts Int_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%d", value);
   return SqlWriteValue(buf, sqlio::Int);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Long_t value)
{
   // converts Long_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%ld", value);
   return SqlWriteValue(buf, sqlio::Long);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Long64_t value)
{
   // converts Long64_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%lld", value);
   return SqlWriteValue(buf, sqlio::Long64);
}

//______________________________________________________________________________
Bool_t  TBufferSQL2::SqlWriteBasic(Float_t value)
{
   // converts Float_t to string and creates correspondent sql structure

   char buf[200];
   snprintf(buf, sizeof(buf), TSQLServer::GetFloatFormat(), value);
   return SqlWriteValue(buf, sqlio::Float);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(Double_t value)
{
   // converts Double_t to string and creates correspondent sql structure

   char buf[128];
   snprintf(buf, sizeof(buf), TSQLServer::GetFloatFormat(), value);
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
   snprintf(buf, sizeof(buf), "%u", value);
   return SqlWriteValue(buf, sqlio::UChar);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(UShort_t value)
{
   // converts UShort_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%hu", value);
   return SqlWriteValue(buf, sqlio::UShort);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(UInt_t value)
{
   // converts UInt_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%u", value);
   return SqlWriteValue(buf, sqlio::UInt);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(ULong_t value)
{
   // converts ULong_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), "%lu", value);
   return SqlWriteValue(buf, sqlio::ULong);
}

//______________________________________________________________________________
Bool_t TBufferSQL2::SqlWriteBasic(ULong64_t value)
{
   // converts ULong64_t to string and creates correspondent sql structure

   char buf[50];
   snprintf(buf, sizeof(buf), FULong64, value);
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
      sscanf(res, FLong64, &value);
   else
      value = 0;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Float_t& value)
{
   // read current value from table and convert it to Float_t value

   const char* res = SqlReadValue(sqlio::Float);
   if (res)
      sscanf(res, "%f", &value);
   else
      value = 0.;
}

//______________________________________________________________________________
void TBufferSQL2::SqlReadBasic(Double_t& value)
{
   // read current value from table and convert it to Double_t value

   const char* res = SqlReadValue(sqlio::Double);
   if (res)
      sscanf(res, "%lf", &value);
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
      sscanf(res, FULong64, &value);
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

   Long64_t objid = Stack()->DefineObjectId(kTRUE);

   Int_t strid = fSQL->IsLongStringCode(objid, res);
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


//______________________________________________________________________________
void TBufferSQL2::SetFloatFormat(const char* fmt)
{
   // set printf format for float/double members, default "%e"
   // changes global TSQLServer variable

   TSQLServer::SetFloatFormat(fmt);
}

//______________________________________________________________________________
const char* TBufferSQL2::GetFloatFormat()
{
   // return current printf format for float/double members, default "%e"
   // return format, hold by TSQLServer

   return TSQLServer::GetFloatFormat();
}

//______________________________________________________________________________
Int_t TBufferSQL2::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *obj)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,obj);
         (*iter)(*this,obj);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,obj);
      }
   }

   DecrementLevel(info);
   return 0;
}

//______________________________________________________________________________
Int_t TBufferSQL2::ApplySequenceVecPtr(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.
   // The collection needs to be a split TClonesArray or a split vector of pointers.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   if (gDebug) {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,*(char**)start_collection);  // Warning: This limits us to TClonesArray and vector of pointers.
         (*iter)(*this,start_collection,end_collection);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,start_collection,end_collection);
      }
   }

   DecrementLevel(info);
   return 0;
}

//______________________________________________________________________________
Int_t TBufferSQL2::ApplySequence(const TStreamerInfoActions::TActionSequence &sequence, void *start_collection, void *end_collection)
{
   // Read one collection of objects from the buffer using the StreamerInfoLoopAction.

   TVirtualStreamerInfo *info = sequence.fStreamerInfo;
   IncrementLevel(info);

   TStreamerInfoActions::TLoopConfiguration *loopconfig = sequence.fLoopConfig;
   if (gDebug) {

      // Get the address of the first item for the PrintDebug.
      // (Performance is not essential here since we are going to print to
      // the screen anyway).
      void *arr0 = loopconfig->GetFirstAddress(start_collection,end_collection);
      // loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter).PrintDebug(*this,arr0);
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }

   } else {
      //loop on all active members
      TStreamerInfoActions::ActionContainer_t::const_iterator end = sequence.fActions.end();
      for(TStreamerInfoActions::ActionContainer_t::const_iterator iter = sequence.fActions.begin();
          iter != end;
          ++iter) {
         // Idea: Try to remove this function call as it is really needed only for XML streaming.
         SetStreamerElementNumber((*iter).fConfiguration->fCompInfo->fElem,(*iter).fConfiguration->fCompInfo->fType);
         (*iter)(*this,start_collection,end_collection,loopconfig);
      }
   }

   DecrementLevel(info);
   return 0;
}

