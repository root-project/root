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
//  This is hierarhical structure, which is created when data is written
//  by TBufferSQL2. It contains data all structurual information such:
//  version of written class, data memeber types of that class, value for
//  each data memeber and so on.
//  Such structure in some sense similar to XML node and subnodes structure
//  Once it created, it converted to SQL statements, which are submitted
//  to database server.
//
//________________________________________________________________________

#include "TSQLStructure.h"

#include "Riostream.h"
#include "TMap.h"
#include "TClass.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TObjString.h"
#include "TClonesArray.h"

#include "TSQLFile.h"
#include "TSQLClassInfo.h"
#include "TSQLObjectData.h"
#include "TBufferSQL2.h"

#include "TSQLStatement.h"
#include "TSQLServer.h"
#include "TDataType.h"

namespace sqlio {
   const Int_t Ids_NullPtr       = 0; // used to identify NULL pointer in tables
   const Int_t Ids_RootDir       = 0; // dir:id, used for keys stored in root directory.
   const Int_t Ids_TSQLFile      = 0; // keyid for TSQLFile entry in keys list
   const Int_t Ids_StreamerInfos = 1; // keyid used to store StreamerInfos in ROOT directory
   const Int_t Ids_FirstKey      =10; // first key id, which is used in KeysTable (beside streamer info or something else)
   const Int_t Ids_FirstObject   = 1; // first object id, allowed in object tables

   const char* ObjectRef     = "ObjectRef";
   const char* ObjectRef_Arr = "ObjectRefArr";
   const char* ObjectPtr     = "ObjectPtr";
   const char* ObjectInst    = "ObjectInst";
   const char* Version       = "Version";
   const char* TObjectUniqueId = "UniqueId";
   const char* TObjectBits = "Bits";
   const char* TObjectProcessId = "ProcessId";
   const char* TStringValue= "StringValue";
   const char* IndexSepar  = "..";
   const char* RawSuffix = ":rawdata";
   const char* ParentSuffix = ":parent";
   const char* ObjectSuffix = ":object";
   const char* PointerSuffix = ":pointer";
   const char* StrSuffix     = ":str";
   const char* LongStrPrefix = "#~#";

   const char* Array       = "Array";
   const char* Bool        = "Bool_t";
   const char* Char        = "Char_t";
   const char* Short       = "Short_t";
   const char* Int         = "Int_t";
   const char* Long        = "Long_t";
   const char* Long64      = "Long64_t";
   const char* Float       = "Float_t";
   const char* Double      = "Double_t";
   const char* UChar       = "UChar_t";
   const char* UShort      = "UShort_t";
   const char* UInt        = "UInt_t";
   const char* ULong       = "ULong_t";
   const char* ULong64     = "ULong64_t";
   const char* CharStar    = "CharStar";
   const char* True        = "1";
   const char* False       = "0";

   // standard tables names
   const char* KeysTable      = "KeysTable";
   const char* KeysTableIndex = "KeysTableIndex";
   const char* ObjectsTable   = "ObjectsTable";
   const char* ObjectsTableIndex = "ObjectsTableIndex";
   const char* IdsTable = "IdsTable";
   const char* IdsTableIndex = "IdsTableIndex";
   const char* StringsTable   = "StringsTable";
   const char* ConfigTable    = "Configurations";

   // colummns in Keys table
   const char* KT_Name      = "Name";
   const char* KT_Title     = "Title";
   const char* KT_Datetime  = "Datime";
   const char* KT_Cycle     = "Cycle";
   const char* KT_Class     = "Class";

   const char* DT_Create    = "CreateDatime";
   const char* DT_Modified  = "ModifiedDatime";
   const char* DT_UUID      = "UUID";

   // colummns in Objects table
   const char* OT_Class     = "Class";
   const char* OT_Version   = "Version";

   // columns in Identifiers Table
   const char* IT_TableID   = "TableId";
   const char* IT_SubID     = "SubId";
   const char* IT_Type      = "Type";
   const char* IT_FullName  = "FullName";
   const char* IT_SQLName   = "SQLName";
   const char* IT_Info      = "Info";

   // colummns in _streamer_ tables
   const char* BT_Field     = "Field";
   const char* BT_Value     = "Value";

   // colummns in string table
   const char* ST_Value     = "LongStringValue";

   // columns in config table
   const char* CT_Field     = "Field";
   const char* CT_Value     = "Value";

   // values in config table
   const char* cfg_Version  = "SQL_IO_version";
   const char* cfg_UseSufixes = "UseNameSuffix";
   const char* cfg_ArrayLimit = "ArraySizeLimit";
   const char* cfg_TablesType = "TablesType";
   const char* cfg_UseTransactions = "UseTransactions";
   const char* cfg_UseIndexes = "UseIndexes";
   const char* cfg_LockingMode = "LockingMode";
   const char* cfg_ModifyCounter = "ModifyCounter";
};

//________________________________________________________________________

#ifdef R__VISUAL_CPLUSPLUS
#define FLong64    "%I64d"
#define FULong64   "%I64u"
#else
#define FLong64    "%lld"
#define FULong64   "%llu"
#endif

Long64_t sqlio::atol64(const char* value)
{
   if ((value==0) || (*value==0)) return 0;
   Long64_t res = 0;
   sscanf(value, FLong64, &res);
   return res;
}


ClassImp(TSQLColumnData)

//______________________________________________________________________________
   TSQLColumnData::TSQLColumnData() :
      TObject(),
      fName(),
      fType(),
      fValue(),
      fNumeric(kFALSE)
{
   // default constructor
}

//________________________________________________________________________
TSQLColumnData::TSQLColumnData(const char* name,
                               const char* sqltype,
                               const char* value,
                               Bool_t numeric) :
   TObject(),
   fName(name),
   fType(sqltype),
   fValue(value),
   fNumeric(numeric)
{
   // normal constructor of TSQLColumnData class
   // specifies name, type and value for one column
}

//________________________________________________________________________
TSQLColumnData::TSQLColumnData(const char* name, Long64_t value) :
   TObject(),
   fName(name),
   fType("INT"),
   fValue(),
   fNumeric(kTRUE)
{
   // constructs TSQLColumnData object for integer column

   fValue.Form("%lld",value);
}

//________________________________________________________________________
TSQLColumnData::~TSQLColumnData()
{
   // TSQLColumnData destructor
}


ClassImp(TSQLTableData);

//________________________________________________________________________
TSQLTableData::TSQLTableData(TSQLFile* f, TSQLClassInfo* info) : 
   TObject(),
   fFile(f),
   fInfo(info),
   fColumns(),
   fColInfos(0)
{
   // normal constructor
   
   if (info && !info->IsClassTableExist()) 
      fColInfos = new TObjArray;
}

//________________________________________________________________________
TSQLTableData::~TSQLTableData()
{
   // destructor
   
   fColumns.Delete();
   if (fColInfos!=0) {
      fColInfos->Delete();
      delete fColInfos;  
   }
}

//________________________________________________________________________
void TSQLTableData::AddColumn(const char* name, Long64_t value)
{
   // Add INT column to list of columns

   TObjString* v = new TObjString(Form("%lld",value));
   v->SetBit(BIT(20), kTRUE);
   fColumns.Add(v);

//   TSQLColumnData* col = new TSQLColumnData(name, value);
//   fColumns.Add(col);
   
   if (fColInfos!=0)
     fColInfos->Add(new TSQLClassColumnInfo(name, DefineSQLName(name), "INT"));
}

//________________________________________________________________________
void TSQLTableData::AddColumn(const char* name, 
                              const char* sqltype, 
                              const char* value, 
                              Bool_t numeric)
{
   // Add nomral column to list of columns

   TObjString* v = new TObjString(value);
   v->SetBit(BIT(20), numeric);
   fColumns.Add(v);
   
//   TSQLColumnData* col = new TSQLColumnData(name, sqltype, value, numeric);
//   fColumns.Add(col);

   if (fColInfos!=0)
     fColInfos->Add(new TSQLClassColumnInfo(name, DefineSQLName(name), sqltype));
}

//________________________________________________________________________
TString TSQLTableData::DefineSQLName(const char* fullname)
{
   // produce suitable name for column, taking into account length limitation
   
   Int_t maxlen = fFile->SQLMaxIdentifierLength();
   
   Int_t len = strlen(fullname);
   
   if ((len<=maxlen) && !HasSQLName(fullname)) return TString(fullname);
   
   Int_t cnt = -1;
   TString res, scnt;
   
   do {
      
      scnt.Form("%d",cnt);
      Int_t numlen = cnt<0 ? 0 : scnt.Length();
      
      res = fullname;
      
      if (len + numlen > maxlen) 
         res.Resize(maxlen - numlen);
      
      if (cnt>=0) res+=scnt;
      
      if (!HasSQLName(res.Data())) return res;
      
      cnt++;
      
   } while (cnt<10000);
   
   Error("DefineSQLName","Cannot find reasonable column name for field %s",fullname);
   
   return TString(fullname);
}

//________________________________________________________________________
Bool_t TSQLTableData::HasSQLName(const char* sqlname)
{
   // checks if columns list already has that sql name
   
   TIter next(fColInfos);

   TSQLClassColumnInfo* col = 0;
   
   while ((col = (TSQLClassColumnInfo*) next()) != 0) {
      const char* colname = col->GetSQLName();
      if (strcmp(colname, sqlname)==0) return kTRUE;
   }
   
   return kFALSE;

}

//________________________________________________________________________
Int_t TSQLTableData::GetNumColumns()
{
   // returns number of columns in provided set
   
   return fColumns.GetLast() +1;
}

//________________________________________________________________________
const char* TSQLTableData::GetColumn(Int_t n)
{
   // returm column value
   return fColumns[n]->GetName();
}

//________________________________________________________________________
Bool_t TSQLTableData::IsNumeric(Int_t n)
{
   // identifies if column has numeric value
   
   return fColumns[n]->TestBit(BIT(20));
}

//________________________________________________________________________
TObjArray* TSQLTableData::TakeColInfos() 
{ 
   // take ownership over colinfos
   
   TObjArray* res = fColInfos; 
   fColInfos = 0; 
   return res; 
}

//________________________________________________________________________

ClassImp(TSQLStructure);

TSQLStructure::TSQLStructure() :
   TObject(),
   fParent(0),
   fType(0),
   fPointer(0),
   fValue(),
   fArrayIndex(-1),
   fRepeatCnt(0),
   fChilds()
{
   // default constructor
}

//________________________________________________________________________
TSQLStructure::~TSQLStructure()
{
   // destructor

   fChilds.Delete();
   if (GetType()==kSqlObjectData) {
      TSQLObjectData* objdata = (TSQLObjectData*) fPointer;
      delete objdata;
   } else
   if (GetType()==kSqlCustomElement) {
      TStreamerElement* elem = (TStreamerElement*) fPointer;
      delete elem;
   }
}

//________________________________________________________________________
Int_t TSQLStructure::NumChilds() const
{
   // number of child structures

   return fChilds.GetLast()+1;
}

//________________________________________________________________________
TSQLStructure* TSQLStructure::GetChild(Int_t n) const
{
   // return child structure of index n

   return (n<0) || (n>fChilds.GetLast()) ? 0 : (TSQLStructure*) fChilds[n];
}

//________________________________________________________________________
void TSQLStructure::SetObjectRef(Long64_t refid, const TClass* cl)
{
   // set structure type as kSqlObject

   fType = kSqlObject;
   fValue.Form("%lld",refid);
   fPointer = cl;
}

//________________________________________________________________________
void TSQLStructure::SetObjectPointer(Long64_t ptrid)
{
   // set structure type as kSqlPointer

   fType = kSqlPointer;
   fValue.Form("%lld",ptrid);
}

//________________________________________________________________________
void TSQLStructure::SetVersion(const TClass* cl, Int_t version)
{
   // set structure type as kSqlVersion

   fType = kSqlVersion;
   fPointer = cl;
   if (version<0) version = cl->GetClassVersion();
   fValue.Form("%d",version);
}

//________________________________________________________________________
void TSQLStructure::SetClassStreamer(const TClass* cl)
{
   // set structure type as kSqlClassStreamer

   fType = kSqlClassStreamer;
   fPointer = cl;
}

//________________________________________________________________________
void TSQLStructure::SetStreamerInfo(const TStreamerInfo* info)
{
   // set structure type as kSqlStreamerInfo

   fType = kSqlStreamerInfo;
   fPointer = info;
}

//________________________________________________________________________
void TSQLStructure::SetStreamerElement(const TStreamerElement* elem, Int_t number)
{
   // set structure type as kSqlElement

   fType = kSqlElement;
   fPointer = elem;
   fArrayIndex = number;
}

//________________________________________________________________________
void TSQLStructure::SetCustomClass(const TClass* cl, Version_t version)
{
   // set structure type as kSqlCustomClass
   
   fType = kSqlCustomClass;
   fPointer = (void*) cl;
   fArrayIndex = version;
}

//________________________________________________________________________
void TSQLStructure::SetCustomElement(TStreamerElement* elem)
{
   // set structure type as kSqlCustomElement
   
   fType = kSqlCustomElement;
   fPointer = elem;
}

//________________________________________________________________________
void TSQLStructure::SetValue(const char* value, const char* tname)
{
   // set structure type as kSqlValue

   fType = kSqlValue;
   fValue = value;
   fPointer = tname;
}

//________________________________________________________________________
void TSQLStructure::ChangeValueOnly(const char* value)
{
   // change value of this structure
   // used as "workaround" to keep object id in kSqlElement node

   fValue = value;
}

//________________________________________________________________________
void TSQLStructure::SetArrayIndex(Int_t indx, Int_t cnt)
{
   // set array index for this structure

   fArrayIndex = indx;
   fRepeatCnt = cnt;
}

//________________________________________________________________________
void TSQLStructure::ChildArrayIndex(Int_t index, Int_t cnt)
{
   // set array index for last child element
   //   if (cnt<=1) return;
   TSQLStructure* last = (TSQLStructure*) fChilds.Last();
   if ((last!=0) && (last->GetType()==kSqlValue))
      last->SetArrayIndex(index, cnt);
}

//________________________________________________________________________
void TSQLStructure::SetArray(Int_t sz)
{
   // Set structure as array element

   fType = kSqlArray;
   if (sz>=0) fValue.Form("%d",sz);
}

//________________________________________________________________________
TClass* TSQLStructure::GetObjectClass() const
{
   // return object class if type kSqlObject

   return (fType==kSqlObject) ? (TClass*) fPointer : 0;
}

//________________________________________________________________________
TClass* TSQLStructure::GetVersionClass() const
{
   // return class for version tag if type is kSqlVersion

   return (fType==kSqlVersion) ? (TClass*) fPointer : 0;
}

//________________________________________________________________________
TStreamerInfo*  TSQLStructure::GetStreamerInfo() const
{
   // return TStreamerInfo* if type is kSqlStreamerInfo

   return (fType==kSqlStreamerInfo) ? (TStreamerInfo*) fPointer : 0;
}

//________________________________________________________________________
TStreamerElement* TSQLStructure::GetElement() const
{
   // return TStremerElement* if type is kSqlElement

   return (fType==kSqlElement) || (fType==kSqlCustomElement) ? (TStreamerElement*) fPointer : 0;
}

//________________________________________________________________________
Int_t TSQLStructure::GetElementNumber() const
{
   // returns number of TStremerElement in TStreamerInfo

   return (fType==kSqlElement) ? fArrayIndex : 0;
}

//________________________________________________________________________
const char* TSQLStructure::GetValueType() const
{
   // return value type if structure is kSqlValue

   return (fType==kSqlValue) ? (const char*) fPointer : 0;
}

//________________________________________________________________________
TClass* TSQLStructure::GetCustomClass() const
{
   // return element custom class if strutures is kSqlCustomClass
   
   return (fType==kSqlCustomClass) ? (TClass*) fPointer : 0;
}

//________________________________________________________________________
Version_t TSQLStructure::GetCustomClassVersion() const
{
   // return custom class version if strutures is kSqlCustomClass
   
   return (fType==kSqlCustomClass) ? fArrayIndex : 0;
}

//________________________________________________________________________
Bool_t TSQLStructure::GetClassInfo(TClass* &cl, Version_t &version)
{
   // provides class info if structure kSqlStreamerInfo or kSqlCustomClass
   
   if (GetType()==kSqlStreamerInfo) {
      TStreamerInfo* info = GetStreamerInfo();
      if (info==0) return kFALSE;
      cl = info->GetClass();
      version = info->GetClassVersion();
   } else
   if (GetType()==kSqlCustomClass) {
      cl = GetCustomClass();
      version = GetCustomClassVersion();
   } else
      return kFALSE;
   return kTRUE;
}

//________________________________________________________________________
const char* TSQLStructure::GetValue() const
{
   // returns value
   // for different structure kinds has different sense
   // For kSqlVersion it version, for kSqlReference it is object id and so on

   return fValue.Data();
}

//________________________________________________________________________
void TSQLStructure::Add(TSQLStructure* child)
{
   // Add child strucure

   if (child!=0) {
      child->SetParent(this);
      fChilds.Add(child);
   }
}

//________________________________________________________________________
void TSQLStructure::AddVersion(const TClass* cl, Int_t version)
{
   // add child as version

   TSQLStructure* ver = new TSQLStructure;
   ver->SetVersion(cl, version);
   Add(ver);
}

//________________________________________________________________________
void TSQLStructure::AddValue(const char* value, const char* tname)
{
   // Add child structure as value

   TSQLStructure* child = new TSQLStructure;
   child->SetValue(value, tname);
   Add(child);
}

//________________________________________________________________________
Long64_t TSQLStructure::DefineObjectId(Bool_t recursive)
{
   // defines current object id, to which this structure belong
   // make life complicated, because some objects do not get id
   // automatically in TBufferSQL, but afterwards

   TSQLStructure* curr = this;
   while (curr!=0) {
      if ((curr->GetType()==kSqlObject) ||
          (curr->GetType()==kSqlPointer) || 
         // workaround to store object id in element structure
          (curr->GetType()==kSqlElement) ||
          (curr->GetType()==kSqlCustomElement) ||
          (curr->GetType()==kSqlCustomClass) ||
          (curr->GetType()==kSqlStreamerInfo)) {
         const char* value = curr->GetValue();
         if ((value!=0) && (strlen(value)>0))
            return sqlio::atol64(value);
      }

      curr = recursive ? curr->GetParent() : 0;
   }
   return -1;
}

//________________________________________________________________________
void TSQLStructure::SetObjectData(TSQLObjectData* objdata)
{
   // set element to be used for object data

   fType = kSqlObjectData;
   fPointer = objdata;
}

//________________________________________________________________________
void TSQLStructure::AddObjectData(TSQLObjectData* objdata)
{
   // add element with pointer to object data

   TSQLStructure* child = new TSQLStructure;
   child->SetObjectData(objdata);
   Add(child);
}

//________________________________________________________________________
TSQLObjectData* TSQLStructure::GetObjectData(Bool_t search)
{
   // searchs for objects data

   TSQLStructure* child = GetChild(0);
   if ((child!=0) && (child->GetType()==kSqlObjectData))
      return (TSQLObjectData*) child->fPointer;
   if (search && (GetParent()!=0))
      return GetParent()->GetObjectData(search);
   return 0;
}

//________________________________________________________________________
void TSQLStructure::Print(Option_t*) const
{
   // print content of complete structure

   PrintLevel(0);
}

//________________________________________________________________________
void TSQLStructure::PrintLevel(Int_t level) const
{
   // print content of current structure

   for(Int_t n=0;n<level;n++) cout << " ";
   switch (fType) {
   case 0: cout << "Undefined type"; break;
   case kSqlObject: cout << "Object ref = " << fValue; break;
   case kSqlPointer: cout << "Pointer ptr = " << fValue; break;
   case kSqlVersion: {
      const TClass* cl = (const TClass*) fPointer;
      cout << "Version cl = " << cl->GetName() << " ver = " << cl->GetClassVersion();
      break;
   }
   case kSqlStreamerInfo: {
      const TStreamerInfo* info = (const TStreamerInfo*) fPointer;
      cout << "Class: " << info->GetName();
      break;
   }
   case kSqlCustomElement:
   case kSqlElement: {
      const TStreamerElement* elem = (const TStreamerElement*) fPointer;
      cout << "Member: " << elem->GetName();
      break;
   }
   case kSqlValue: {
      cout << "Value: " << fValue;
      if (fRepeatCnt>1) cout << "  cnt:" << fRepeatCnt;
      if (fPointer!=0) cout << "  type = " << (const char*)fPointer;
      break;
   }
   case kSqlArray: {
      cout << "Array ";
      if (fValue.Length()>0) cout << "  sz = " << fValue;
      break;
   }
   case kSqlCustomClass: {
      TClass* cl = (TClass*) fPointer;
      cout << "CustomClass: " << cl->GetName() << "  ver = " << fValue;
      break;
   }
   default:
      cout << "Unknown type";
   }
   cout << endl;

   for(Int_t n=0;n<NumChilds();n++)
      GetChild(n)->PrintLevel(level+2);
}

//________________________________________________________________________
Bool_t TSQLStructure::IsNumericType(Int_t typ)
{
   // defines if value is numeric and not requires quotes when writing

   switch(typ) {
   case TStreamerInfo::kShort   : return kTRUE;
   case TStreamerInfo::kInt     : return kTRUE;
   case TStreamerInfo::kLong    : return kTRUE;
   case TStreamerInfo::kFloat   : return kTRUE;
   case TStreamerInfo::kFloat16 : return kTRUE;
   case TStreamerInfo::kCounter : return kTRUE;
   case TStreamerInfo::kDouble  : return kTRUE;
   case TStreamerInfo::kDouble32: return kTRUE;
   case TStreamerInfo::kUChar   : return kTRUE;
   case TStreamerInfo::kUShort  : return kTRUE;
   case TStreamerInfo::kUInt    : return kTRUE;
   case TStreamerInfo::kULong   : return kTRUE;
   case TStreamerInfo::kBits    : return kTRUE;
   case TStreamerInfo::kLong64  : return kTRUE;
   case TStreamerInfo::kULong64 : return kTRUE;
   case TStreamerInfo::kBool    : return kTRUE;
   }
   return kFALSE;
}

//________________________________________________________________________
const char* TSQLStructure::GetSimpleTypeName(Int_t typ)
{
   // provides name for basic types
   // used as suffix for column name or field suffix in raw table

   switch(typ) {
   case TStreamerInfo::kChar    : return sqlio::Char;
   case TStreamerInfo::kShort   : return sqlio::Short;
   case TStreamerInfo::kInt     : return sqlio::Int;
   case TStreamerInfo::kLong    : return sqlio::Long;
   case TStreamerInfo::kFloat   : return sqlio::Float;
   case TStreamerInfo::kFloat16 : return sqlio::Float;
   case TStreamerInfo::kCounter : return sqlio::Int;
   case TStreamerInfo::kDouble  : return sqlio::Double;
   case TStreamerInfo::kDouble32: return sqlio::Double;
   case TStreamerInfo::kUChar   : return sqlio::UChar;
   case TStreamerInfo::kUShort  : return sqlio::UShort;
   case TStreamerInfo::kUInt    : return sqlio::UInt;
   case TStreamerInfo::kULong   : return sqlio::ULong;
   case TStreamerInfo::kBits    : return sqlio::UInt;
   case TStreamerInfo::kLong64  : return sqlio::Long64;
   case TStreamerInfo::kULong64 : return sqlio::ULong64;
   case TStreamerInfo::kBool    : return sqlio::Bool;
   }

   return 0;
}

//___________________________________________________________

// TSqlCmdsBuffer used as buffer for data, which are correspond to
// particular class, defined by TSQLClassInfo instance
// Support both TSQLStatement and Query modes

class TSqlCmdsBuffer : public TObject {

public:
   TSqlCmdsBuffer(TSQLFile* f, TSQLClassInfo* info) :
      TObject(),
      fFile(f),
      fInfo(info),
      fBlobStmt(0),
      fNormStmt(0)
   {
   }

   virtual ~TSqlCmdsBuffer()
   {
      fNormCmds.Delete();
      fBlobCmds.Delete();
      fFile->SQLDeleteStatement(fBlobStmt);
      fFile->SQLDeleteStatement(fNormStmt);
   }

   void AddValues(Bool_t isnorm, const char* values)
   {
      TObjString* str = new TObjString(values);
      if (isnorm) fNormCmds.Add(str);
      else fBlobCmds.Add(str);
   }

   TSQLFile* fFile;
   TSQLClassInfo* fInfo;
   TObjArray fNormCmds;
   TObjArray fBlobCmds;
   TSQLStatement* fBlobStmt;
   TSQLStatement* fNormStmt;
};

//________________________________________________________________________
// TSqlRegistry keeps data, used when object data transformed to sql query or
// statements 

class TSqlRegistry : public TObject {

public:
   TSqlRegistry() :
      TObject(),
      f(0),
      fKeyId(0),
      fLastObjId(-1),
      fCmds(0),
      fFirstObjId(0),
      fCurrentObjId(0),
      fCurrentObjClass(0),
      fLastLongStrId(0),
      fPool(),
      fLongStrValues(),
      fRegValues(),
      fRegStmt(0)
   {
   }

   TSQLFile*  f;
   Long64_t   fKeyId;
   Long64_t   fLastObjId;
   TObjArray* fCmds;
   Long64_t   fFirstObjId;

   Long64_t   fCurrentObjId;
   TClass*    fCurrentObjClass;

   Int_t      fLastLongStrId;

   TMap       fPool;
   TObjArray  fLongStrValues;
   TObjArray  fRegValues;
   
   TSQLStatement* fRegStmt;


   virtual ~TSqlRegistry()
   {
      fPool.DeleteValues();
      fLongStrValues.Delete();
      fRegValues.Delete();
      f->SQLDeleteStatement(fRegStmt);
   }

   Long64_t GetNextObjId() { return ++fLastObjId; }

   void AddSqlCmd(const char* query)
   {
      // add SQL command to the list
      if (fCmds==0) fCmds = new TObjArray;
      fCmds->Add(new TObjString(query));
   }

   TSqlCmdsBuffer* GetCmdsBuffer(TSQLClassInfo* sqlinfo)
   {
      if (sqlinfo==0) return 0;
      TSqlCmdsBuffer* buf = (TSqlCmdsBuffer*) fPool.GetValue(sqlinfo);
      if (buf==0) {
         buf = new TSqlCmdsBuffer(f, sqlinfo);
         fPool.Add(sqlinfo, buf);
      }
      return buf;
   }

   void ConvertSqlValues(TObjArray& values, const char* tablename)
   {
   // this function transforms array of values for one table
   // to SQL command. For MySQL one INSERT querie can
   // contain data for more than one row

      if ((values.GetLast()<0) || (tablename==0)) return;

      Bool_t canbelong = f->IsMySQL();

      Int_t maxsize = 50000;
      TString sqlcmd(maxsize), value, onecmd, cmdmask;

      const char* quote = f->SQLIdentifierQuote();

      TIter iter(&values);
      TObject* cmd = 0;
      while ((cmd = iter())!=0) {

         if (sqlcmd.Length()==0)
            sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s)",
                        quote, tablename, quote, cmd->GetName());
         else {
            sqlcmd+=", (";
            sqlcmd += cmd->GetName();
            sqlcmd+=")";
         }

         if (!canbelong || (sqlcmd.Length()>maxsize*0.9)) {
            AddSqlCmd(sqlcmd.Data());
            sqlcmd = "";
         }
      }

      if (sqlcmd.Length()>0) AddSqlCmd(sqlcmd.Data());
   }

   void ConvertPoolValues()
   {
      TSQLClassInfo* sqlinfo = 0;
      TIter iter(&fPool);
      while ((sqlinfo = (TSQLClassInfo*) iter())!=0) {
         TSqlCmdsBuffer* buf = (TSqlCmdsBuffer*) fPool.GetValue(sqlinfo);
         if (buf==0) continue;
         ConvertSqlValues(buf->fNormCmds, sqlinfo->GetClassTableName());
         // ensure that raw table will be created
         if (buf->fBlobCmds.GetLast()>=0) f->CreateRawTable(sqlinfo);
         ConvertSqlValues(buf->fBlobCmds, sqlinfo->GetRawTableName());
         if (buf->fBlobStmt)
            buf->fBlobStmt->Process();
         if (buf->fNormStmt)
            buf->fNormStmt->Process();
      }

      ConvertSqlValues(fLongStrValues, sqlio::StringsTable);
      ConvertSqlValues(fRegValues, sqlio::ObjectsTable);
      if (fRegStmt) fRegStmt->Process();
   }


   void AddRegCmd(Long64_t objid, TClass* cl)
   {
      Long64_t indx = objid-fFirstObjId;
      if (indx<0) {
         Error("AddRegCmd","Something wrong with objid = %lld", objid);
         return;
      }
      
      if (f->IsOracle() || f->IsODBC()) {
         if ((fRegStmt==0) && f->SQLCanStatement()) {
            const char* quote = f->SQLIdentifierQuote();
            
            TString sqlcmd;
            const char* pars = f->IsOracle() ? ":1, :2, :3, :4" : "?, ?, ?, ?";
            sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s)", 
                     quote, sqlio::ObjectsTable, quote, pars);
            fRegStmt = f->SQLStatement(sqlcmd.Data(), 1000);
         }
         
         if (fRegStmt!=0) {
            fRegStmt->NextIteration();
            fRegStmt->SetLong64(0, fKeyId);
            fRegStmt->SetLong64(1, objid);
            fRegStmt->SetString(2, cl->GetName(), f->SQLSmallTextTypeLimit());
            fRegStmt->SetInt(3, cl->GetClassVersion());
            return;
         }
      }      
      
      const char* valuequote = f->SQLValueQuote();
      TString cmd;
      cmd.Form("%lld, %lld, %s%s%s, %d",
                fKeyId, objid,
                valuequote, cl->GetName(), valuequote,
                cl->GetClassVersion());
      fRegValues.AddAtAndExpand(new TObjString(cmd), indx);
   }

   Int_t AddLongString(const char* strvalue)
   {
      // add value to special string table,
      // where large (more than 255 bytes) strings are stored

      if (fLastLongStrId==0) f->VerifyLongStringTable();
      Int_t strid = ++fLastLongStrId;
      TString value = strvalue;
      const char* valuequote = f->SQLValueQuote();
      TSQLStructure::AddStrBrackets(value, valuequote);

      TString cmd;
      cmd.Form("%lld, %d, %s", fCurrentObjId, strid, value.Data());

      fLongStrValues.Add(new TObjString(cmd));

      return strid;
   }

   Bool_t InsertToNormalTableOracle(TSQLTableData* columns, TSQLClassInfo* sqlinfo)
   {
      TSqlCmdsBuffer* buf = GetCmdsBuffer(sqlinfo);
      if (buf==0) return kFALSE;
      
      TSQLStatement* stmt = buf->fNormStmt;
      if (stmt==0) {
         // if one cannot create statement, do it normal way
         if (!f->SQLCanStatement()) return kFALSE;
         
         const char* quote = f->SQLIdentifierQuote();
         TString sqlcmd;
         sqlcmd.Form("INSERT INTO %s%s%s VALUES (", 
                     quote, sqlinfo->GetClassTableName(), quote);
         for (int n=0;n<columns->GetNumColumns();n++) {
            if (n>0) sqlcmd +=", ";
            if (f->IsOracle()) {
               sqlcmd += ":"; 
               sqlcmd += (n+1);
            } else
               sqlcmd += "?";
         }
         sqlcmd += ")";
                     
         stmt = f->SQLStatement(sqlcmd.Data(), 1000);
         if (stmt==0) return kFALSE;
         buf->fNormStmt = stmt;
      }
      
      stmt->NextIteration(); 
      
      Int_t sizelimit = f->SQLSmallTextTypeLimit();
       
      for (Int_t ncol=0;ncol<columns->GetNumColumns();ncol++) {
         const char* value = columns->GetColumn(ncol);
         if (value==0) value = "";
         stmt->SetString(ncol, value, sizelimit);
      }
      
      return kTRUE;
   }

   void InsertToNormalTable(TSQLTableData* columns, TSQLClassInfo* sqlinfo)
   {
      // produce SQL query to insert object data into normal table

      if (f->IsOracle() || f->IsODBC())
         if (InsertToNormalTableOracle(columns, sqlinfo))
           return;

      const char* valuequote = f->SQLValueQuote();

      TString values;
      
      for (Int_t n=0;n<columns->GetNumColumns();n++) {
         if (n>0) values+=", "; 
         
         if (columns->IsNumeric(n))
            values+=columns->GetColumn(n);
         else {
            TString value = columns->GetColumn(n);
            TSQLStructure::AddStrBrackets(value, valuequote);
            values += value;
         }
      }
     
      TSqlCmdsBuffer* buf = GetCmdsBuffer(sqlinfo);
      if (buf!=0) buf->AddValues(kTRUE, values.Data());
   }
};


//_____________________________________________________________________________

// TSqlRawBuffer is used to convert raw data, which corresponds to one
// object and belong to single SQL tables. Supoorts both statements 
// and query mode

class TSqlRawBuffer : public TObject {

public:

   TSqlRawBuffer(TSqlRegistry* reg, TSQLClassInfo* sqlinfo) :
      TObject(),
      fFile(0),
      fInfo(0),
      fCmdBuf(0),
      fObjId(0),
      fRawId(0),
      fValueMask(),
      fValueQuote(0),
      fMaxStrSize(255)
   {
      fFile = reg->f;
      fInfo = sqlinfo;
      fCmdBuf = reg->GetCmdsBuffer(sqlinfo);
      fObjId = reg->fCurrentObjId;
      fValueQuote = fFile->SQLValueQuote();
      fValueMask.Form("%lld, %s, %s%s%s, %s", fObjId, "%d", fValueQuote, "%s", fValueQuote, "%s");      
      fMaxStrSize = reg->f->SQLSmallTextTypeLimit();
   }
   
   virtual ~TSqlRawBuffer() 
   {
      // close blob statement for Oracle
      TSQLStatement* stmt = fCmdBuf->fBlobStmt;
      if ((stmt!=0) && fFile->IsOracle()) {
         stmt->Process();
         delete stmt;
         fCmdBuf->fBlobStmt = 0;
      }
   }
   
   Bool_t IsAnyData() const { return fRawId>0; }

   void AddLine(const char* name, const char* value, const char* topname = 0, const char* ns = 0)
   {
      if (fCmdBuf==0) return;
      
      // when first line is created, check all problems
      if (fRawId==0) {
         Bool_t maketmt = kFALSE;
         if (fFile->IsOracle() || fFile->IsODBC())
            maketmt = (fCmdBuf->fBlobStmt==0) && fFile->SQLCanStatement();
            
         if (maketmt) {
            // ensure that raw table is exists
            fFile->CreateRawTable(fInfo);
            
            const char* quote = fFile->SQLIdentifierQuote();
            TString sqlcmd;
            const char* params = fFile->IsOracle() ? ":1, :2, :3, :4" : "?, ?, ?, ?";
            sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s)", 
                        quote, fInfo->GetRawTableName(), quote, params);
            TSQLStatement* stmt = fFile->SQLStatement(sqlcmd.Data(), 2000);
            fCmdBuf->fBlobStmt = stmt;
         }
      }
      
      TString buf;
      const char* fullname = name;
      if ((topname!=0) && (ns!=0)) {
         buf+=topname;
         buf+=ns;
         buf+=name;
         fullname = buf.Data();
      }
      
      TSQLStatement* stmt = fCmdBuf->fBlobStmt;
      
      if (stmt!=0) {
         stmt->NextIteration();
         stmt->SetLong64(0, fObjId);
         stmt->SetInt(1, fRawId++);
         stmt->SetString(2, fullname, fMaxStrSize);
//         Info("AddLine","name = %s value = %s",fullname, value);
         stmt->SetString(3, value, fMaxStrSize);
      } else {
         TString valuebuf(value);
         TSQLStructure::AddStrBrackets(valuebuf, fValueQuote);
         TString cmd;
         cmd.Form(fValueMask.Data(), fRawId++, fullname, valuebuf.Data());
         fCmdBuf->AddValues(kFALSE, cmd.Data());         
      }
   }
      
   TSQLFile*  fFile;
   TSQLClassInfo* fInfo;
   TSqlCmdsBuffer* fCmdBuf;
   Long64_t fObjId; 
   Int_t fRawId;
   TString fValueMask;
   const char* fValueQuote;
   Int_t fMaxStrSize;
};

//________________________________________________________________________
Long64_t TSQLStructure::FindMaxObjectId()
{
   // define maximum reference id, used for objects

   Long64_t max = DefineObjectId(kFALSE);

   for (Int_t n=0;n<NumChilds();n++) {
      Long64_t zn = GetChild(n)->FindMaxObjectId();
      if (zn>max) max = zn;
   }

   return max;
}

//________________________________________________________________________
Bool_t TSQLStructure::ConvertToTables(TSQLFile* file, Long64_t keyid, TObjArray* cmds)
{
   // Convert structure to sql statements
   // This function is called immidiately after TBufferSQL2 produces
   // this structure with object data
   // Should be only called for toplevel structure

   if ((file==0) || (cmds==0)) return kFALSE;

   TSqlRegistry reg;

   reg.fCmds = cmds;
   reg.f = file;
   reg.fKeyId = keyid;
   // this is id of main object to be stored
   reg.fFirstObjId = DefineObjectId(kFALSE);
   // this is maximum objectid which is now in use
   reg.fLastObjId = FindMaxObjectId();

   Bool_t res = StoreObject(&reg, reg.fFirstObjId, GetObjectClass());

   // convert values from pool to SQL commands
   reg.ConvertPoolValues();

   return res;
}

//________________________________________________________________________
void TSQLStructure::PerformConversion(TSqlRegistry* reg, TSqlRawBuffer* blobs, const char* topname, Bool_t useblob)
{
   // perform conversion of structure to sql statements
   // first tries convert it to normal form
   // if fails, produces data for raw table

   TString sbuf;
   const char* ns = reg->f->SQLNameSeparator();

   switch (fType) {
   case kSqlObject: {

      if (!StoreObject(reg, DefineObjectId(kFALSE), GetObjectClass())) break;

      blobs->AddLine(sqlio::ObjectRef, GetValue(), topname, ns);

      break;
   }

   case kSqlPointer: {
      blobs->AddLine(sqlio::ObjectPtr, fValue.Data(), topname,ns);
      break;
   }

   case kSqlVersion: {
      if (fPointer!=0)
         topname = ((TClass*) fPointer)->GetName();
      else
         Error("PerformConversion","version without class");
      blobs->AddLine(sqlio::Version, fValue.Data(), topname, ns);
      break;
   }

   case kSqlStreamerInfo: {

      TStreamerInfo* info = GetStreamerInfo();
      if (info==0) return;

      if (useblob) {
         for(Int_t n=0;n<=fChilds.GetLast();n++) {
            TSQLStructure* child = (TSQLStructure*) fChilds.At(n);
            child->PerformConversion(reg, blobs, info->GetName(), useblob);
         }
      } else {
         Long64_t objid = reg->GetNextObjId();
         TString sobjid;
         sobjid.Form("%lld",objid);
         if (!StoreObject(reg, objid, info->GetClass(), kTRUE)) return;
         blobs->AddLine(sqlio::ObjectInst, sobjid.Data(), topname, ns);
      }
      break;
   }

   case kSqlCustomElement:
   case kSqlElement: {
      const TStreamerElement* elem = (const TStreamerElement*) fPointer;

      Int_t indx = 0;
      while (indx<NumChilds()) {
         TSQLStructure* child = GetChild(indx++);
         child->PerformConversion(reg, blobs, elem->GetName(), useblob);
      }
      break;
   }

   case kSqlValue: {
      const char* tname = (const char*) fPointer;
      if (fArrayIndex>=0) {
         if (fRepeatCnt>1)
            sbuf.Form("%s%d%s%d%s%s%s", "[", fArrayIndex, sqlio::IndexSepar, fArrayIndex+fRepeatCnt-1, "]", ns, tname);
         else
            sbuf.Form("%s%d%s%s%s", "[", fArrayIndex, "]", ns, tname);
      } else {
         if (tname!=0) sbuf = tname;
         else sbuf = "Value";
      }

      TString buf;
      const char* value = fValue.Data();

      if ((tname==sqlio::CharStar) && (value!=0)) {
         Int_t size = strlen(value);
         if (size > reg->f->SQLSmallTextTypeLimit()) {
            Int_t strid = reg->AddLongString(value);
            buf = reg->f->CodeLongString(reg->fCurrentObjId, strid);
            value = buf.Data();
         }
      }

      blobs->AddLine(sbuf.Data(), value, (fArrayIndex>=0) ? 0 : topname, ns);

      break;
   }

   case kSqlArray: {
      if (fValue.Length()>0)
         blobs->AddLine(sqlio::Array, fValue.Data(), topname, ns);
      for(Int_t n=0;n<=fChilds.GetLast();n++) {
         TSQLStructure* child = (TSQLStructure*) fChilds.At(n);
         child->PerformConversion(reg, blobs, topname, useblob);
      }
      break;
   }
   }
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreObject(TSqlRegistry* reg, Long64_t objid, TClass* cl, Bool_t registerobj)
{
   // convert object data to sql statements
   // if normal (columnwise) representation is not possible,
   // complete object will be converted to raw format

   if ((cl==0) || (objid<0)) return kFALSE;

   if (gDebug>1) {
      cout << "Store object " << objid <<" cl = " << cl->GetName() << endl;
      if (GetStreamerInfo()) cout << "Info = " << GetStreamerInfo()->GetName() << endl; else
         if (GetElement()) cout << "Element = " << GetElement()->GetName() << endl;
   }

   Long64_t oldid = reg->fCurrentObjId;
   TClass* oldcl = reg->fCurrentObjClass;

   reg->fCurrentObjId = objid;
   reg->fCurrentObjClass = cl;

   Bool_t normstore = kFALSE;

   Bool_t res = kTRUE;

   if (cl==TObject::Class())
      normstore = StoreTObject(reg);
   else
   if (cl==TString::Class())
      normstore = StoreTString(reg);
   else
      if (GetType()==kSqlStreamerInfo)
         // this is a case when array of objects are stored in blob and each object
         // has normal streamer. Then it will be stored in normal form and only one tag
         // will be kept to remind about
         normstore = StoreClassInNormalForm(reg);
      else
         normstore = StoreObjectInNormalForm(reg);

   if (gDebug>2)
      cout << "Store object " << objid << " of class " << cl->GetName() << "  normal = " << normstore << " sqltype = " << GetType() << endl;

   if (!normstore) {

      // This is a case, when only raw table is exists

      TSQLClassInfo* sqlinfo = reg->f->RequestSQLClassInfo(cl);
      TSqlRawBuffer rawdata(reg, sqlinfo);

      for(Int_t n=0;n<NumChilds();n++) {
         TSQLStructure* child = GetChild(n);
         child->PerformConversion(reg, &rawdata, 0 /*cl->GetName()*/);
      }

      res = rawdata.IsAnyData();
   }

   if (registerobj)
      reg->AddRegCmd(objid, cl);

   reg->fCurrentObjId = oldid;
   reg->fCurrentObjClass = oldcl;

   return res;
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreObjectInNormalForm(TSqlRegistry* reg)
{
   // this function verify object child elements and
   // calls transformation to class table

   if (fChilds.GetLast()!=1) return kFALSE;

   TSQLStructure* s_ver = GetChild(0);

   TSQLStructure* s_info = GetChild(1);

   if (!CheckNormalClassPair(s_ver, s_info)) return kFALSE;

   return s_info->StoreClassInNormalForm(reg);
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreClassInNormalForm(TSqlRegistry* reg)
{
   // produces data for complete class table
   // where not possible, raw data for some elements are created

   TClass* cl = 0;
   Version_t version = 0;
   if (!GetClassInfo(cl, version)) return kFALSE;
   if (cl==0) return kFALSE;

   TSQLClassInfo* sqlinfo = reg->f->RequestSQLClassInfo(cl->GetName(), version);

   TSQLTableData columns(reg->f, sqlinfo);
   // Bool_t needblob = kFALSE;

   TSqlRawBuffer rawdata(reg, sqlinfo);

//   Int_t currrawid = 0;

   // add first column with object id
   columns.AddColumn(reg->f->SQLObjectIdColumn(), reg->fCurrentObjId);

   for(Int_t n=0;n<=fChilds.GetLast();n++) {
      TSQLStructure* child = (TSQLStructure*) fChilds.At(n);
      TStreamerElement* elem = child->GetElement();

      if (elem==0) {
         Error("StoreClassInNormalForm", "CAN NOT BE");
         continue;
      }

      if (child->StoreElementInNormalForm(reg, &columns)) continue;

      Int_t columntyp = DefineElementColumnType(elem, reg->f);
      if ((columntyp!=kColRawData) && (columntyp!=kColObjectArray)) {
         Error("StoreClassInNormalForm","Element %s typ=%d has problem with normal store ", elem->GetName(), columntyp);
         continue;
      }


      Bool_t doblobs = kTRUE;

      Int_t blobid = rawdata.fRawId; // keep id of first raw, used in class table 

      if (columntyp==kColObjectArray)
         if (child->TryConvertObjectArray(reg, &rawdata))
            doblobs = kFALSE;

      if (doblobs)
         child->PerformConversion(reg, &rawdata, elem->GetName(), kFALSE);

      if (blobid==rawdata.fRawId) 
         blobid = -1; // no data for blob was created
      else { 
         //reg->f->CreateRawTable(sqlinfo);
         //blobid = currrawid; // column will contain first raw id
         //reg->ConvertBlobs(&blobs, sqlinfo, currrawid);
         //needblob = kTRUE;
      }
      //blobs.Delete();

      TString blobname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         blobname += sqlio::RawSuffix;

      columns.AddColumn(blobname, blobid);
   }

   reg->f->CreateClassTable(sqlinfo, columns.TakeColInfos());

   reg->InsertToNormalTable(&columns, sqlinfo);

   return kTRUE;
}

//________________________________________________________________________
TString TSQLStructure::MakeArrayIndex(TStreamerElement* elem, Int_t index)
{
   // produce string with complete index like [1][2][0]

   TString res;
   if ((elem==0) || (elem->GetArrayLength()==0)) return res;

   for(Int_t ndim=elem->GetArrayDim()-1;ndim>=0;ndim--) {
      Int_t ix = index % elem->GetMaxIndex(ndim);
      index = index / elem->GetMaxIndex(ndim);
      TString buf;
      buf.Form("%s%d%s","[",ix,"]");
      res = buf + res;
   }
   return res;
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreElementInNormalForm(TSqlRegistry* reg, TSQLTableData* columns)
{
   // tries to store element data in column

   TStreamerElement* elem = GetElement();
   if (elem==0) return kFALSE;

   Int_t typ = elem->GetType();

   Int_t columntyp = DefineElementColumnType(elem, reg->f);

   if (gDebug>4)
      cout << "Element " << elem->GetName()
           << "   type = " << typ
           << "  column = " << columntyp << endl;

   TString colname = DefineElementColumnName(elem, reg->f);

   if (columntyp==kColTString) {
      const char* value;
      if (!RecognizeTString(value)) return kFALSE;

      Int_t len = value ? strlen(value) : 0;

      Int_t sizelimit = reg->f->SQLSmallTextTypeLimit();

      const char* stype = reg->f->SQLSmallTextType();

      if (len<=sizelimit)
         columns->AddColumn(colname.Data(), stype, value, kFALSE);
      else {
         Int_t strid = reg->AddLongString(value);
         TString buf = reg->f->CodeLongString(reg->fCurrentObjId, strid);
         columns->AddColumn(colname.Data(), stype, buf.Data(), kFALSE);
      }

      return kTRUE;
   }

   if (columntyp==kColParent) {
      Long64_t objid = reg->fCurrentObjId;
      TClass* basecl = elem->GetClassPointer();
      Int_t resversion = basecl->GetClassVersion();
      if (!StoreObject(reg, objid, basecl, kFALSE))
         resversion = -1;
      columns->AddColumn(colname.Data(), resversion);
      return kTRUE;
   }

   if (columntyp==kColObject) {

      Long64_t objid = -1;

      if (NumChilds()==1) {
         TSQLStructure* child = GetChild(0);

         if (child->GetType()==kSqlObject) {
            objid = child->DefineObjectId(kFALSE);
            if (!child->StoreObject(reg, objid, child->GetObjectClass())) return kFALSE;
         } else
         if (child->GetType()==kSqlPointer) {
            TString sobjid = child->GetValue();
            if (sobjid.Length()>0)
               objid = sqlio::atol64(sobjid.Data());
         }
      }

      if (objid<0) {
         //cout << "!!!! Not standard " << elem->GetName() << " class = " << elem->GetClassPointer()->GetName() << endl;
         objid = reg->GetNextObjId();
         if (!StoreObject(reg, objid, elem->GetClassPointer()))
            objid = -1;  // this is a case, when no data was stored for this object
      }

      columns->AddColumn(colname.Data(), objid);
      return kTRUE;
   }

   if (columntyp==kColNormObject) {

      if (NumChilds()!=1) {
         Error("kColNormObject","NumChilds()=%d", NumChilds());
         PrintLevel(20);
         return kFALSE;
      }
      TSQLStructure* child = GetChild(0);
      if ((child->GetType()!=kSqlPointer) && (child->GetType()!=kSqlObject)) return kFALSE;

      Bool_t normal = kTRUE;

      Long64_t objid = -1;

      if (child->GetType()==kSqlObject) {
         objid = child->DefineObjectId(kFALSE); 
         normal = child->StoreObject(reg, objid, child->GetObjectClass());
      } else {
         objid = child->DefineObjectId(kFALSE);
      }

      if (!normal) {
         Error("kColNormObject","child->StoreObject fails");
         return kFALSE;
      }

      columns->AddColumn(colname.Data(), objid);
      return kTRUE;
   }

   if (columntyp==kColNormObjectArray) {
       
      if (elem->GetArrayLength()!=NumChilds()) return kFALSE;

      for (Int_t index=0;index<NumChilds();index++) {
         TSQLStructure* child = GetChild(index);
         if ((child->GetType()!=kSqlPointer) &&
             (child->GetType()!=kSqlObject)) return kFALSE;
         Bool_t normal = kTRUE;

         Long64_t objid = child->DefineObjectId(kFALSE);

         if (child->GetType()==kSqlObject)
            normal = child->StoreObject(reg, objid, child->GetObjectClass());

         if (!normal) return kFALSE;

         colname = DefineElementColumnName(elem, reg->f, index);

         columns->AddColumn(colname.Data(), objid);
      }
      return kTRUE;
   }

   if (columntyp==kColObjectPtr) {
      if (NumChilds()!=1) return kFALSE;
      TSQLStructure* child = GetChild(0);
      if ((child->GetType()!=kSqlPointer) && (child->GetType()!=kSqlObject)) return kFALSE;

      Bool_t normal = kTRUE;
      Long64_t objid = -1;

      if (child->GetType()==kSqlObject) {
         objid = child->DefineObjectId(kFALSE);
         normal = child->StoreObject(reg, objid, child->GetObjectClass());
      }

      if (!normal) return kFALSE;

      columns->AddColumn(colname.Data(), objid);
      return kTRUE;
   }

   if (columntyp==kColSimple) {

      // only child shoud existing for element
      if (NumChilds()!=1) {
         Error("StoreElementInNormalForm","Enexpected number %d for simple element %s", NumChilds(), elem->GetName());
         return kFALSE;
      }

      TSQLStructure* child = GetChild(0);
      if (child->GetType()!=kSqlValue) return kFALSE;

      const char* value = child->GetValue();
      if (value==0) return kFALSE;

      const char* sqltype = reg->f->SQLCompatibleType(typ);

      columns->AddColumn(colname.Data(), sqltype, value, IsNumericType(typ));

      return kTRUE;
   }

   if (columntyp==kColSimpleArray) {
      // number of items should be exactly equal to number of childs

      if (NumChilds()!=1) {
         Error("StoreElementInNormalForm","In fixed array %s only array node should be", elem->GetName());
         return kFALSE;
      }
      TSQLStructure* arr = GetChild(0);

      const char* sqltype = reg->f->SQLCompatibleType(typ % 20);

      for(Int_t n=0;n<arr->NumChilds();n++) {
         TSQLStructure* child = arr->GetChild(n);
         if (child->GetType()!=kSqlValue) return kFALSE;

         const char* value = child->GetValue();
         if (value==0) return kFALSE;

         Int_t index = child->GetArrayIndex();
         Int_t last = index + child->GetRepeatCounter();

         while (index<last) {
            colname = DefineElementColumnName(elem, reg->f, index);
            columns->AddColumn(colname.Data(), sqltype, value, kTRUE);
            index++;
         }
      }
      return kTRUE;
   }

   return kFALSE;
}

//________________________________________________________________________
Bool_t TSQLStructure::TryConvertObjectArray(TSqlRegistry* reg, TSqlRawBuffer* blobs)
{
   // tries to write array of objects as lis of object refereneces
   // in _streamer_ table, while objects itself will be stored in
   // other tables. If not successfull, object data will be stored
   // in _streamer_ table

   TStreamerElement* elem = GetElement();
   if (elem==0) return kFALSE;

   if (NumChilds() % 2 !=0) return kFALSE;

   Int_t indx = 0;

   while (indx<NumChilds()) {
      TSQLStructure* s_ver = GetChild(indx++);
      TSQLStructure* s_info = GetChild(indx++);
      if (!CheckNormalClassPair(s_ver, s_info)) return kFALSE;
   }

   indx = 0;
   const char* ns = reg->f->SQLNameSeparator();

   while (indx<NumChilds()-1) {
      indx++; //TSQLStructure* s_ver = GetChild(indx++);
      TSQLStructure* s_info = GetChild(indx++);
      TClass* cl = 0;
      Version_t version = 0;
      if (!s_info->GetClassInfo(cl, version)) return kFALSE;
      Long64_t objid = reg->GetNextObjId();
      if (!s_info->StoreObject(reg, objid, cl))
         objid = -1;  // this is a case, when no data was stored for this object

      TString sobjid;
      sobjid.Form("%lld", objid);

      blobs->AddLine(sqlio::ObjectRef_Arr, sobjid.Data(), elem->GetName(), ns);
   }

   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::CheckNormalClassPair(TSQLStructure* s_ver, TSQLStructure* s_info)
{
   // check if pair of two element corresponds
   // to start of object, stored in normal form

   if ((s_ver==0) || (s_info==0) || (s_ver->GetType()!=kSqlVersion)) return kFALSE;

   TClass* ver_cl = s_ver->GetVersionClass();

   TClass* info_cl = 0;
   Version_t info_ver = 0;
   if (!s_info->GetClassInfo(info_cl, info_ver)) return kFALSE;

   if ((ver_cl==0) || (info_cl==0) || (ver_cl!=info_cl) ||
       (ver_cl->GetClassVersion()!=info_ver)) return kFALSE;

   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreTObject(TSqlRegistry* reg)
{
   // store data of TObject in special table
   // workaround custom TObject streamer

   // check if it is really Looks like TObject data
   if ((NumChilds()<3) || (NumChilds()>4)) return kFALSE;

   TSQLStructure* str_ver  = GetChild(0);
   TSQLStructure* str_id   = GetChild(1);
   TSQLStructure* str_bits = GetChild(2);
   TSQLStructure* str_prid = GetChild(3);

   if (str_ver->GetType()!=kSqlVersion) return kFALSE;
   if ((str_id->GetType()!=kSqlValue) ||
       (str_id->GetValueType()!=sqlio::UInt)) return kFALSE;
   if ((str_bits->GetType()!=kSqlValue) ||
       (str_bits->GetValueType()!=sqlio::UInt)) return kFALSE;
   if (str_prid!=0)
      if ((str_prid->GetType()!=kSqlValue) ||
          (str_prid->GetValueType()!=sqlio::UShort)) return kFALSE;

   TSQLClassInfo* sqlinfo = reg->f->RequestSQLClassInfo(TObject::Class());

   if (sqlinfo==0) return kFALSE;

   TSQLTableData columns(reg->f, sqlinfo);

   const char* uinttype = reg->f->SQLCompatibleType(TStreamerInfo::kUInt);

   columns.AddColumn(reg->f->SQLObjectIdColumn(), reg->fCurrentObjId);

   columns.AddColumn(sqlio::TObjectUniqueId, uinttype, str_id->GetValue(), kTRUE);
   columns.AddColumn(sqlio::TObjectBits, uinttype, str_bits->GetValue(), kTRUE);
   columns.AddColumn(sqlio::TObjectProcessId, "CHAR(3)", (str_prid ? str_prid->GetValue() : ""), kFALSE);

   reg->f->CreateClassTable(sqlinfo, columns.TakeColInfos());

   reg->InsertToNormalTable(&columns, sqlinfo);

   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreTString(TSqlRegistry* reg)
{
   // store data of TString in special table
   // it is required when TString stored as pointer and reference to it possible

   const char* value = 0;
   if (!RecognizeTString(value)) return kFALSE;

   TSQLClassInfo* sqlinfo = reg->f->RequestSQLClassInfo(TString::Class());
   if (sqlinfo==0) return kFALSE;

   TSQLTableData columns(reg->f, sqlinfo);

   columns.AddColumn(reg->f->SQLObjectIdColumn(), reg->fCurrentObjId);
   columns.AddColumn(sqlio::TStringValue, reg->f->SQLBigTextType(), value, kFALSE);

   reg->f->CreateClassTable(sqlinfo, columns.TakeColInfos());
   
   reg->InsertToNormalTable(&columns, sqlinfo);
   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::RecognizeTString(const char* &value)
{
   // prove that structure containes TString data

   value = 0;

   if ((NumChilds()==0) || (NumChilds()>3)) return kFALSE;

   TSQLStructure *len=0, *lenbig=0, *chars=0;
   for (Int_t n=0;n<NumChilds();n++) {
      TSQLStructure* curr = GetChild(n);
      if (curr->fType!=kSqlValue) return kFALSE;
      if (curr->fPointer==sqlio::UChar) {
         if (len==0) len=curr; else return kFALSE;
      } else
         if (curr->fPointer==sqlio::Int) {
            if (lenbig==0) lenbig=curr; else return kFALSE;
         } else
            if (curr->fPointer==sqlio::CharStar) {
               if (chars==0) chars=curr; else return kFALSE;
            } else return kFALSE;
   }

   if (len==0) return kFALSE;
   if ((lenbig!=0) && ((chars==0) || (len==0))) return kFALSE;

   if (chars!=0)
      value = chars->GetValue();

   return kTRUE;
}

//________________________________________________________________________
Int_t TSQLStructure::DefineElementColumnType(TStreamerElement* elem, TSQLFile* f)
{
   // defines which kind of column can be assigned for this element
   // Possible cases
   //    kColSimple       -  basic data type
   //    kColSimpleArray  -  fixed arary of basic types
   //    kColParent       -  parent class
   //    kColObject       -  object as data memeber
   //    kColObjectPtr    -  object as pointer
   //    kColTString      -  TString
   //    kColRawData      -  anything else as raw data

   if (elem==0) return kColUnknown;

   Int_t typ = elem->GetType();

   if (typ == TStreamerInfo::kMissing) return kColRawData;

   if ((typ>0) && (typ<20) &&
       (typ!=TStreamerInfo::kCharStar)) return kColSimple;

   if ((typ>TStreamerInfo::kOffsetL) &&
       (typ<TStreamerInfo::kOffsetP))
      if ((f->GetArrayLimit()<0) ||
          (elem->GetArrayLength()<=f->GetArrayLimit()))
         return kColSimpleArray;

   if (typ==TStreamerInfo::kTObject) {
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;
      else
         return kColObject;
   }

   if (typ==TStreamerInfo::kTNamed) {
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;
      else
         return kColObject;
   }

   if (typ==TStreamerInfo::kTString) return kColTString;

   if (typ==TStreamerInfo::kBase) return kColParent;

   if (typ==TStreamerInfo::kSTL)
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;

   // this is workaround
   // these two tags stored with WriteFastArray, but read with cl->Streamer()
   if ((typ==TStreamerInfo::kObject)  ||
       (typ==TStreamerInfo::kAny)) {
      if (elem->GetArrayLength()==0)
         return kColObject;
      else
         if (elem->GetStreamer()==0)
            return kColObjectArray;
   }

   if ((typ==TStreamerInfo::kObject)  ||
       (typ==TStreamerInfo::kAny) ||
       (typ==TStreamerInfo::kAnyp) ||
       (typ==TStreamerInfo::kObjectp) ||
       (typ==TStreamerInfo::kAnyP) ||
       (typ==TStreamerInfo::kObjectP)) {
      if ((elem->GetArrayLength()==0) ||
         (elem->GetStreamer()!=0))
         return kColNormObject;
      else
         return kColNormObjectArray;
   }

   if ((typ==TStreamerInfo::kObject + TStreamerInfo::kOffsetL) ||
       (typ==TStreamerInfo::kAny + TStreamerInfo::kOffsetL) ||
       (typ==TStreamerInfo::kAnyp + TStreamerInfo::kOffsetL) ||
       (typ==TStreamerInfo::kObjectp + TStreamerInfo::kOffsetL) ||
       (typ==TStreamerInfo::kAnyP + TStreamerInfo::kOffsetL) ||
       (typ==TStreamerInfo::kObjectP + TStreamerInfo::kOffsetL)) {
      if (elem->GetStreamer()!=0)
         return kColNormObject;
      else
         return kColNormObjectArray;
   }

   if ((typ==TStreamerInfo::kObject) ||
       (typ==TStreamerInfo::kAny) ||
       (typ==TStreamerInfo::kAnyp) ||
       (typ==TStreamerInfo::kObjectp) ||
       (typ==TStreamerInfo::kSTL)) {
      if (elem->GetArrayLength()==0)
         return kColObject;
      else
         if (elem->GetStreamer()==0)
            return kColObjectArray;
   }

   if (((typ==TStreamerInfo::kAnyP) ||
        (typ==TStreamerInfo::kObjectP)) &&
       (elem->GetArrayDim()==0)) return kColObjectPtr;

   //   if ((typ==TStreamerInfo::kSTLp) &&
   //       (elem->GetArrayDim()==0)) {
   //      TStreamerSTL* stl = dynamic_cast<TStreamerSTL*> (elem);
   //      if ((stl!=0) && (dynamic_cast<TStreamerSTLstring*>(elem)==0))
   //        return kColObjectPtr;
   //   }

   return kColRawData;
}

//________________________________________________________________________
TString TSQLStructure::DefineElementColumnName(TStreamerElement* elem, TSQLFile* f, Int_t indx)
{
   // returns name of the column in class table for that element

   TString colname = "";

   Int_t coltype = DefineElementColumnType(elem, f);
   if (coltype==kColUnknown) return colname;

   const char* elemname = elem->GetName();

   switch (coltype) {
   case kColSimple: {
      colname = elemname;
      if (f->GetUseSuffixes()) {
         colname+=f->SQLNameSeparator();
         colname+=GetSimpleTypeName(elem->GetType());
      }
      break;
   }

   case kColSimpleArray: {
      colname = elemname;
      colname+=MakeArrayIndex(elem, indx);
      break;
   }

   case kColParent: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname+=sqlio::ParentSuffix;
      break;
   }

   case kColNormObject: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::ObjectSuffix;
      break;
   }

   case kColNormObjectArray: {
      colname = elemname;
      colname+=MakeArrayIndex(elem, indx);
      if (f->GetUseSuffixes())
         colname += sqlio::ObjectSuffix;
      break;
   }

   case kColObject: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::ObjectSuffix;
      break;
   }

   case kColObjectPtr: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::PointerSuffix;
      break;
   }

   case kColTString: {
      colname = elem->GetName();
      if (f->GetUseSuffixes())
         colname+=sqlio::StrSuffix;
      break;
   }

   case kColRawData: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::RawSuffix;
      break;
   }

   case kColObjectArray: {
      colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::RawSuffix;
      break;
   }
   }

   return colname;
}

//________________________________________________________________________
Int_t TSQLStructure::LocateElementColumn(TSQLFile* f, TBufferSQL2* buf, TSQLObjectData* data)
{
   // find column in TSQLObjectData object, which correspond to current element

   TStreamerElement* elem = GetElement();
   if ((elem==0) || (data==0)) return kColUnknown;

   Int_t coltype = DefineElementColumnType(elem, f);

   if (gDebug>4)
      cout <<"TSQLStructure::LocateElementColumn " << elem->GetName() <<
         " coltyp = " << coltype << " : " << elem->GetType() << " len = " << elem->GetArrayLength() << endl;

   if (coltype==kColUnknown) return kColUnknown;

   const char* elemname = elem->GetName();
   Bool_t located = kFALSE;

   TString colname = DefineElementColumnName(elem, f);

   if (gDebug>4)
      cout << "         colname = " << colname << " in " <<
            data->GetInfo()->GetClassTableName() << endl;

   switch (coltype) {
   case kColSimple: {
      located = data->LocateColumn(colname.Data());
      break;
   }

   case kColSimpleArray: {
      located = data->LocateColumn(colname);
      break;
   }

   case kColParent: {
      located = data->LocateColumn(colname.Data());
      if (located==kColUnknown) return kColUnknown;

      Long64_t objid = DefineObjectId(kTRUE);
      const char* clname = elemname;
      Version_t version = atoi(data->GetValue());

      // this is a case, when parent store nothing in the database
      if (version<0) break;

      // special treatment for TObject
      if (strcmp(clname,TObject::Class()->GetName())==0) {
         UnpackTObject(f, buf, data, objid, version);
         break;
      }

      TSQLClassInfo* sqlinfo = f->FindSQLClassInfo(clname, version);
      if (sqlinfo==0) return kColUnknown;

      // this will indicate that streamer is completely custom
      if (sqlinfo->IsClassTableExist()) {
         data->AddUnpackInt(sqlio::Version, version);
      } else {
         TSQLObjectData* objdata = buf->SqlObjectData(objid, sqlinfo);
         if ((objdata==0) || !objdata->PrepareForRawData()) return kColUnknown;
         AddObjectData(objdata);
      }

      break;
   }

      // This is a case when streamer of object will be called directly.
      // Typically it happens when object is data memeber of the class.
      // Here we need to define class of object and if it was written by
      // normal streamer (via TStreamerInfo methods) or directly as blob.
      // When blob was used, blob data should be readed.
      // In normal case only version is required. Other object data will be
      // read by TBufferSQL2::IncrementLevel method
   case kColObject: {
      located = data->LocateColumn(colname.Data());
      if (located==kColUnknown) return located;

      const char* strobjid = data->GetValue();
      if (strobjid==0) return kColUnknown;

      Long64_t objid = sqlio::atol64(strobjid);

      // when nothing was stored, nothing need to be read. skip
      if (objid<0) break;

      TString clname;
      Version_t version;

      if (!buf->SqlObjectInfo(objid, clname, version)) return kColUnknown;

      // special treatment for TObject
      if (clname==TObject::Class()->GetName()) {
         UnpackTObject(f, buf, data, objid, version);
         break;
      }

      TSQLClassInfo* sqlinfo = f->FindSQLClassInfo(clname.Data(), version);
      if (sqlinfo==0) return kColUnknown;

      if (sqlinfo->IsClassTableExist()) {
         data->AddUnpackInt(sqlio::Version, version);
      } else {
         TSQLObjectData* objdata = buf->SqlObjectData(objid, sqlinfo);
         if ((objdata==0) || !objdata->PrepareForRawData()) return kColUnknown;
         AddObjectData(objdata);
      }

      // work around to store objid of object, which is memeber of class
      fValue = strobjid;

      break;
   }

      // this is case of pointer on any object
      // field contains objectid.
      // Object id, class of object and so on will be checked
      // when TBuffer::ReadObject method will be called
   case kColObjectPtr: {
      located = data->LocateColumn(colname.Data());
      break;
   }

   // this is case of on object which is treated normally in TBuffer
   // field should contains objectid.
   // Object id, class of object and so on will be checked
   // when TBuffer::StreamObject method will be called
   case kColNormObject: {
      located = data->LocateColumn(colname.Data());
      break;
   }

   case kColNormObjectArray: {
      located = data->LocateColumn(colname.Data());
      break;
   }

   case kColTString: {
      located = data->LocateColumn(colname);
      if (located==kColUnknown) return located;
      const char* value = data->GetValue();

      Long64_t objid = DefineObjectId(kTRUE);
      Int_t strid = f->IsLongStringCode(objid, value);

      TString buf2;

      // if special prefix found, than try get such string
      if (strid>0)
         if (f->GetLongString(objid, strid, buf2))
            value = buf2.Data();

      Int_t len = (value==0) ? 0 : strlen(value);
      if (len<255) {
         data->AddUnpackInt(sqlio::UChar, len);
      } else {
         data->AddUnpackInt(sqlio::UChar, 255);
         data->AddUnpackInt(sqlio::Int, len);
      }
      if (len>0)
         data->AddUnpack(sqlio::CharStar, value);
      break;
   }

   case kColRawData: {
      located = data->LocateColumn(colname.Data(), kTRUE);
      break;
   }

   case kColObjectArray: {
      located = data->LocateColumn(colname.Data(), kTRUE);
      break;
   }
   }

   if (!located) coltype = kColUnknown;

   return coltype;
}

//________________________________________________________________________
Bool_t TSQLStructure::UnpackTObject(TSQLFile* f, TBufferSQL2* buf, TSQLObjectData* data, Long64_t objid, Int_t clversion)
{
   // Unpack TObject data in form, understodable by custom TObject streamer

   TSQLClassInfo* sqlinfo = f->FindSQLClassInfo(TObject::Class()->GetName(), clversion);
   if (sqlinfo==0) return kFALSE;

   TSQLObjectData* tobjdata = buf->SqlObjectData(objid, sqlinfo);
   if (tobjdata==0) return kFALSE;

   data->AddUnpackInt(sqlio::Version, clversion);

   tobjdata->LocateColumn(sqlio::TObjectUniqueId);
   data->AddUnpack(sqlio::UInt, tobjdata->GetValue());
   tobjdata->ShiftToNextValue();

   tobjdata->LocateColumn(sqlio::TObjectBits);
   data->AddUnpack(sqlio::UInt, tobjdata->GetValue());
   tobjdata->ShiftToNextValue();

   tobjdata->LocateColumn(sqlio::TObjectProcessId);
   const char* value = tobjdata->GetValue();
   if ((value!=0) && (strlen(value)>0))
      data->AddUnpack(sqlio::UShort, value);

   delete tobjdata;

   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::UnpackTString(TSQLFile* f, TBufferSQL2* buf, TSQLObjectData* data, Long64_t objid, Int_t clversion)
{
   // Unpack TString data in form, understodable by custom TString streamer

   TSQLClassInfo* sqlinfo = f->FindSQLClassInfo(TString::Class()->GetName(), clversion);
   if (sqlinfo==0) return kFALSE;

   TSQLObjectData* tstringdata = buf->SqlObjectData(objid, sqlinfo);
   if (tstringdata==0) return kFALSE;

   tstringdata->LocateColumn(sqlio::TStringValue);

   const char* value = tstringdata->GetValue();

   Int_t len = (value==0) ? 0 : strlen(value);
   if (len<255) {
      data->AddUnpackInt(sqlio::UChar, len);
   } else {
      data->AddUnpackInt(sqlio::UChar, 255);
      data->AddUnpackInt(sqlio::Int, len);
   }
   if (len>0)
      data->AddUnpack(sqlio::CharStar, value);

   delete tstringdata;

   return kTRUE;
}

//________________________________________________________________________
void TSQLStructure::AddStrBrackets(TString &s, const char* quote)
{
   // adds quotes arround string value and replaces some special symbols
   if (strcmp(quote,"\"")==0) s.ReplaceAll("\"","\\\"");
                        else  s.ReplaceAll("'","''");
   s.Prepend(quote);
   s.Append(quote);
}
