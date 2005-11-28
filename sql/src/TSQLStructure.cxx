// @(#)root/net:$Name:  $:$Id: TSQLStructure.cxx,v 1.3 2005/11/24 16:57:23 pcanal Exp $
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
#include "TClass.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TObjString.h"
#include "TClonesArray.h"

#include "TSQLFile.h"
#include "TSQLClassInfo.h"
#include "TSQLObjectData.h"

namespace sqlio {
   const Int_t Ids_NullPtr       = 0; // used to identify NULL pointer in tables
   const Int_t Ids_RootDir       = 0; // dir:id, used for keys stored in root directory.
   const Int_t Ids_StreamerInfos = 0; // keyid used to store StreamerInfos in ROOT directory
   const Int_t Ids_FirstKey      = 1; // first key id, which is used in KeysTable (beside streamer info or something else)
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
   const char* StringsTable   = "StringsTable";
   const char* ConfigTable    = "Configurations";

   // colummns in Keys table
   const char* KT_Name      = "Name";
   const char* KT_Datetime  = "Datime";
   const char* KT_Cycle     = "Cycle";
   const char* KT_Class     = "Class";

   // colummns in Objects table
   const char* OT_Class     = "Class";
   const char* OT_Version   = "Version";

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
TSQLColumnData::TSQLColumnData(const char* name, Int_t value) :
   TObject(),
   fName(name),
   fType("INT"),
   fValue(),
   fNumeric(kTRUE)
{
   // constructs TSQLColumnData object for integer column

   fValue.Form("%d",value);
}

//________________________________________________________________________
TSQLColumnData::~TSQLColumnData()
{
   // TSQLColumnData destructor
}

//________________________________________________________________________

ClassImp(TSQLStructure)

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
void TSQLStructure::SetObjectRef(Int_t refid, const TClass* cl)
{
   // set structure type as kSqlObject

   fType = kSqlObject;
   fValue.Form("%d",refid);
   fPointer = cl;
}

//________________________________________________________________________
void TSQLStructure::SetObjectPointer(Int_t ptrid)
{
   // set structure type as kSqlPointer

   fType = kSqlPointer;
   fValue.Form("%d",ptrid);
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

   return (fType==kSqlElement) ? (TStreamerElement*) fPointer : 0;
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
Int_t TSQLStructure::DefineObjectId()
{
   // defines current object id, to which this structure belong
   // make life complicated, because some objects do not get id
   // automatically in TBufferSQL, but afterwards

   TSQLStructure* curr = this;
   while (curr!=0) {
      // workaround to store object id in element structure
      if ((curr->GetType()==kSqlElement) ||
          (curr->GetType()==kSqlStreamerInfo)) {
         const char* value = curr->GetValue();
         if ((value!=0) && (strlen(value)>0)) return atoi(value);
      }

      if (curr->GetType()==kSqlObject)
         return atoi(curr->GetValue());
      curr = curr->GetParent();
   }
   return -1;
}

//________________________________________________________________________
Bool_t TSQLStructure::IsClonesArray()
{
   // defines if this structure below node, which correspondes to
   // clones array. Used to force convertion of all data into raw format

   TSQLStructure* curr = this;
   while (curr!=0) {
      if (curr->GetType()==kSqlObject)
         return curr->GetObjectClass()==TClonesArray::Class();

      // workaround for nested TClonesArray
      if ((curr->GetType()==kSqlElement) ||
          (curr->GetType()==kSqlStreamerInfo)) {
         const char* value = curr->GetValue();
         // check that element has object id and one can analyse it class
         if ((value!=0) && (strlen(value)>0)) {
            TStreamerInfo* info = GetStreamerInfo();
            if (info!=0)
               return info->GetClass()==TClonesArray::Class();
            TStreamerElement* elem = GetElement();
            if (elem!=0)
               return elem->GetClassPointer()==TClonesArray::Class();
            return kFALSE;
         }
      }

      curr = curr->GetParent();
   }
   return kFALSE;
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
   default:
      cout << "Unknown type";
   }
   cout << endl;

   for(Int_t n=0;n<NumChilds();n++)
      GetChild(n)->PrintLevel(level+2);
}

//________________________________________________________________________
void TSQLStructure::AddCmd(TObjArray* cmds,
                           const char* name, const char* value,
                           const char* topname, const char* ns)
{
   // Add SQL command for raw table

   if ((topname!=0) && (ns!=0)) {
      TString buf;
      buf+=topname;
      buf+=ns;
      buf+=name;
      cmds->Add(new TNamed(buf.Data(), value));
   } else
      cmds->Add(new TNamed(name, value));
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

//________________________________________________________________________
class TSqlRegistry : public TObject {

public:
   TSqlRegistry() :
      TObject(),
      f(0),
      fKeyId(0),
      fLastObjId(-1),
      fCmds(0),
      fFirstObjId(0),
      fRegCmds(),
      fCurrentObjId(),
      fCurrentObjClass(0),
      fLastLongStrId(0)
   {
   }

   TSQLFile*  f;
   Int_t      fKeyId;
   Int_t      fLastObjId;
   TObjArray* fCmds;
   Int_t      fFirstObjId;
   TObjArray  fRegCmds;

   TString    fCurrentObjId;
   TClass*    fCurrentObjClass;

   Int_t      fLastLongStrId;

   virtual ~TSqlRegistry() {}

   Int_t GetNextObjId() { return ++fLastObjId; }

   void AddSqlCmd(const char* query)
   {
      // add SQL command to the list
      if (fCmds==0) fCmds = new TObjArray;
      fCmds->Add(new TObjString(query));
   }

   void AddBrackets(TString& s, const char* quote)
   {
      // add brackets to the string value
      if (strcmp(quote,"\"")==0) s.ReplaceAll("\"","\\\"");
      else  s.ReplaceAll("'","''");
      s.Prepend(quote);
      s.Append(quote);
   }

   void AddRegCmd(const char* cmd, Int_t objid)
   {
      // add special command to register object in objects table

      Int_t indx = objid-fFirstObjId;
      if (indx<0)
         Error("AddRegCmd","Soemthing wrong");
      else
         fRegCmds.AddAtAndExpand(new TObjString(cmd), indx);
   }

   Int_t AddLongString(const char* strvalue)
   {
      // add value to special string table,
      // where large (more than 255 bytes) strings are stored

      if (fLastLongStrId==0) f->VerifyLongStringTable();
      Int_t strid = ++fLastLongStrId;
      TString value = strvalue;
      const char* valuequote = f->SQLValueQuote();
      const char* quote = f->SQLIdentifierQuote();
      AddBrackets(value, valuequote);

      TString cmd;
      cmd.Form("INSERT INTO %s%s%s VALUES (%s, %d, %s)",
               quote, sqlio::StringsTable, quote,
               fCurrentObjId.Data(), strid, value.Data());
      AddSqlCmd(cmd.Data());
      return strid;
   }

   void ConvertRawOracle(TObjArray* blobs, TSQLClassInfo* sqlinfo, Int_t& rawid)
   {
      // special code for Oracle, which does not allow multiple INSERT syntax
      // therefore eachline in raw table should be inseretd with
      // separate INSERT command

      TString sqlcmd, value;
      TIter iter(blobs);
      TNamed* cmd = 0;
      const char* valuequote = f->SQLValueQuote();
      const char* quote = f->SQLIdentifierQuote();
      while ((cmd = (TNamed*)iter())!=0) {
         value = cmd->GetTitle();
         AddBrackets(value, valuequote);

         sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s, %d, %s%s%s, %s)",
                     quote, sqlinfo->GetRawTableName(), quote,
                     fCurrentObjId.Data(),
                     rawid++,
                     valuequote, cmd->GetName(), valuequote,
                     value.Data());
         AddSqlCmd(sqlcmd.Data());
      }
   }

   void ConvertBlobs(TObjArray* blobs, TSQLClassInfo* sqlinfo, Int_t& rawid)
   {
      // this function transforms blob pairs (field, value) to SQL command
      // one command includes more than one row to improve speed

      if (f->IsOracle()) {
         ConvertRawOracle(blobs, sqlinfo, rawid);
         return;
      }

      Int_t maxsize = 50000;
      TString sqlcmd(maxsize), value, onecmd, cmdmask;

      const char* valuequote = f->SQLValueQuote();
      const char* quote = f->SQLIdentifierQuote();

      cmdmask.Form("(%s, %s, %s%s%s, %s)", fCurrentObjId.Data(), "%d", valuequote, "%s", valuequote, "%s");

      TIter iter(blobs);
      TNamed* cmd = 0;
      while ((cmd = (TNamed*)iter())!=0) {
         value = cmd->GetTitle();
         AddBrackets(value, valuequote);
         onecmd.Form(cmdmask.Data(), rawid++, cmd->GetName(), value.Data());

         if (sqlcmd.Length()==0)
            sqlcmd.Form("INSERT INTO %s%s%s VALUES %s",
                        quote, sqlinfo->GetRawTableName(), quote,
                        onecmd.Data());
         else {
            sqlcmd+=", ";
            sqlcmd += onecmd;
         }

         if (sqlcmd.Length()>maxsize*0.9) {
            AddSqlCmd(sqlcmd.Data());
            sqlcmd = "";
         }
      }

      if (sqlcmd.Length()>0) AddSqlCmd(sqlcmd.Data());
   }

   void InsertToNormalTable(TObjArray* columns, const char* tablename)
   {
      // produce SQL query to insert object data into normal table

      TString names, values;

      TIter iter(columns);
      TSQLColumnData* col;
      const char* quote = f->SQLIdentifierQuote();
      const char* valuequote = f->SQLValueQuote();

      Bool_t forcequote = f->IsOracle();

      while ((col=(TSQLColumnData*)iter())!=0) {
         if (names.Length()>0) names+=", ";
         const char* colname = col->GetName();
         if (forcequote || (strpbrk(colname,"[:.]<>")!=0)) {
            names+=quote;
            names+=colname;
            names+=quote;
         } else names+=colname;

         if (values.Length()>0) values+=", ";
         if (col->IsNumeric())
            values+=col->GetValue();
         else {
            TString value = col->GetValue();
            AddBrackets(value, valuequote);
            values += value;
         }
      }

      TString cmd;
      cmd.Form("INSERT INTO %s%s%s (%s) VALUES(%s)",
               quote, tablename, quote,
               names.Data(), values.Data());
      AddSqlCmd(cmd.Data());
   }
};

//________________________________________________________________________
Int_t TSQLStructure::FindMaxRef()
{
   // define maximum reference id, used for objects

   Int_t max = 0;
   if ((GetType()==kSqlPointer) || (GetType()==kSqlObject))
      max = atoi(fValue.Data());

   for (Int_t n=0;n<NumChilds();n++) {
      Int_t zn = GetChild(n)->FindMaxRef();
      if (zn>max) max = zn;
   }

   return max;
}

//________________________________________________________________________
Bool_t TSQLStructure::ConvertToTables(TSQLFile* file, Int_t keyid, TObjArray* cmds)
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
   reg.fFirstObjId = atoi(GetValue()); // this is id of main object to be stored
   // this is maximum objectid which is now in use
   reg.fLastObjId = FindMaxRef();

   Bool_t res = StoreObject(&reg, GetValue(), GetObjectClass());

   cmds->AddAll(&(reg.fRegCmds));

   return res;
}

//________________________________________________________________________
void TSQLStructure::PerformConversion(TSqlRegistry* reg, TObjArray* blobs, const char* topname, Bool_t useblob)
{
   // perform conversion of structure to sql statements
   // first tries convert it to normal form
   // if fails, produces data for raw table

   TString sbuf;
   const char* ns = reg->f->SQLNameSeparator();

   switch (fType) {
   case kSqlObject: {

      if (!StoreObject(reg, GetValue(), GetObjectClass())) break;

      AddCmd(blobs, sqlio::ObjectRef, GetValue(), topname, ns);

      break;
   }

   case kSqlPointer: {
      AddCmd(blobs, sqlio::ObjectPtr, fValue.Data(), topname,ns);
      break;
   }

   case kSqlVersion: {
      if (fPointer!=0)
         topname = ((TClass*) fPointer)->GetName();
      else
         Error("PerformConversion","version without class");
      AddCmd(blobs, sqlio::Version, fValue.Data(), topname, ns);
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
         Int_t objid = reg->GetNextObjId();
         TString sobjid;
         sobjid.Form("%d",objid);
         if (!StoreObject(reg, sobjid.Data(), info->GetClass(), kTRUE)) return;
         AddCmd(blobs, sqlio::ObjectInst, sobjid.Data(), topname, ns);
      }
      break;
   }

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
            buf = reg->f->CodeLongString(atoi(reg->fCurrentObjId.Data()), strid);
            value = buf.Data();
         }
      }

      AddCmd(blobs, sbuf.Data(), value, (fArrayIndex>=0) ? 0 : topname, ns);

      break;
   }

   case kSqlArray: {
      if (fValue.Length()>0)
         AddCmd(blobs, sqlio::Array, fValue.Data(), topname, ns);
      for(Int_t n=0;n<=fChilds.GetLast();n++) {
         TSQLStructure* child = (TSQLStructure*) fChilds.At(n);
         child->PerformConversion(reg, blobs, topname, useblob);
      }
      break;
   }
   }
}

//________________________________________________________________________
Bool_t TSQLStructure::StoreObject(TSqlRegistry* reg, const char* objid, TClass* cl, Bool_t registerobj)
{
   // convert object data to sql statements
   // if normal (columnwise) representation is not possible,
   // complete object will be converted to raw format

   if (cl==0) return kFALSE;

   if (gDebug>1) {
      cout << "Store object " << objid <<" cl = " << cl->GetName() << endl;
      if (GetStreamerInfo()) cout << "Info = " << GetStreamerInfo()->GetName() << endl; else
         if (GetElement()) cout << "Element = " << GetElement()->GetName() << endl;
   }

   TSQLClassInfo* sqlinfo = 0;

   TString oldid = reg->fCurrentObjId;
   TClass* oldcl = reg->fCurrentObjClass;

   reg->fCurrentObjId = objid;
   reg->fCurrentObjClass = cl;

   Bool_t normstore = kFALSE;

   Bool_t res = kTRUE;

   Bool_t isclonesarray = (cl==TClonesArray::Class());

   if (cl==TObject::Class())
      normstore = StoreTObject(reg);
   else
      if (cl==TString::Class())
         normstore = StoreTString(reg);
      else
         if (!isclonesarray)
            if (GetType()==kSqlStreamerInfo)
               // this is a case when array of objects are stored in blob and each object
               // has normal streamer. Then it will be stored in normal form and only one tag
               // will be kept to remind about
               normstore = StoreClassInNormalForm(reg);
            else
               normstore = StoreObjectInNormalForm(reg);

   if (gDebug>2)
      cout << "Store object " << objid << " of class " << cl->GetName() << "  normal = " << normstore << endl;

   if (!normstore) {

      TObjArray objblobs;

      // when TClonesArray, all data will be stored in raw format
      for(Int_t n=0;n<NumChilds();n++) {
         TSQLStructure* child = GetChild(n);
         child->PerformConversion(reg, &objblobs, 0 /*cl->GetName()*/, isclonesarray);
      }

      if (objblobs.GetLast()<0)
         res = kFALSE;
      else {
         sqlinfo = reg->f->RequestSQLClassInfo(cl);
         Int_t currrawid = 0;
         reg->ConvertBlobs(&objblobs, sqlinfo, currrawid);
         reg->f->SyncSQLClassInfo(sqlinfo, 0, kTRUE);
      }

      objblobs.Delete();
   }

   if (registerobj) {
      Int_t objidint = atoi(objid);
      TString regcmd = reg->f->SetObjectDataCmd(reg->fKeyId, objidint, cl);
      reg->AddRegCmd(regcmd.Data(), objidint);
   }

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

   TStreamerInfo* info = GetStreamerInfo();
   if (info==0) return kFALSE;
   TClass* cl = info->GetClass();
   Int_t version = info->GetClassVersion();

   TSQLClassInfo* sqlinfo = reg->f->RequestSQLClassInfo(cl->GetName(), version);

   TObjArray columns;
   Bool_t needblob = kFALSE;

   Int_t currrawid = 0;

   // add first column with object id
   columns.Add(new TSQLColumnData(reg->f->SQLObjectIdColumn(), reg->f->SQLIntType(), reg->fCurrentObjId, kTRUE));

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

      TObjArray blobs;

      Bool_t doblobs = kTRUE;

      if (columntyp==kColObjectArray)
         if (child->TryConvertObjectArray(reg, &blobs))
            doblobs = kFALSE;

      if (doblobs)
         child->PerformConversion(reg, &blobs, elem->GetName(), kFALSE);

      Int_t blobid = -1;
      if (blobs.GetLast()>=0) {
         blobid = currrawid; // column will contain first raw id
         reg->ConvertBlobs(&blobs, sqlinfo, currrawid);
         needblob = kTRUE;
      }
      blobs.Delete();

      TString blobname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         blobname += sqlio::RawSuffix;

      columns.Add(new TSQLColumnData(blobname, blobid));
   }

   reg->f->SyncSQLClassInfo(sqlinfo, &columns, needblob);

   reg->InsertToNormalTable(&columns, sqlinfo->GetClassTableName());

   columns.Delete();

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
Bool_t TSQLStructure::StoreElementInNormalForm(TSqlRegistry* reg, TObjArray* columns)
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

   if (columntyp==kColTString) {
      const char* value;
      if (!RecognizeTString(value)) return kFALSE;

      Int_t len = value ? strlen(value) : 0;

      Int_t sizelimit = reg->f->SQLSmallTextTypeLimit();

      const char* stype = reg->f->SQLSmallTextType();

      TString colname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         colname+=sqlio::StrSuffix;

      if (len<=sizelimit)
         columns->Add(new TSQLColumnData(colname.Data(), stype, value, kFALSE));
      else {
         Int_t strid = reg->AddLongString(value);
         TString buf = reg->f->CodeLongString(atoi(reg->fCurrentObjId.Data()), strid);
         columns->Add(new TSQLColumnData(colname.Data(), stype, buf.Data(), kFALSE));
      }

      return kTRUE;
   }

   if (columntyp==kColParent) {
      TString objid = reg->fCurrentObjId; // DefineObjectIdStr();
      TClass* basecl = elem->GetClassPointer();
      Int_t resversion = basecl->GetClassVersion();
      if (!StoreObject(reg, objid.Data(), basecl, kFALSE))
         resversion = -1;
      TString colname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         colname+=sqlio::ParentSuffix;
      columns->Add(new TSQLColumnData(colname.Data(), resversion));
      return kTRUE;
   }

   if (columntyp==kColObject) {
      Int_t objid = reg->GetNextObjId();
      TString sobjid;
      sobjid.Form("%d",objid);

      if (!StoreObject(reg, sobjid.Data(), elem->GetClassPointer()))
         objid = -1;  // this is a case, when no data was stored for this object

      TString colname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         colname += sqlio::ObjectSuffix;
      columns->Add(new TSQLColumnData(colname.Data(), objid));
      return kTRUE;
   }

   if (columntyp==kColObjectPtr) {

      if (NumChilds()!=1) return kFALSE;
      TSQLStructure* child = GetChild(0);
      if ((child->GetType()!=kSqlPointer) && (child->GetType()!=kSqlObject)) return kFALSE;

      Bool_t normal = kTRUE;

      if (child->GetType()==kSqlObject)
         normal = child->StoreObject(reg, child->GetValue(), child->GetObjectClass());

      if (!normal) return kFALSE;

      TString colname = elem->GetName();
      if (reg->f->GetUseSuffixes())
         colname += sqlio::PointerSuffix;

      columns->Add(new TSQLColumnData(colname.Data(), "INT", child->GetValue(), kTRUE));
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

      TString colname = elem->GetName();
      if (reg->f->GetUseSuffixes()) {
         colname+=":";
         colname+=GetSimpleTypeName(typ);
      }

      columns->Add(new TSQLColumnData(colname.Data(), sqltype, value, IsNumericType(typ)));

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
            TString colname = elem->GetName();
            colname+=MakeArrayIndex(elem, index);
            columns->Add(new TSQLColumnData(colname.Data(), sqltype, value, kTRUE));
            index++;
         }
      }
      return kTRUE;
   }

   return kFALSE;
}

//________________________________________________________________________
Bool_t TSQLStructure::TryConvertObjectArray(TSqlRegistry* reg, TObjArray* blobs)
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
      TSQLStructure* s_ver = GetChild(indx++);
      TSQLStructure* s_info = GetChild(indx++);
      TStreamerInfo* info = s_info->GetStreamerInfo();
      Int_t objid = reg->GetNextObjId();
      TString sobjid;
      sobjid.Form("%d",objid);
      if (!s_info->StoreObject(reg, sobjid.Data(), info->GetClass())) {
         objid = -1;  // this is a case, when no data was stored for this object
         sobjid = "-1";
      }

      AddCmd(blobs, sqlio::ObjectRef_Arr, sobjid.Data(), elem->GetName(), ns);
   }

   return kTRUE;
}

//________________________________________________________________________
Bool_t TSQLStructure::CheckNormalClassPair(TSQLStructure* s_ver, TSQLStructure* s_info)
{
   // check if pair of two element corresponds
   // to start of object, stored in normal form

   if ((s_ver==0) || (s_info==0) || (s_ver->GetType()!=kSqlVersion) ||
       (s_info->GetType()!=kSqlStreamerInfo)) return kFALSE;

   TClass* cl = s_ver->GetVersionClass();
   TStreamerInfo* info = s_info->GetStreamerInfo();
   if ((cl==0) || (cl!=info->GetClass()) || (cl->GetClassVersion()!=info->GetClassVersion())) return kFALSE;

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

   TObjArray columns;

   const char* uinttype = reg->f->SQLCompatibleType(TStreamerInfo::kUInt);

   columns.Add(new TSQLColumnData(reg->f->SQLObjectIdColumn(), reg->f->SQLIntType(), reg->fCurrentObjId, kTRUE));

   columns.Add(new TSQLColumnData(sqlio::TObjectUniqueId, uinttype, str_id->GetValue(), kTRUE));
   columns.Add(new TSQLColumnData(sqlio::TObjectBits, uinttype, str_bits->GetValue(), kTRUE));
   columns.Add(new TSQLColumnData(sqlio::TObjectProcessId, "CHAR(3)", (str_prid ? str_prid->GetValue() : ""), kFALSE));

   reg->f->SyncSQLClassInfo(sqlinfo, &columns, kFALSE);

   reg->InsertToNormalTable(&columns, sqlinfo->GetClassTableName());

   columns.Delete();

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

   TObjArray columns;

   columns.Add(new TSQLColumnData(reg->f->SQLObjectIdColumn(), reg->f->SQLIntType(), reg->fCurrentObjId, kTRUE));
   columns.Add(new TSQLColumnData(sqlio::TStringValue, reg->f->SQLBigTextType(), value, kFALSE));

   reg->f->SyncSQLClassInfo(sqlinfo, &columns, kFALSE);
   reg->InsertToNormalTable(&columns, sqlinfo->GetClassTableName());
   columns.Delete();
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

   if ((typ>0) && (typ<20) &&
       (typ!=TStreamerInfo::kCharStar)) return kColSimple;

   if ((typ>TStreamerInfo::kOffsetL) &&
       (typ<TStreamerInfo::kOffsetP))
      if ((f->GetArrayLimit()<0) ||
          (elem->GetArrayLength()<=f->GetArrayLimit()))
         return kColSimpleArray;

   if (typ==TStreamerInfo::kTObject)
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;
      else
         return kColObject;

   if (typ==TStreamerInfo::kTNamed)
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;
      else
         return kColObject;

   if (typ==TStreamerInfo::kTString) return kColTString;

   if (typ==TStreamerInfo::kBase) return kColParent;

   if (typ==TStreamerInfo::kSTL)
      if (elem->InheritsFrom(TStreamerBase::Class()))
         return kColParent;

   if ((typ==TStreamerInfo::kObject) ||
       (typ==TStreamerInfo::kAny) ||
       (typ==TStreamerInfo::kAnyp) ||
       (typ==TStreamerInfo::kObjectp) ||
       (typ==TStreamerInfo::kSTL))
      if (elem->GetArrayLength()==0) return kColObject; else
         if (elem->GetStreamer()==0) return kColObjectArray;

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
Int_t TSQLStructure::LocateElementColumn(TSQLFile* f, TSQLObjectData* data)
{
   // find column in TSQLObjectData object, which correspond to current element

   // for TClonesArray do nothing, just exit
   if (IsClonesArray()) return kColClonesArray;

   TStreamerElement* elem = GetElement();
   if ((elem==0) || (data==0)) return kColUnknown;

   Int_t coltype = DefineElementColumnType(elem, f);

   if (gDebug>4)
      cout <<"TSQLStructure::LocateElementColumn " << elem->GetName() <<
         " coltyp = " << coltype << " : " << elem->GetType() << " len = " << elem->GetArrayLength() << endl;

   if (coltype==kColUnknown) return kColUnknown;

   const char* elemname = elem->GetName();
   Bool_t located = kFALSE;

   switch (coltype) {
   case kColSimple: {
      TString colname = elemname;
      if (f->GetUseSuffixes()) {
         colname+=f->SQLNameSeparator();
         colname+=GetSimpleTypeName(elem->GetType());
      }
      located = data->LocateColumn(colname.Data());
      break;
   }

   case kColSimpleArray: {
      TString colname = elemname;
      colname+=MakeArrayIndex(elem, 0);
      located = data->LocateColumn(colname);
      break;
   }

   case kColParent: {
      TString colname = elemname;
      if (f->GetUseSuffixes())
         colname+=sqlio::ParentSuffix;

      located = data->LocateColumn(colname.Data());
      if (located==kColUnknown) return kColUnknown;

      Int_t objid = DefineObjectId();
      const char* clname = elemname;
      Int_t version = atoi(data->GetValue());

      // this is a case, when parent store nothing in the database
      if (version<0) break;

      // special treatment for TObject
      if (strcmp(clname,TObject::Class()->GetName())==0) {
         UnpackTObject(f, data, objid, version);
         break;
      }

      TSQLClassInfo* sqlinfo = f->RequestSQLClassInfo(clname, version);
      if (sqlinfo==0) return kColUnknown;

      // this will indicate that streamer is completely custom
      if (sqlinfo->IsClassTableExist()) {
         data->AddUnpackInt(sqlio::Version, version);
      } else {
         TSQLObjectData* objdata = f->GetObjectClassData(objid, sqlinfo);
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
      TString colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::ObjectSuffix;

      located = data->LocateColumn(colname.Data());
      if (located==kColUnknown) return located;

      const char* strobjid = data->GetValue();
      if (strobjid==0) return kColUnknown;

      Int_t objid = atoi(strobjid);

      // when nothing was stored, nothing need to be read. skip
      if (objid<0) break;

      TString clname;
      Version_t version;

      if (!f->GetObjectData(objid, clname, version)) return kColUnknown;

      // special treatment for TObject
      if (clname==TObject::Class()->GetName()) {
         UnpackTObject(f, data, objid, version);
         break;
      }

      TSQLClassInfo* sqlinfo = f->RequestSQLClassInfo(clname.Data(), version);
      if (sqlinfo==0) return kColUnknown;

      if (sqlinfo->IsClassTableExist()) {
         data->AddUnpackInt(sqlio::Version, version);
      } else {
         TSQLObjectData* objdata = f->GetObjectClassData(objid, sqlinfo);
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
      TString colname = elemname;
      if (f->GetUseSuffixes())
         colname += sqlio::PointerSuffix;

      located = data->LocateColumn(colname.Data());
      break;
   }

   case kColTString: {
      TString colname = elem->GetName();
      if (f->GetUseSuffixes())
         colname+=sqlio::StrSuffix;

      located = data->LocateColumn(colname);
      if (located==kColUnknown) return located;
      const char* value = data->GetValue();

      Int_t objid = DefineObjectId();
      Int_t strid = f->IsLongStringCode(value, objid);

      TString buf;

      // if special prefix found, than try get such string
      if (strid>0)
         if (f->GetLongString(objid, strid, buf))
            value = buf.Data();

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
      TString blobname = elemname;
      if (f->GetUseSuffixes())
         blobname += sqlio::RawSuffix;
      located = data->LocateColumn(blobname.Data(), kTRUE);
      break;
   }

   case kColObjectArray: {
      TString blobname = elemname;
      if (f->GetUseSuffixes())
         blobname += sqlio::RawSuffix;
      located = data->LocateColumn(blobname.Data(), kTRUE);

      break;
   }
   }

   if (!located) coltype = kColUnknown;

   return coltype;
}

//________________________________________________________________________
Bool_t TSQLStructure::UnpackTObject(TSQLFile* f, TSQLObjectData* data, Int_t objid, Int_t clversion)
{
   // Unpack TObject data in form, understodable by custom TObject streamer

   TSQLClassInfo* sqlinfo = f->RequestSQLClassInfo(TObject::Class()->GetName(), clversion);
   if (sqlinfo==0) return kFALSE;

   TSQLObjectData* tobjdata = f->GetObjectClassData(objid, sqlinfo);
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
Bool_t TSQLStructure::UnpackTString(TSQLFile* f, TSQLObjectData* data, Int_t objid, Int_t clversion)
{
   // Unpack TString data in form, understodable by custom TString streamer

   TSQLClassInfo* sqlinfo = f->RequestSQLClassInfo(TString::Class()->GetName(), clversion);
   if (sqlinfo==0) return kFALSE;

   TSQLObjectData* tstringdata = f->GetObjectClassData(objid, sqlinfo);
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
