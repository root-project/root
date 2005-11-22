// @(#)root/net:$Name:  $:$Id: TSQLFile.cxx,v 1.2 2005/11/22 11:30:00 brun Exp $
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
// The main motivation for the TSQLFile development is to have
// "transparent" access to SQL data base via standard TFile interface.
//
// The main approach that each class (but not each object) has one or two tables
// with names like $(CLASSNAME)_ver$(VERSION) and $(CLASSNAME)_streamer_ver$(VERSION)
// For example: TAxis_ver8 or TList_streamer_ver5
// Second kind of tables appears, when some of class members can not be converted to
// normalized form or when class has custom streamer.
// For instance, for TH1 class two tables are required: TH1_ver4 and TH1_streamer_ver4
// Most of memebers are stored in TH1_ver4 table columnwise, and only memeber:
//
//  Double_t*  fBuffer;  //[fBufferSize]
//
// can not be represented as column while size of array is not known apriory.
// Therefore, fBuffer will be written as list of values in TH1_streamer_ver4 table.
//
// All objects, stored in the DB, will be registered in table "ObjectsTable".
// In this there are following columns:
//     "key:id"  - key identifier to which belong object
//     "obj:id"  - object identifier
//     "Class"   - object class name
//     "Version" - object class version
//  Data in each "ObjectsTable" row uniqly identify, in which table
//  and which column object is stored.
//
// In normal situation all class data should be sorted columnwise.
// Up to now following member are supported:
// 1) Basic data types
//     Here is everything clear. Column SQL type will be as much as possible
//     close to the original type of value.
// 2) Fixed array of basic data types
//     In this case n columns like fArr[0], fArr[1] and so on will be created.
//     If there is multidimensional array, names will be fArr2[1][2][1] and so on
// 3) Parent class
//     In this case version of parent class is stored and
//     data of parent class will be stored with the same obj:id in corrspondent table.
//     There is a special case, when parent store nothing (this is for instance TQObject).
//     In that case just -1 is written to avoid any extra checks if table exist or not.
// 4) Object as data member.
//     In that case object is saved in normal way to data base and column
//     will contain id of this object.
// 5) Pointer on object
//     Same as before. In case if object was already stored, just its id
//     will be placed in the column. For NULL pointer 0 is used.
// 6) TString
//     Now column with limited width like VARCAHR(255) in MySQL is used.
//     Later this will be improved to support maximum possible strings
// 7) Anything else.
//     Data will be converted to raw format and saved in _streamer_ table.
//     Each row supplied with obj:id and row:id, where row:id indicates
//     data, corresponding to this particular data member, and column
//     will contain this raw:id
//
// Optionally (default this options on) name of column includes
// suffix which indicates type of column. For instance:
//   *:parent  - parent class, column contain class version
//   *:object  - other object, column contain object id
//   *:rawdata - raw data, column contains id of raw data from _streamer_ table
//   *:Int_t   - column with integer value
// Use TSQLFile::SetUseSuffixes(kFALSE) to disable with behaviour.
// Option can only be changed before first object was written to file.
// Normally this should be done immidiately after createion of TSQLFile instance.
//
// All conversion to SQL statements are done with help of TSQLStructure class.
// This is special hierarchical structure wich internally is very similar
// to XML structures. TBufferSQL2 creates these structures, when object
// data is streamed by ROOT and only afterwards all SQL statements will be produced
// and applied all together.
//
// When data is reading, TBufferSQL2 will produce requests to database
// during unstreaming of object data.
//
// There is two examples: tables.C and canvas.C in sql directory
// To run them, correct database name, username and password should be specified.
// Be carefull. First, what TSQLFile does when "recreate" or "create" option
// are specified, is delete all tables (including non-ROOT) from data base.
//
// In first example histogram and list of TBox classes
// are written and then read back.
//
// Second produces canvas (like in ntuple1.C tutorial),
// store its to DB and reads back. There is also benchmark.
// On MySQL 4.1, running on remote PC, store taking ~4s and read ~2s.
//
// Up to now MySQL 4.1 and Oracle 9i were tested. Definitely, some
// adjustments required for other SQL databases. Hopefully, this should
// be straigthforward.
//
// Known problems and open questions.
// 1) TTree is not supported by TSQLFile. There is independent development
//    of TTreeSQL, which allows to store trees directly in SQL database
// 2) TClonesArray is not tested, will be adjusted soon.
// 3) TDirectory cannot work. Hopefully, will (changes in ROOT basic I/O is required)
// 4) Streamer infos are not written to file, therefore schema evolution
//    is not yet supported. All eforts are done to enable this feature in
//    the near future
//
// Example how TSQLFile can be used:
//
// example of a session saving data to a SQL data base
// =====================================================
//
//  const char* dbname = "mysql://host.domain:3306/dbname";
//  const char* username = "username";
//  const char* userpass = "userpass";
//
//  // Clean data base and create primary tables
//  TSQLFile* f = new TSQLFile(dbname, "recreate", username, userpass);
//  // Write with standard I/O functions
//  arr->Write("arr",TObject::kSingleKey);
//  h1->Write("histo");
//  // Close connection to DB
//  delete f;
//
// example of a session read data from SQL data base
// =====================================================
//
//  // Open database again in read-only mode
//  TSQLFile* f = new TSQLFile(dbname, "open", username, userpass);
//  // Show list of keys
//  f->ls();
//  // Read stored object, again standard ROOT I/O
//  TH1* h1 = (TH1*) f->Get("histo");
//  if (h1!=0) { h1->SetDirectory(0); h1->Draw(); }
//  TObject* obj = f->Get("arr");
//  if (obj!=0) obj->Print("*");
//  // close connection to DB
//  delete f;
//
// The "SQL I/O" package is currently under development.
// Documentaion is not complete, not all data combination are tested.
// Any bug reports and suggestions are welcome.
// Author: S.Linev, GSI Darmstadt,   S.Linev@gsi.de
//
//______________________________________________________________________________

#include "TSQLFile.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TList.h"
#include "TBrowser.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TArrayC.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TError.h"

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TBufferSQL2.h"
#include "TSQLStructure.h"
#include "TKeySQL.h"
#include "TSQLClassInfo.h"
#include "TSQLObjectData.h"

#include "Riostream.h"

ClassImp(TSQLFile);

const char* mysql_BasicTypes[20] = {
"VARCHAR(255)",     // kBase     =  0,  used for text
"TINYINT UNSIGNED", // kChar     =  1,
"SMALLINT",         // kShort   =  2,
"INT",              // kInt     =  3,
"BIGINT",           // kLong    =  4,
"FLOAT",            // kFloat    = 5,
"INT",              // kCounter =  6,
"VARCHAR(255)",     // kCharStar = 7,
"DOUBLE",           // kDouble   = 8,
"DOUBLE",           // kDouble32=  9,
"",                 // nothing
"TINYINT UNSIGNED", // kUChar    = 11,
"SMALLINT UNSIGNED",// kUShort  = 12,
"INT UNSIGNED",     // kUInt    = 13,
"BIGINT UNSIGNED",  // kULong   = 14,
"INT UNSIGNED",     // kBits     = 15,
"BIGINT",           // kLong64   = 16,
"BIGINT UNSIGNED",  // kULong64 = 17,
"BOOL",             // kBool    = 18,
""
};

const char* mysql_OtherTypes[12] = {
"VARCHAR(255)",     // smallest text
"255",              // maximum length of small text
"TEXT",             // biggest size text
"DATETIME",         // date & time
"`",                // quote for identifier like table name or column name
"dir:id",           // dir id column
"key:id",           // key id column
"obj:id",           // object id column
"raw:id",           // raw data id column
"str:id",           // string id column
":",                // name separator between name and type like TObject:Parent
"\""                // quote for string values in MySQL
};

const char* oracle_BasicTypes[20] = {
"VARCHAR(255)",     // kBase     =  0,  used for text
"INT",              // kChar     =  1,
"INT",              // kShort   =  2,
"INT",              // kInt     =  3,
"INT",              // kLong    =  4,
"FLOAT",            // kFloat    = 5,
"INT",              // kCounter =  6,
"VARCHAR(255)",     // kCharStar = 7,
"DOUBLE PRECISION", // kDouble   = 8,
"DOUBLE PRECISION", // kDouble32=  9,
"",                 // nothing
"INT",              // kUChar    = 11,
"INT",              // kUShort  = 12,
"INT",              // kUInt    = 13,
"INT",              // kULong   = 14,
"INT",              // kBits     = 15,
"INT",              // kLong64   = 16,
"INT",              // kULong64 = 17,
"INT",              // kBool    = 18,
""
};

const char* oracle_OtherTypes[12] = {
"VARCHAR(1000)",    // smallest text
"4095",             // maximum size of smallest text
"VARCHAR(4000)",    // biggest size text, CLOB is not yet supported by TOracleRow
"VARCHAR(50)",      // date & time
"\"",               // quote for identifier like table name or column name
"dir:id",           // dir id column
"key:id",           // key id column
"obj:id",           // object id column
"raw:id",           // raw data id column
"str:id",           // string id column
":",                // name separator between name and type like TObject:parent
"'"                 // quote for string values in Oracle
};

// ******************************************************************

// these are two class to produce deep copy of sql result tables
// introduced to overcome Oracle problem

class TSQLRowCopy : public TSQLRow {

protected:
   TObjArray fFields;

public:
   TSQLRowCopy(TSQLRow* res, Int_t nfields) :
      TSQLRow(),
      fFields()
   {
      for(Int_t n=0;n<nfields;n++) {
         const char* value = res->GetField(n);
         fFields.Add(new TObjString(value));
      }
   }

   virtual ~TSQLRowCopy()
   {
   }

   virtual void        Close(Option_t *option="")
   {
      fFields.Delete(option);
   }

   virtual ULong_t     GetFieldLength(Int_t field)
   {
      const char * value = GetField(field);
      if ((value==0) || (*value==0)) return 0;
      return strlen(value);
   }

   virtual const char *GetField(Int_t field)
   {
      if ((field<0) || (field>fFields.GetLast())) return 0;
      return fFields[field]->GetName();
   }
};

class TSQLResultCopy : public TSQLResult {
protected:
   TObjArray    fRows;
   TObjArray    fNames;
   Int_t        fCounter;

public:
   TSQLResultCopy(TSQLResult* res) :
      TSQLResult(),
      fRows(),
      fNames(),
      fCounter(0)
   {
      Int_t nfields = res->GetFieldCount();

      for (Int_t n=0;n<nfields;n++) {
         const char* name = res->GetFieldName(n);
         fNames.Add(new TObjString(name));
      }

      fRowCount = 0;
      TSQLRow* row = res->Next();
      while (row!=0) {
         fRowCount++;
         fRows.Add(new TSQLRowCopy(row, nfields));
         delete row;
         row = res->Next();
      }

      delete res;
   }

   virtual ~TSQLResultCopy()
   {
   }

   virtual void   Close(Option_t* option="")
   {
      fRows.Delete(option);
      fNames.Delete(option);
   }

   virtual Int_t       GetFieldCount()
   {
      return fNames.GetLast() + 1;
   }

   virtual const char *GetFieldName(Int_t field)
   {
      if ((field<0) || (field>fNames.GetLast())) return 0;
      return fNames[field]->GetName();
   }

   virtual TSQLRow* Next()
   {
      if (fCounter>fRows.GetLast()) return 0;
      TSQLRow* curr = (TSQLRow*) fRows.At(fCounter++);
      fRows.Remove(curr); // not make a copy, just remove from buffer, cannot use again
      return curr;
   }
};

//______________________________________________________________________________
TSQLFile::TSQLFile() :
   TFile(),
   fSQL(0),
   fSQLClassInfos(0),
   fUseSuffixes(kTRUE),
   fSQLIOversion(1),
   fArrayLimit(20),
   fCanChangeConfig(kFALSE),
   fBasicTypes(0),
   fUserName(),
   fLogFile(0)
{
   // default TSQLFile constructor
}


//______________________________________________________________________________
TSQLFile::TSQLFile(const char* dbname, Option_t* option, const char* user, const char* pass) :
   TFile(),
   fSQL(0),
   fSQLClassInfos(0),
   fUseSuffixes(kTRUE),
   fSQLIOversion(1),
   fArrayLimit(20),
   fCanChangeConfig(kFALSE),
   fBasicTypes(mysql_BasicTypes),
   fOtherTypes(mysql_OtherTypes),
   fUserName(user),
   fLogFile(0)
{
   // Connects to SQL server with provided arguments.
   // If the constructor fails in any way IsZombie() will
   // return true. Use IsOpen() to check if the file is (still) open.
   //
   // If option = NEW or CREATE   create a ROOT tables in database
   //                             if the tables already exists connection is
   //                             not opened.
   //           = RECREATE        create completely new tables. Any existing tables
   //                             will be deleted
   //           = UPDATE          open an existing database for writing.
   //                             if no tables existing, they will be created.
   //           = READ or OPEN    open an existing data base for reading.
   //
   // For more details see comments for TFile::TFile() constructor
   //
   // For a moment TSQLFile does not support TTree objects and subdirectories

   if (!gROOT)
      ::Fatal("TFile::TFile", "ROOT system not initialized");

   gDirectory = 0;
   SetName(dbname);
   SetTitle("TFile interface to SQL DB");
   TDirectory::Build();
   fFile = this;

   if (strstr(dbname,"oracle://")!=0) {
      fBasicTypes = oracle_BasicTypes;
      fOtherTypes = oracle_OtherTypes;
   }

   fD          = -1;
   fFile       = this;
   fFree       = 0;
   fVersion    = gROOT->GetVersionInt();  //ROOT version in integer format
   fUnits      = 4;
   fOption     = option;
   SetCompressionLevel(5);
   fWritten    = 0;
   fSumBuffer  = 0;
   fSum2Buffer = 0;
   fBytesRead  = 0;
   fBytesWrite = 0;
   fClassIndex = 0;
   fSeekInfo   = 0;
   fNbytesInfo = 0;
   fCache      = 0;
   fProcessIDs = 0;
   fNProcessIDs= 0;

   fOption = option;
   fOption.ToUpper();

   if (fOption == "NEW") fOption = "CREATE";

   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;

   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   if (!dbname || !strlen(dbname)) {
      Error("TSQLFile", "Database not specified");
      goto zombie;
   }

   gROOT->cd();

   fSQL = TSQLServer::Connect(dbname, user, pass);
   if (fSQL==0) {
      Error("TSQLFile", "Cannot connect to DB %s", dbname);
      goto zombie;
   }

   if (recreate) {
      SQLDeleteAllTables();
      recreate = kFALSE;
      create   = kTRUE;
      fOption  = "CREATE";
   }

   if (create && IsTablesExists()) {
      Error("TSQLFile", "DB tables already exists");
      goto zombie;
   }

   if (update) {
      if (!IsTablesExists()) {
         update = kFALSE;
         create = kTRUE;
      }

      if (update && !IsWriteAccess()) {
         Error("TSQLFile", "no write permission, could not open DB %s", dbname);
         goto zombie;
      }
   }

   if (read) {
      if (!IsTablesExists()) {
         Error("TSQLFile", "DB %s tables not exist", dbname);
         goto zombie;
      }
      if (!IsReadAccess()) {
         Error("TSQLFile", "no read permission, could not open DB %s tables", dbname);
         goto zombie;
      }
   }

   fRealName = dbname;

   if (create || update)
      SetWritable(kTRUE);
   else
      SetWritable(kFALSE);

   // user can change configurations only when create (recreate) options
   // was specified. When first object will be saved, configurations will
   // be frozen.
   fCanChangeConfig = create;

   InitSqlDatabase(create);

   return;

 zombie:
   delete fSQL;
   fSQL = 0;
   MakeZombie();
   gDirectory = gROOT;
}

//______________________________________________________________________________
void TSQLFile::StartLogFile(const char* fname)
{
   // start logging of all SQL statements in specified file

   StopLogFile();
   fLogFile = new std::ofstream(fname);
}

//______________________________________________________________________________
void TSQLFile::StopLogFile()
{
   // close logging file
   if (fLogFile!=0) {
      delete fLogFile;
      fLogFile = 0;
   }
}

//______________________________________________________________________________
Bool_t TSQLFile::IsMySQL() const
{
   // checks, if MySQL database
   if (fSQL==0) return kFALSE;
   return strcmp(fSQL->ClassName(),"TMySQLServer")==0;
}

//______________________________________________________________________________
Bool_t TSQLFile::IsOracle() const
{
   // checks, if Oracle database

   if (fSQL==0) return kFALSE;
   return strcmp(fSQL->ClassName(),"TOracleServer")==0;
}

//______________________________________________________________________________
void TSQLFile::SetUseSuffixes(Bool_t on)
{
   // enable/disable uasge of suffixes in columns names
   // can be changed before first object is saved into file

   if (!fCanChangeConfig)
      Error("SetUseSuffixes", "Configurations already cannot be changed");
   else
      fUseSuffixes = on;
}

//______________________________________________________________________________
void TSQLFile::SetArrayLimit(Int_t limit)
{
   // Defines maximum number of columns for array representation
   // If array size bigger than limit, array data will be converted to raw format
   // This is usefull to prevent tables with very big number of columns
   // If limit==0, all arrays will be stored in raw format
   // If limit<0, all array values will be stored in column form
   // Default value is 20

   if (!fCanChangeConfig)
      Error("SetArrayLimit", "Configurations already cannot be changed");
   else
      fArrayLimit = limit;
}

//______________________________________________________________________________
const char* TSQLFile::GetDataBaseName() const
{
   // Return name of data base on the host
   // For Oracle always return 0

   if (IsOracle()) return 0;
   const char* name = strrchr(GetName(),'/');
   if (name==0) return 0;
   return name + 1;
}

//______________________________________________________________________________
void TSQLFile::Close(Option_t *option)
{
   // Close a SQL file
   // For more comments see TFile::Close() function

   if (!IsOpen()) return;

   TString opt = option;
   if (opt.Length()>0)
      opt.ToLower();

   if (IsWritable()) SaveToDatabase();
   fWritable = kFALSE;

   if (fClassIndex) {
      delete fClassIndex;
      fClassIndex = 0;
   }

   TDirectory *cursav = gDirectory;
   cd();

   if (cursav == this || cursav->GetFile() == this) {
      cursav = 0;
   }

   // Delete all supported directories structures from memory
   TDirectory::Close();
   cd();      // Close() sets gFile = 0

   if (cursav)
      cursav->cd();
   else {
      gFile      = 0;
      gDirectory = gROOT;
   }

   //delete the TProcessIDs
   TList pidDeleted;
   TIter next(fProcessIDs);
   TProcessID *pid;
   while ((pid = (TProcessID*)next())) {
      if (!pid->DecrementCount()) {
         if (pid != TProcessID::GetSessionProcessID()) pidDeleted.Add(pid);
      } else if(opt.Contains("r")) {
         pid->Clear();
      }
   }
   pidDeleted.Delete();

   gROOT->GetListOfFiles()->Remove(this);
}

//______________________________________________________________________________
TSQLFile::~TSQLFile()
{
   // destructor of TSQLFile object

   Close();

   if (fSQLClassInfos!=0) {
      fSQLClassInfos->Delete();
      delete fSQLClassInfos;
   }

   StopLogFile();

   if (fSQL!=0) {
      delete fSQL;
      fSQL = 0;
   }
}

//______________________________________________________________________________
void TSQLFile::operator=(const TSQLFile &)
{
   // make private to exclude copy operator
}

//______________________________________________________________________________
Bool_t TSQLFile::IsOpen() const
{
   // return kTRUE if file is opened and can be accessed

   return fSQL != 0;
}

//______________________________________________________________________________
Int_t TSQLFile::ReOpen(Option_t* mode)
{
   // Reopen a file with a different access mode, like from READ to
   // See TFile::Open() for details

   cd();

   TString opt = mode;
   opt.ToUpper();

   if (opt != "READ" && opt != "UPDATE") {
      Error("ReOpen", "mode must be either READ or UPDATE, not %s", opt.Data());
      return 1;
   }

   if (opt == fOption || (opt == "UPDATE" && fOption == "CREATE"))
      return 1;

   if (opt == "READ") {
      // switch to READ mode

      if (IsOpen() && IsWritable())
         SaveToDatabase();
      fOption = opt;

      SetWritable(kFALSE);

   } else {
      fOption = opt;

      SetWritable(kTRUE);
   }

   return 0;
}

//______________________________________________________________________________
TKey* TSQLFile::CreateKey(const TObject* obj, const char* name, Int_t )
{
   // create SQL key, which will store object in data base
   return new TKeySQL(this, obj, name);
}

//______________________________________________________________________________
TKey* TSQLFile::CreateKey(const void* obj, const TClass* cl, const char* name, Int_t )
{
   // create SQL key, which will store object in data base
   return new TKeySQL(this, obj, cl, name);
}

//______________________________________________________________________________
void TSQLFile::WriteStreamerInfo()
{
   // Store all TStreamerInfo, used in file, in sql database
   // For the moment function is disabled while no proper reader is
   // existing

   return;

   // do not write anything when no basic tables was created
   if (!IsTablesExists()) return;

   Int_t keyid = sqlio::Ids_StreamerInfos;

   if (gDebug>1)
      Info("WriteStreamerInfo","Saving streamer infos to database");

   TList list;

   TIter iter(gROOT->GetListOfStreamerInfo());

   TStreamerInfo* info = 0;

   while ((info = (TStreamerInfo*) iter()) !=0 ) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid]) {
         if (gDebug>1) Info("WriteStreamerInfo","Add %s",info->GetName());
         list.Add(info);
      }
   }
   if (list.GetSize()==0) return;
   fClassIndex->fArray[0] = 2; //to prevent adding classes in TStreamerInfo::TagFile

   DeleteKeyFromDB(keyid);

   Int_t objid = VerifyObjectTable();
   if (objid<=0) objid = 1;
   else objid++;

   TBufferSQL2 buffer(TBuffer::kWrite, this);

   TSQLStructure* s = buffer.SqlWrite(&list, list.IsA(), objid);
   if (gDebug>4)
      s->Print("*");

   TObjArray cmds;
   TDatime now;
   if (s->ConvertToTables(this, keyid, &cmds))
      if (SQLApplyCommands(&cmds))
         WriteKeyData(keyid, sqlio::Ids_RootDir, objid, "StreamerInfo", now.AsSQLString(), 1, list.IsA()->GetName());

   cmds.Delete();

   fClassIndex->fArray[0] = 0; //to prevent adding classes in TStreamerInfo::TagFile

}

//______________________________________________________________________________
TList* TSQLFile::GetStreamerInfoList()
{
   // Read back streamer infos from database
   // List of streamer infos is always stored with key:id 0,
   // which is not shown in normal keys list
   // Method is not active while TStreamerElement and TStreamerBase has custom
   // streamers, which can not be handled by TBufferSQL2.
   // Hopefully, problem will be solved soon

   return 0;

   if (gDebug>1)
      Info("GetStreamerInfoList","Start reading of streamer infos");

   Int_t objid = -1;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();

   sqlcmd.Form("SELECT %s%s%s FROM %s%s%s WHERE %s%s%s=%d",
               quote, SQLObjectIdColumn(), quote,
               quote, sqlio::KeysTable, quote,
               quote, SQLKeyIdColumn(), quote, sqlio::Ids_StreamerInfos);
   TSQLResult* res = SQLQuery(sqlcmd.Data(), 1);
   TSQLRow* row = (res==0) ? 0 : res->Next();
   if (row!=0) objid = atoi((*row)[0]);
   delete res;
   delete row;

   if (objid<=0) return 0;

   TBufferSQL2 buffer(TBuffer::kRead, this);

   buffer.SetIgnoreVerification();

   TObject* obj = buffer.SqlRead(objid);

   TList* list = dynamic_cast<TList*> (obj);
   if (list==0) delete obj;
   else list->Print("*");

   return list;
}

//______________________________________________________________________________
void TSQLFile::SaveToDatabase()
{
   // save data which is not yet in Database
   // Typically this is streamerinfos structures

   if (fSQL==0) return;
   WriteStreamerInfo();
}

//______________________________________________________________________________
Bool_t TSQLFile::ReadKeysForDirectory(TDirectory* dir, Int_t dir_id)
{
   // read keys for specified direccctory

   if ((dir==0) || (dir_id<0)) return kFALSE;

   const char* quote = SQLIdentifierQuote();

   TString sqlcmd;
   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s=%d",
               quote, sqlio::KeysTable, quote,
               quote, SQLDirIdColumn(), quote, dir_id);
   TSQLResult* res = SQLQuery(sqlcmd.Data(), 2);

   if (res==0) return kFALSE;

   for(Int_t nrow=0;nrow<res->GetRowCount();nrow++) {
      TSQLRow* row = res->Next();

      Int_t keyid = atoi((*row)[0]);
      //      Int_t dirid = atoi((*row)[1]);
      Int_t objid = atoi((*row)[2]);

      Int_t cycle = atoi((*row)[5]);

      if (keyid!=sqlio::Ids_StreamerInfos) {
         TKeySQL* key = new TKeySQL(this, keyid, dir_id, objid, (*row)[3], (*row)[4], cycle, (*row)[6]);
         dir->AppendKey(key);
      }

      delete row;
   }

   return kTRUE;
}

//______________________________________________________________________________
void TSQLFile::InitSqlDatabase(Bool_t create)
{
   // initialize sql database and correspondent structures
   // identical to TFile::Init() function

   Int_t len = gROOT->GetListOfStreamerInfo()->GetSize()+1;
   if (len<5000) len = 5000;
   fClassIndex = new TArrayC(len);
   fClassIndex->Reset(0);

   if (!create) {
      Bool_t ok = ReadConfigurations();

      if (ok) {
         ReadStreamerInfo();
         ok = ReadKeysForDirectory(this, sqlio::Ids_RootDir);
      }

      if (!ok) {
         Error("InitSqlDatabase", "Cannot detect proper tabled in database. Close.");
         Close();
         delete fSQL;
         fSQL = 0;
         MakeZombie();
         gDirectory = gROOT;
         return;
      }
   }

   gROOT->GetListOfFiles()->Add(this);
   cd();

   fNProcessIDs = 0;
   TKey* key = 0;
   TIter iter(fKeys);
   while ((key = (TKey*)iter())!=0) {
      if (!strcmp(key->GetClassName(),"TProcessID")) fNProcessIDs++;
   }

   fProcessIDs = new TObjArray(fNProcessIDs+1);
}

//______________________________________________________________________________
Bool_t TSQLFile::ReadConfigurations()
{
   // read table configurations as special table

   const char* quote = SQLIdentifierQuote();

   TString sqlcmd;
   sqlcmd.Form("SELECT * FROM %s%s%s",
               quote, sqlio::ConfigTable, quote);
   TSQLResult* res = SQLQuery(sqlcmd.Data(), 2);

   if (res==0) return kFALSE;

   // should be found, otherwise will be error
   fSQLIOversion = 0;

   for(Int_t nrow=0;nrow<res->GetRowCount();nrow++) {
      TSQLRow* row = res->Next();

      TString field = row->GetField(0);
      TString value = row->GetField(1);

      delete row;

      if (field.CompareTo(sqlio::cfg_Version,TString::kIgnoreCase)==0)
         fSQLIOversion = value.Atoi();
      else
         if (field.CompareTo(sqlio::cfg_UseSufixes,TString::kIgnoreCase)==0)
            fUseSuffixes = value.CompareTo(sqlio::True, TString::kIgnoreCase)==0;
         else
            if (field.CompareTo(sqlio::cfg_ArrayLimit,TString::kIgnoreCase)==0)
               fArrayLimit = value.Atoi();
            else {
               Error("ReadConfigurations","Invalid configuration field %s", field.Data());
               fSQLIOversion = 0;
               break;
            }
   }

   delete res;

   return (fSQLIOversion>0);
}

//______________________________________________________________________________
void TSQLFile::CreateBasicTables()
{
   TString sqlcmd;

   const char* quote = SQLIdentifierQuote();
   const char* vquote = SQLValueQuote();

   if (SQLTestTable(sqlio::ConfigTable)) {
      sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlio::ConfigTable, quote);
      SQLQuery(sqlcmd.Data());
   }

   sqlcmd.Form("CREATE TABLE %s%s%s (%s %s, %s %s)",
               quote, sqlio::ConfigTable, quote,
               sqlio::CT_Field, SQLSmallTextType(),
               sqlio::CT_Value, SQLSmallTextType());

   SQLQuery(sqlcmd.Data());

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s%s%s, %s%d%s)",
               quote, sqlio::ConfigTable, quote,
               vquote, sqlio::cfg_Version, vquote,
               vquote, fSQLIOversion, vquote);
   SQLQuery(sqlcmd.Data());

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s%s%s, %s%s%s)",
               quote, sqlio::ConfigTable, quote,
               vquote, sqlio::cfg_UseSufixes, vquote,
               vquote, fUseSuffixes ? sqlio::True : sqlio::False, vquote);
   SQLQuery(sqlcmd.Data());

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s%s%s, %s%d%s)",
               quote, sqlio::ConfigTable, quote,
               vquote, sqlio::cfg_ArrayLimit, vquote,
               vquote, fArrayLimit, vquote);
   SQLQuery(sqlcmd.Data());

   // from this moment on user cannot change configurations
   fCanChangeConfig = kFALSE;

   if (SQLTestTable(sqlio::KeysTable)) {
      sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlio::KeysTable, quote);
      SQLQuery(sqlcmd.Data());
   }

   sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s%s%s %s, %s %s, %s %s, %s %s, %s %s)",
               quote, sqlio::KeysTable, quote,
               quote, SQLKeyIdColumn(), quote, SQLIntType(),
               quote, SQLDirIdColumn(), quote, SQLIntType(),
               quote, SQLObjectIdColumn(), quote, SQLIntType(),
               sqlio::KT_Name, SQLSmallTextType(),
               sqlio::KT_Datetime, SQLDatetimeType(),
               sqlio::KT_Cycle, SQLIntType(),
               sqlio::KT_Class, SQLSmallTextType());

   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
Bool_t TSQLFile::IsTablesExists()
{
   // Checks if main keys table is existing
   return SQLTestTable(sqlio::KeysTable);
}

//______________________________________________________________________________
Bool_t TSQLFile::IsWriteAccess()
{
   // dummy, in future should check about write access to database
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLFile::IsReadAccess()
{
   // dummy, in future should check about read access to database
   return kTRUE;
}

//______________________________________________________________________________
TSQLResult* TSQLFile::SQLQuery(const char* cmd, Int_t flag)
{
   // submits query to SQL server
   // if flag==0, result is not interesting and will be deleted
   // if flag==1, return result of submitted query
   // if flag==2, results is may be necessary for long time
   //             Oracle plugin do not support working with several TSQLResult
   //             objects, therefore explicit deep copy will be produced

   if (fLogFile!=0)
      *fLogFile << cmd << endl;

   if (fSQL==0) return 0;

   if (gDebug>2) Info("SQLQuery",cmd);

   TSQLResult* res = fSQL->Query(cmd);
   if (res==0) return 0;
   if (flag==0) {
      delete res;
      return 0;
   }
   if ((flag==2) && IsOracle())
      res = new TSQLResultCopy(res);
   return res;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLApplyCommands(TObjArray* cmds)
{
   // supplies set of commands to server
   // Commands is stored as array of TObjString

   if ((cmds==0) || (fSQL==0)) return kFALSE;
   TIter iter(cmds);
   TObject* cmd= 0;
   while ((cmd=iter())!=0)
      SQLQuery(cmd->GetName());
   return kTRUE;
}

//______________________________________________________________________________
TObjArray* TSQLFile::SQLTablesList(const char* searchtable)
{
   // Produces list of tables, presented in database
   // if searchtable!=0, looks only for this specific table
   // list should be deleted by user afterwards
   // P.S. Unfortunately, the TSQLServer::GetTables function is not
   // correctly implemented for all cases,
   // therefore special function is required.

   if (fSQL==0) return 0;

   TObjArray* res = 0;

   if (IsOracle()) {
      TString sqlcmd;
      TString user = fUserName;
      user.ToUpper();
      sqlcmd.Form("SELECT object_name FROM ALL_OBJECTS WHERE object_type='TABLE' and owner='%s'",user.Data());
      if (searchtable!=0) {
         TString table = searchtable;
         table.ToUpper();
         sqlcmd += Form(" and object_name='%s'",table.Data());
      }

      TSQLResult* tables = SQLQuery(sqlcmd.Data(), 1);
      if (tables==0) return 0;

      TSQLRow* row = tables->Next();
      while (row!=0) {
         const char* tablename = row->GetField(0);
         if (strpbrk(tablename,"$=")==0) {
            if (res==0) res = new TObjArray;
            res->Add(new TObjString(tablename));
         }
         delete row;
         row = tables->Next();
      }
      delete tables;
   } else {
      TSQLResult* tables = fSQL->GetTables(GetDataBaseName(), searchtable);
      if (tables==0) return 0;

      TSQLRow* row = tables->Next();
      while (row!=0) {
         if (res==0) res = new TObjArray;
         res->Add(new TObjString(row->GetField(0)));
         delete row;
         row = tables->Next();
      }
      delete tables;
   }
   if (res!=0) res->SetOwner(kTRUE);

   return res;
}

//______________________________________________________________________________
TObjArray* TSQLFile::SQLTableColumns(const char* tablename)
{
   // produces list of columns for specified table
   // list consist of TNamed objects with name and type for each column
   // list should be deleted aftrewards
   // P.S. Oracle plug-in do not provides types

   if (fSQL==0) return 0;

   TObjArray* res = 0;

   if (IsOracle()) {
      TSQLResult* cols = fSQL->GetColumns(0, tablename, "");
      if  (cols==0) return 0;
      for (Int_t n=0;n<cols->GetFieldCount();n++) {
         TNamed* col = new TNamed(cols->GetFieldName(n), "TYPE?");
         if (res==0) res = new TObjArray;
         res->Add(col);
      }
      delete cols;

   } else {
      TSQLResult* cols = fSQL->GetColumns(GetDataBaseName(), tablename, "");

      if (cols==0) return 0;

      TSQLRow* row = cols->Next();
      while (row!=0) {
         TNamed* col = new TNamed(row->GetField(0), row->GetField(1));
         if (res==0) res = new TObjArray;
         res->Add(col);
         delete row;
         row = cols->Next();
      }

      delete cols;
   }
   return res;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLTestTable(const char* tablename)
{
   // Test, if table of specified name exists

   TObjArray* list = SQLTablesList(tablename);

   Bool_t res = (list!=0);

   delete list;

   return res;
}

//______________________________________________________________________________
Int_t TSQLFile::SQLMaximumValue(const char* tablename, const char* columnname)
{
   // Returns maximum value, found in specified columnname of table tablename
   // Column type should be numeric

   if (fSQL==0) return -1;

   if (gDebug>2)
      Info("SQLMaximumValue","Requests for %s column %s", tablename, columnname);

   const char* quote = SQLIdentifierQuote();

   TString query;
   query.Form("SELECT MAX(%s%s%s) FROM %s%s%s",
              quote, columnname, quote,
              quote, tablename, quote);
   TSQLResult* res = SQLQuery(query.Data(), 1);

   if (res==0) return -1;

   TSQLRow* row = res->Next();

   Int_t maxid = -1;
   if (row!=0)
      if (row->GetField(0)!=0)
         maxid = atoi(row->GetField(0));

   delete row;
   delete res;

   if (gDebug>2)
      Info("SQLMaximumValue","Result = %d",maxid);;

   return maxid;
}

//______________________________________________________________________________
void TSQLFile::SQLDeleteAllTables()
{
   // Delete all tables in database

   TObjArray* tables = SQLTablesList();
   if (tables==0) return;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();

   TIter iter(tables);
   TObject* obj = 0;
   while ((obj=iter())!=0) {
      sqlcmd.Form("DROP TABLE %s%s%s", quote, obj->GetName(), quote);
      SQLQuery(sqlcmd.Data());
   }
   delete tables;
}

//______________________________________________________________________________
void TSQLFile::DeleteKeyFromDB(Int_t keyid)
{
   // remove key with specified id from keys table
   // also removes all objects data, releta to this table

   if (!IsWritable() || (keyid<0) || (fSQL==0)) return;

   TString query;
   const char* quote = SQLIdentifierQuote();

   query.Form("SELECT * FROM %s%s%s WHERE %s%s%s=%d",
              quote, sqlio::ObjectsTable, quote,
              quote, SQLKeyIdColumn(), quote,
              keyid);
   TSQLResult* res = SQLQuery(query.Data(), 2);
   if (res==0) return;
   TSQLRow* row = res->Next();
   while (row!=0) {
      Int_t objid = atoi(row->GetField(1));
      DeleteObjectFromTables(objid);
      delete row;
      row = res->Next();
   }

   delete res;

   query.Form("DELETE FROM %s WHERE %s%s%s=%d", sqlio::ObjectsTable, quote, SQLKeyIdColumn(), quote, keyid);
   SQLQuery(query.Data());

   query.Form("DELETE FROM %s WHERE %s%s%s=%d", sqlio::KeysTable, quote, SQLKeyIdColumn(), quote, keyid);
   SQLQuery(query.Data());
}

//______________________________________________________________________________
void TSQLFile::WriteKeyData(Int_t keyid, Int_t dirid, Int_t objid, const char* keyname, const char* datime, Int_t cycle, const char* clname)
{
   // add entry into keys table

   if (fSQL==0) return;

   if (!IsTablesExists()) CreateBasicTables();

   TString sqlcmd;
   const char* valuequote = SQLValueQuote();
   const char* quote = SQLIdentifierQuote();

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%d, %d, %d, %s%s%s, %s%s%s, %d, %s%s%s)",
               quote, sqlio::KeysTable, quote,
               keyid, dirid, objid,
               valuequote, keyname, valuequote,
               valuequote, datime, valuequote,
               cycle,
               valuequote, clname, valuequote);

   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
Int_t TSQLFile::DefineNextKeyId()
{
   // Returns next possible key identifier

   Int_t max = -1;

   if (SQLTestTable(sqlio::KeysTable))
      max = SQLMaximumValue(sqlio::KeysTable, SQLKeyIdColumn());

   if (max<0) return sqlio::Ids_FirstKey;

   return max+1;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::FindSQLClassInfo(const char* clname, Int_t version)
{
   // return (if exists) TSQLClassInfo for specified class and version

   if (fSQLClassInfos==0) return 0;

   TIter iter(fSQLClassInfos);
   TSQLClassInfo* info = 0;

   while ((info = (TSQLClassInfo*) iter()) !=0 ) {
      if (strcmp(info->GetName(), clname)==0)
         if (info->GetClassVarsion()==version) return info;
   }
   return 0;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::RequestSQLClassInfo(const char* clname, Int_t version, Bool_t force)
{
   // search in database tables for specified class and return TSQLClassInfo object

   TSQLClassInfo* info = FindSQLClassInfo(clname, version);
   if (!force && (info!=0)) return info;

   if (fSQL==0) return 0;

   if (info==0) info = new TSQLClassInfo(clname, version);

   TObjArray* columns = 0;

   // first check if class table is exist
   if (SQLTestTable(info->GetClassTableName()))
      columns = SQLTableColumns(info->GetClassTableName());

   Bool_t israwtable = SQLTestTable(info->GetRawTableName());

   info->SetTableStatus(columns, israwtable);

   if (fSQLClassInfos==0) fSQLClassInfos = new TList;
   fSQLClassInfos->Add(info);

   return info;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::RequestSQLClassInfo(const TClass* cl, Bool_t force)
{
   // search in database tables for specified class and return TSQLClassInfo object

   return RequestSQLClassInfo(cl->GetName(), cl->GetClassVersion(), force);
}

//______________________________________________________________________________
Bool_t TSQLFile::SyncSQLClassInfo(TSQLClassInfo* sqlinfo, TObjArray* columns, Bool_t hasrawdata)
{
   // Synchronise TSQLClassInfo structure with specified columns list and
   // create/delete appropriate tables in database

   if (sqlinfo==0) return kFALSE;

   if (gDebug>2)
      Info("SyncSQLClassInfo", sqlinfo->GetName());

   const char* quote = SQLIdentifierQuote();

   if (sqlinfo->IsClassTableExist() && (columns==0)) {
      TString sqlcmd;
      sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlinfo->GetClassTableName(), quote);
      SQLQuery(sqlcmd.Data());

      sqlinfo->SetColumns(0);
   }

   if (!sqlinfo->IsClassTableExist() && (columns!=0)) {

      TString sqlcmd;

      if (IsMySQL()) {
         sqlcmd.Form("DROP TABLE IF EXISTS %s%s%s",
                     quote, sqlinfo->GetClassTableName(), quote);
         SQLQuery(sqlcmd.Data());
      }

      sqlcmd.Form("CREATE TABLE %s%s%s (", quote, sqlinfo->GetClassTableName(), quote);

      TObjArray* newcolumns = new TObjArray();
      TIter iter(columns);
      TSQLColumnData* col;
      Bool_t first = kTRUE;
      Bool_t forcequote = IsOracle();
      while ((col=(TSQLColumnData*)iter())!=0) {
         if (!first) sqlcmd+=", "; else first = false;

         const char* colname = col->GetName();
         if ((strpbrk(colname,"[:]")!=0) || forcequote) {
            sqlcmd += quote;
            sqlcmd += colname;
            sqlcmd += quote;
            sqlcmd += " ";
         } else {
            sqlcmd += colname,
               sqlcmd += " ";
         }

         sqlcmd += col->GetType();

         newcolumns->Add(new TNamed(col->GetName(), col->GetType()));
      }
      sqlcmd += ")";

      SQLQuery(sqlcmd.Data());

      sqlinfo->SetColumns(newcolumns);
   }

   if (hasrawdata && !sqlinfo->IsRawTableExist()) {
      TString sqlcmd;

      if (IsMySQL()) {
         sqlcmd.Form("DROP TABLE IF EXISTS %s%s%s",
                     quote, sqlinfo->GetRawTableName(), quote);
         SQLQuery(sqlcmd.Data());
      }

      sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s %s, %s %s)",
                  quote, sqlinfo->GetRawTableName(), quote,
                  quote, SQLObjectIdColumn(), quote, SQLIntType(),
                  quote, SQLRawIdColumn(), quote, SQLIntType(),
                  sqlio::BT_Field, SQLSmallTextType(),
                  sqlio::BT_Value, SQLSmallTextType());

      SQLQuery(sqlcmd.Data());
      sqlinfo->SetRawExist(kTRUE);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLFile::VerifyLongStringTable()
{
   // Checks that table for big strings is exists
   // If not, will be created

   if (fSQL==0) return kFALSE;

   if (SQLTestTable(sqlio::StringsTable)) return kTRUE;

   const char* quote = SQLIdentifierQuote();

   TString cmd_cr;
   cmd_cr.Form("CREATE TABLE %s (%s%s%s %s, %s%s%s %s, %s %s)",
               sqlio::StringsTable,
               quote, SQLObjectIdColumn(), quote, SQLIntType(),
               quote, SQLStrIdColumn(), quote, SQLIntType(),
               sqlio::ST_Value, SQLBigTextType());

   SQLQuery(cmd_cr.Data());

   return kTRUE;
}

//______________________________________________________________________________
TString TSQLFile::CodeLongString(Int_t objid, Int_t strid)
{
   // produces id which will be placed in column instead of string itself
   TString res;
   res.Form("%s %d %s %d %s", sqlio::LongStrPrefix, objid, sqlio::LongStrPrefix, strid, sqlio::LongStrPrefix);
   return res;
}

//______________________________________________________________________________
Int_t TSQLFile::IsLongStringCode(const char* value, Int_t objid)
{
   // checks if this is long string code
   // returns 0, if not or string id
   if (value==0) return 0;
   if (strlen(value)<strlen(sqlio::LongStrPrefix)*3+6) return 0;
   if (strstr(value, sqlio::LongStrPrefix)!=value) return 0;

   value+=strlen(sqlio::LongStrPrefix);
   if (*value++!=' ') return 0;
   TString s_strid, s_objid;
   if ((*value<'1') || (*value>'9')) return 0;
   do {
      s_objid.Append(*value++);
   } while ((*value!=0) && (*value>='0') && (*value<='9'));

   if (*value++ != ' ') return 0;
   if ((*value==0) || (strstr(value, sqlio::LongStrPrefix)!=value)) return 0;
   value+=strlen(sqlio::LongStrPrefix);
   if (*value++!=' ') return 0;

   if ((*value<'1') || (*value>'9')) return 0;
   do {
      s_strid.Append(*value++);
   } while ((*value!=0) && (*value>='0') && (*value<='9'));
   if (*value++!=' ') return 0;

   if ((*value==0) || (strcmp(value, sqlio::LongStrPrefix)!=0)) return 0;

   if (atoi(s_objid.Data())!=objid) return 0;

   return atoi(s_strid.Data());
}

//______________________________________________________________________________
Bool_t TSQLFile::GetLongString(Int_t objid, Int_t strid, TString& value)
{
   // returns value of string, extracted from special table,
   // where long strings are stored

   if (!SQLTestTable(sqlio::StringsTable)) return kFALSE;

   TString cmd;
   const char* quote = SQLIdentifierQuote();
   cmd.Form("SELECT %s FROM %s%s%s WHERE %s%s%s=%d AND %s%s%s=%d",
            sqlio::ST_Value,
            quote, sqlio::StringsTable, quote,
            quote, SQLObjectIdColumn(), quote, objid,
            quote, SQLStrIdColumn(), quote, strid);

   TSQLResult* res = SQLQuery(cmd.Data(), 1);
   if (res==0) return kFALSE;
   TSQLRow* row = res->Next();
   if (row==0) { delete res; return kFALSE; }
   value = row->GetField(0);

   delete row;
   delete res;

   return kTRUE;
}

//______________________________________________________________________________
Int_t TSQLFile::VerifyObjectTable()
{
   // Checks that objects table is exists
   // If not, table will be created
   // Returns maximum value for existing objects id

   if (fSQL==0) return -1;

   Int_t maxid = -1;

   if (gDebug>2)
      Info("VerifyObjectTable", "Checks if object table is there");

   if (SQLTestTable(sqlio::ObjectsTable))
      maxid = SQLMaximumValue(sqlio::ObjectsTable, SQLObjectIdColumn());
   else {
      TString cmd_cr;
      const char* quote = SQLIdentifierQuote();
      cmd_cr.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s %s, %s %s)",
                  quote, sqlio::ObjectsTable, quote,
                  quote, SQLKeyIdColumn(), quote, SQLIntType(),
                  quote, SQLObjectIdColumn(), quote, SQLIntType(),
                  sqlio::OT_Class, SQLSmallTextType(),
                  sqlio::OT_Version, SQLIntType());
      SQLQuery(cmd_cr.Data());
   }

   return maxid;
}

//______________________________________________________________________________
TString TSQLFile::SetObjectDataCmd(Int_t keyid, Int_t objid, TClass* cl)
{
   // produces SQL query to set object data
   // command will be submited later

   TString cmd;
   const char* quote = SQLIdentifierQuote();
   const char* valuequote = SQLValueQuote();
   cmd.Form("INSERT INTO %s%s%s (%s%s%s, %s%s%s, %s, %s) VALUES (%d, %d, %s%s%s, %d)",
            quote, sqlio::ObjectsTable, quote,
            quote, SQLKeyIdColumn(), quote,
            quote, SQLObjectIdColumn(), quote,
            sqlio::OT_Class,
            sqlio::OT_Version,
            keyid, objid,
            valuequote, cl->GetName(), valuequote,
            cl->GetClassVersion());
   return cmd;
}

//______________________________________________________________________________
Bool_t TSQLFile::GetObjectData(Int_t objid, TString& clname, Version_t &version)
{
   // Read from objects table data for specified objectid

   if (fSQL==0) return kFALSE;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT %s, %s FROM %s%s%s WHERE %s%s%s=%d",
               sqlio::OT_Class, sqlio::OT_Version,
               quote, sqlio::ObjectsTable, quote,
               quote, SQLObjectIdColumn(), quote, objid);
   TSQLResult* res = SQLQuery(sqlcmd.Data(), 1);
   if (res==0) return kFALSE;
   TSQLRow* row = res->Next();
   if (row!=0) {
      clname = row->GetField(0);
      version = atoi(row->GetField(1));
   }

   delete row;
   delete res;
   return row!=0;
}

//______________________________________________________________________________
TSQLObjectData* TSQLFile::GetObjectClassData(Int_t objid, TSQLClassInfo* sqlinfo)
{
   // Get data for specified object from particular class table
   // Returns TSQLObjectData object, which contains one row from
   // normal class table and data from _streamer_ table.
   // TSQLObjectData object used later in TBufferSQL2 to unstream object

   if ((fSQL==0) || (objid<0) || (sqlinfo==0)) return 0;

   if (gDebug>1)
      Info("GetObjectClassData","Request for %s", sqlinfo->GetName());

   const char* quote = SQLIdentifierQuote();

   TSQLResult *classdata = 0, *blobdata = 0;

   if (sqlinfo->IsClassTableExist()) {
      TString sqlcmd;
      sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s=%d",
                  quote, sqlinfo->GetClassTableName(), quote,
                  quote, SQLObjectIdColumn(), quote, objid);
      classdata = SQLQuery(sqlcmd.Data(), 2);
   }

   if (sqlinfo->IsRawTableExist()) {
      TString sqlcmd;
      sqlcmd.Form("SELECT %s, %s FROM %s%s%s WHERE %s%s%s=%d ORDER BY %s%s%s",
                  sqlio::BT_Field, sqlio::BT_Value,
                  quote, sqlinfo->GetRawTableName(), quote,
                  quote, SQLObjectIdColumn(), quote, objid,
                  quote, SQLRawIdColumn(), quote);
      blobdata = SQLQuery(sqlcmd.Data(), 2);
   }

   return new TSQLObjectData(sqlinfo, objid, classdata, blobdata);
}

//______________________________________________________________________________
void TSQLFile::DeleteObjectFromTables(Int_t objid)
{
   // delete object with specified id from all tables

   TObjArray* tables = SQLTablesList();
   if (tables==0) return;

   const char* quote = SQLIdentifierQuote();

   TIter iter(tables);
   TObject* obj = 0;
   while ((obj=iter())!=0) {
      TString tablename = obj->GetName();

      if ((tablename.CompareTo(sqlio::KeysTable,TString::kIgnoreCase)==0) ||
          (tablename.CompareTo(sqlio::ObjectsTable,TString::kIgnoreCase)==0)) continue;

      TString query;
      query.Form("DELETE FROM %s WHERE %s%s%s=%d", tablename.Data(),
                 quote, SQLObjectIdColumn(), quote, objid);
      SQLQuery(query.Data());

   }

   delete tables;
}

//______________________________________________________________________________
const char* TSQLFile::SQLCompatibleType(Int_t typ) const
{
   // returns sql type name which is most closer to ROOT basic type
   // typ should be from TStreamerInfo:: constansts like TStreamerInfo::kInt

   return (typ<0) || (typ>18) ? 0 : fBasicTypes[typ];
}

//______________________________________________________________________________
const char* TSQLFile::SQLIntType() const
{
   // return SQL integer type

   return SQLCompatibleType(TStreamerInfo::kInt);
}
