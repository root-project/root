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
// The main motivation for the TSQLFile development is to have
// "transparent" access to SQL data base via standard TFile interface.
//
// The main approach that each class (but not each object) has one or two tables
// with names like $(CLASSNAME)_ver$(VERSION) and $(CLASSNAME)_raw$(VERSION)
// For example: TAxis_ver8 or TList_raw5
// Second kind of tables appears, when some of class members can not be converted to
// normalized form or when class has custom streamer.
// For instance, for TH1 class two tables are required: TH1_ver4 and TH1_raw4
// Most of memebers are stored in TH1_ver4 table columnwise, and only memeber:
//
//  Double_t*  fBuffer;  //[fBufferSize]
//
// can not be represented as column while size of array is not known apriory.
// Therefore, fBuffer will be written as list of values in TH1_raw4 table.
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
// All conversion to SQL statements are done with help of TSQLStructure class.
// This is special hierarchical structure wich internally is very similar
// to XML structures. TBufferSQL2 creates these structures, when object
// data is streamed by ROOT and only afterwards all SQL statements will be produced
// and applied all together.
//
// When data is reading, TBufferSQL2 will produce requests to database
// during unstreaming of object data.
//
// Optionally (default this options on) name of column includes
// suffix which indicates type of column. For instance:
//   *:parent  - parent class, column contain class version
//   *:object  - other object, column contain object id
//   *:rawdata - raw data, column contains id of raw data from _streamer_ table
//   *:Int_t   - column with integer value
// Use TSQLFile::SetUseSuffixes(kFALSE) to disable suffixes usage.
//
// This and several other options can be changed only when
// TSQLFile created with options "CREATE" or "RECREATE" and only before
// first write operation. These options are:
//     SetUseSuffixes() - suffix usage in column names (default - on)
//     SetArrayLimit()  - defines maximum array size, which can
//                        has column for each element (default 21)
//     SetTablesType()  - table type name in MySQL database (default "InnoDB")
//     SetUseIndexes()  - usage of indexes in database (default kIndexesBasic)
// Normally these functions should be called immidiately after TSQLFile constructor.
//
// When objects data written to database, by default START TRANSACTION/COMMIT
// SQL commands are used before and after data storage. If TSQLFile detects
// any problems, ROLLBACK command will be used to restore
// previous state of data base. If transactions not supported by SQL server,
// they can be disabled by SetUseTransactions(kTransactionsOff). Or user
// can take responsibility to use transactions function to hime
//
// By default only indexes for basic tables are created.
// In most cases usage of indexes increase perfomance to data reading,
// but it also can increase time of writing data to database.
// There are several modes of index usage available in SetUseIndexes() method
//
// There is MakeSelectQuery(TClass*) method, which
// produces SELECT statement to get objects data of specified class.
// Difference from simple statement like:
//   mysql> SELECT * FROM TH1I_ver1
// that not only data for that class, but also data from parent classes
// will be extracted from other tables and combined in single result table.
// Such select query can be usufull for external access to objects data.
//
// Up to now MySQL 4.1 and Oracle 9i were tested.
// Some extra work is required for other SQL databases.
// Hopefully, this should be straigthforward.
//
// Known problems and open questions.
// 1) TTree is not supported by TSQLFile. There is independent development
//    of TTreeSQL class, which allows to store trees directly in SQL database
// 2) TClonesArray is store objects in raw format,
//    which can not be accessed outside ROOT.
//    This will be changed later.
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
//  arr->Write("arr", TObject::kSingleKey);
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
#include "TList.h"
#include "TArrayC.h"
#include "TVirtualStreamerInfo.h"
#include "TStreamerElement.h"
#include "TProcessID.h"
#include "TError.h"
#include "TClass.h"

#include "TSQLServer.h"
#include "TSQLTableInfo.h"
#include "TSQLColumnInfo.h"
#include "TSQLStatement.h"
#include "TSQLResult.h"
#include "TSQLRow.h"
#include "TBufferSQL2.h"
#include "TSQLStructure.h"
#include "TKeySQL.h"
#include "TSQLClassInfo.h"
#include "TSQLObjectData.h"

#include "Riostream.h"

ClassImp(TSQLFile);

const char* mysql_BasicTypes[21] = {
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
"DOUBLE",           // kFloat16 = 19,
""
};

const char* mysql_OtherTypes[13] = {
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
"\"",               // quote for string values in MySQL
"InnoDB"            // default tables types, used only for MySQL tables
};

const char* oracle_BasicTypes[21] = {
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
"FLOAT",            // kFloat16 = 19,
""
};

const char* oracle_OtherTypes[13] = {
"VARCHAR(1000)",    // smallest text
"1000",             // maximum size of smallest text
"VARCHAR(4000)",    // biggest size text, CLOB is not yet supported by TOracleRow
"VARCHAR(50)",      // date & time
"\"",               // quote for identifier like table name or column name
"dir:id",           // dir id column
"key:id",           // key id column
"obj:id",           // object id column
"raw:id",           // raw data id column
"str:id",           // string id column
":",                // name separator between name and type like TObject:parent
"'",                // quote for string values in Oracle
""                  // default tables types, used only for MySQL tables
};


//______________________________________________________________________________
TSQLFile::TSQLFile() :
   TFile(),
   fSQL(0),
   fSQLClassInfos(0),
   fUseSuffixes(kTRUE),
   fSQLIOversion(1),
   fArrayLimit(21),
   fCanChangeConfig(kFALSE),
   fTablesType(),
   fUseTransactions(0),
   fUseIndexes(0),
   fModifyCounter(0),
   fQuerisCounter(0),
   fBasicTypes(0),
   fOtherTypes(0),
   fUserName(),
   fLogFile(0),
   fIdsTableExists(kFALSE),
   fStmtCounter(0)
{
   // default TSQLFile constructor
   SetBit(kBinaryFile, kFALSE);
}

//______________________________________________________________________________
TSQLFile::TSQLFile(const char* dbname, Option_t* option, const char* user, const char* pass) :
   TFile(),
   fSQL(0),
   fSQLClassInfos(0),
   fUseSuffixes(kTRUE),
   fSQLIOversion(1),
   fArrayLimit(21),
   fCanChangeConfig(kFALSE),
   fTablesType(),
   fUseTransactions(0),
   fUseIndexes(0),
   fModifyCounter(0),
   fQuerisCounter(0),
   fBasicTypes(mysql_BasicTypes),
   fOtherTypes(mysql_OtherTypes),
   fUserName(user),
   fLogFile(0),
   fIdsTableExists(kFALSE),
   fStmtCounter(0)
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
   //                             If data base open by other TSQLFile instance for writing,
   //                             write access will be rejected
   //           = BREAKLOCK       Special case when lock was not correctly released
   //                             by TSQLFile instance. This may happen if program crashed when
   //                             TSQLFile was open with write access mode.
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
   TDirectoryFile::Build();
   fFile = this;

   if (dbname && strstr(dbname,"oracle://")!=0) {
      fBasicTypes = oracle_BasicTypes;
      fOtherTypes = oracle_OtherTypes;
   }

   fArrayLimit = 21;
   fTablesType = SQLDefaultTableType();
   fUseIndexes = 1;
   fUseTransactions = kTransactionsAuto;

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
   fProcessIDs = 0;
   fNProcessIDs= 0;
   fSeekDir    = sqlio::Ids_RootDir;
   SetBit(kBinaryFile, kFALSE);

   fOption = option;
   fOption.ToUpper();

   if (fOption == "NEW") fOption = "CREATE";

   Bool_t breaklock = kFALSE;

   if (fOption == "BREAKLOCK") { breaklock = kTRUE; fOption = "UPDATE"; }

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
      if (IsTablesExists())
         if (!IsWriteAccess()) {
            Error("TSQLFile", "no write permission, DB %s locked", dbname);
            goto zombie;
         }
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

      if (update && !breaklock && !IsWriteAccess()) {
         Error("TSQLFile", "no write permission, DB %s locked", dbname);
         goto zombie;
      }
   }

   if (read) {
      if (!IsTablesExists()) {
         Error("TSQLFile", "DB %s tables not exist", dbname);
         goto zombie;
      }
      if (!IsReadAccess()) {
         Error("TSQLFile", "no read permission for DB %s tables", dbname);
         goto zombie;
      }
   }

   fRealName = dbname;

   if (create || update) {
      SetWritable(kTRUE);
      if (update) SetLocking(kLockBusy);
   } else
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
Bool_t TSQLFile::IsODBC() const
{
   // checks, if ODBC driver used for database connection

   if (fSQL==0) return kFALSE;
   return strcmp(fSQL->ClassName(),"TODBCServer")==0;

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
   // Default value is 21

   if (!fCanChangeConfig)
      Error("SetArrayLimit", "Configurations already cannot be changed");
   else
      fArrayLimit = limit;
}

//______________________________________________________________________________
void TSQLFile::SetTablesType(const char* tables_type)
{
   // Defines tables type, which is used in CREATE TABLE statements
   // Now is only used for MySQL database, where following types are supported:
   //    "BDB", "HEAP", "ISAM", "InnoDB", "MERGE", "MRG_MYISAM", "MYISAM"
   // Default for TSQLFile is "InnoDB". For more detailes see MySQL docs.

   if (!fCanChangeConfig)
      Error("SetTablesType", "Configurations already cannot be changed");
   else
      fTablesType = tables_type;
}

//______________________________________________________________________________
void TSQLFile::SetUseTransactions(Int_t mode)
{
   // Defines usage of transactions statements for writing objects data to database.
   //    kTransactionsOff=0   - no transaction operation are allowed
   //    kTransactionsAuto=1  - automatic mode. Each write operation,
   //        produced by TSQLFile, will be supplied by START TRANSACTION and COMMIT calls.
   //        If any error happen, ROLLBACK will returns database to previous state
   //    kTransactionsUser=2  - transactions are delegated to user. Methods
   //        StartTransaction(), Commit() and Rollback() should be called by user.
   // Default UseTransactions option is kTransactionsAuto

   fUseTransactions = mode;
}

//______________________________________________________________________________
Bool_t TSQLFile::StartTransaction()
{
   // Start user transaction.
   // This can be usesfull, when big number of objects should be stored in
   // data base and commitment required only if all operations were succesfull.
   // In that case in the end of all operations method Commit() should be
   // called. If operation on user-level is looks like not successfull,
   // method Rollback() will return database data and TSQLFile instance to
   // previous state.
   // In MySQL not all tables types support transaction mode of operation.
   // See SetTablesType() method for details .

   if (GetUseTransactions()!=kTransactionsUser) {
      Error("SQLStartTransaction","Only allowed when SetUseTransactions(kUserTransactions) was configured");
      return kFALSE;
   }

   return SQLStartTransaction();
}

//______________________________________________________________________________
Bool_t TSQLFile::Commit()
{
   // Commit transaction, started by StartTransaction() call.
   // Only after that call data will be written and visible on database side.

   if (GetUseTransactions()!=kTransactionsUser) {
      Error("SQLCommit","Only allowed when SetUseTransactions(kUserTransactions) was configured");
      return kFALSE;
   }

   return SQLCommit();
}

//______________________________________________________________________________
Bool_t TSQLFile::Rollback()
{
   // Rollback all operations, done after StartTransaction() call.
   // Database should return to initial state.

   if (GetUseTransactions()!=kTransactionsUser) {
      Error("SQLRollback","Only allowed when SetUseTransactions(kUserTransactions) was configured");
      return kFALSE;
   }

   return SQLRollback();
}

//______________________________________________________________________________
void TSQLFile::SetUseIndexes(Int_t use_type)
{
   // Specify usage of indexes for data tables
   //    kIndexesNone = 0  - no indexes are used
   //    kIndexesBasic = 1 - indexes used only for keys list and
   //                        objects list tables (default)
   //    kIndexesClass = 2 - index also created for every normal class table
   //    kIndexesAll = 3   - index created for every table, including _streamer_ tables
   // Indexes in general should increase speed of access to objects data,
   // but they required more operations and more disk space on server side

   if (!fCanChangeConfig)
      Error("SetUseIndexes", "Configurations already cannot be changed");
   else
      fUseIndexes = use_type;
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

   if (IsWritable()) {
      SaveToDatabase();
      SetLocking(kLockFree);
   }

   fWritable = kFALSE;

   if (fClassIndex) {
      delete fClassIndex;
      fClassIndex = 0;
   }

   {
      TDirectory::TContext ctxt(this);
      // Delete all supported directories structures from memory
      TDirectoryFile::Close();
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

      if (IsOpen() && IsWritable()) {
         SaveToDatabase();
         SetLocking(kLockFree);
      }
      fOption = opt;

      SetWritable(kFALSE);

   } else {
      // switch to UPDATE mode

      if (!IsWriteAccess()) {
         Error("ReOpen","Tables are locked, no write access");
         return 1;
      }

      fOption = opt;

      SetWritable(kTRUE);

      SetLocking(kLockBusy);
   }

   return 0;
}

//______________________________________________________________________________
TKey* TSQLFile::CreateKey(TDirectory* mother, const TObject* obj, const char* name, Int_t )
{
   // create SQL key, which will store object in data base
   return new TKeySQL(mother, obj, name);
}

//______________________________________________________________________________
TKey* TSQLFile::CreateKey(TDirectory* mother, const void* obj, const TClass* cl, const char* name, Int_t )
{
   // create SQL key, which will store object in data base
   return new TKeySQL(mother, obj, cl, name);
}

//______________________________________________________________________________
void TSQLFile::WriteHeader()
{
   // Write file info like configurations, title, UUID and other

   WriteSpecialObject(sqlio::Ids_TSQLFile, this, GetName(), GetTitle());
}

//______________________________________________________________________________
void TSQLFile::WriteStreamerInfo()
{
   // Store all TVirtualStreamerInfo, used in file, in sql database

   // return;

   // do not write anything when no basic tables was created
   if (!IsTablesExists()) return;

   if (gDebug>1)
      Info("WriteStreamerInfo","Saving streamer infos to database");

   TList list;

   TIter iter(gROOT->GetListOfStreamerInfo());

   TVirtualStreamerInfo* info = 0;

   while ((info = (TVirtualStreamerInfo*) iter()) !=0 ) {
      Int_t uid = info->GetNumber();
      if (fClassIndex->fArray[uid]) {
         if (gDebug>1) Info("WriteStreamerInfo","Add %s",info->GetName());
         list.Add(info);
      }
   }
   if (list.GetSize()==0) return;
   fClassIndex->fArray[0] = 2; //to prevent adding classes in TVirtualStreamerInfo::TagFile

   WriteSpecialObject(sqlio::Ids_StreamerInfos, &list, "StreamerInfo", "StreamerInfos of this file");

   fClassIndex->fArray[0] = 0; //to prevent adding classes in TVirtualStreamerInfo::TagFile
}

//______________________________________________________________________________
Bool_t TSQLFile::WriteSpecialObject(Long64_t keyid, TObject* obj, const char* name, const char* title)
{
// write special kind of object like streamer infos or file itself
// keys for that objects should exist in tables but not indicated in list of keys,
// therefore users can not get them with TDirectoryFile::Get() method

   DeleteKeyFromDB(keyid);
   if (obj==0) return kTRUE;

   Long64_t objid = StoreObjectInTables(keyid, obj, obj->IsA());

   if (objid>0) {
      TDatime now;

      TKeySQL* key = new TKeySQL(this, keyid, objid,
                                 name, title,
                                 now.AsSQLString(), 1, obj->ClassName());
      WriteKeyData(key);
      delete key;
   }

   return (objid>0);
}

//______________________________________________________________________________
TObject* TSQLFile::ReadSpecialObject(Long64_t keyid, TObject* obj)
{
   // Read data of special kind of objects

   TKeySQL* key = 0;

   StreamKeysForDirectory(this, kFALSE, keyid, &key);
   if (key==0) return obj;

   TBufferSQL2 buffer(TBuffer::kRead, this);

   TClass* cl = 0;

   void* res = buffer.SqlReadAny(key->GetDBKeyId(), key->GetDBObjId(), &cl, obj);

   if ((cl==TSQLFile::Class()) && (res!=0) && (obj==this)) {
      // name should not be preserved while name of database may be changed
      SetTitle(key->GetTitle());
   }

   delete key;

   return (TObject*) res;
}

//______________________________________________________________________________
TList* TSQLFile::GetStreamerInfoList()
{
   // Read back streamer infos from database
   // List of streamer infos is always stored with key:id 0,
   // which is not shown in normal keys list

//   return new TList;

   if (gDebug>1)
      Info("GetStreamerInfoList","Start reading of streamer infos");

   TObject* obj = ReadSpecialObject(sqlio::Ids_StreamerInfos);

   TList* list = dynamic_cast<TList*> (obj);
   if (list==0) { delete obj; list = new TList; }

   return list;
}

//______________________________________________________________________________
void TSQLFile::SaveToDatabase()
{
   // save data which is not yet in Database
   // Typically this is streamerinfos structures or

   if (fSQL==0) return;

   WriteStreamerInfo();
   WriteHeader();
}

//______________________________________________________________________________
Int_t TSQLFile::StreamKeysForDirectory(TDirectory* dir, Bool_t doupdate, Long64_t specialkeyid, TKeySQL** specialkey)
{
   // read keys for specified directory (when update == kFALSE)
   // or update value for modified keys when update == kTRUE
   // Returns number of succesfully read keys or -1 if error

   if (dir==0) return -1;

   const char* quote = SQLIdentifierQuote();

   Long64_t dirid = dir->GetSeekDir();

   TString sqlcmd;
   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s=%lld",
               quote, sqlio::KeysTable, quote,
               quote, SQLDirIdColumn(), quote, dirid);
   if (specialkeyid>=0) {
      TString buf;
      buf.Form(" AND %s%s%s=%lld", quote, SQLKeyIdColumn(), quote, specialkeyid);
      sqlcmd += buf;
   }

   TSQLResult* res = SQLQuery(sqlcmd.Data(), 2);

   if (res==0) return -1;

   Int_t nkeys = 0;

   TSQLRow* row = 0;

   while ((row = res->Next()) != 0) {
      nkeys++;

      Long64_t keyid = sqlio::atol64((*row)[0]);
      //      Int_t dirid = atoi((*row)[1]);
      Long64_t objid = sqlio::atol64((*row)[2]);
      const char* keyname = (*row)[3];
      const char* keytitle = (*row)[4];
      const char* keydatime = (*row)[5];
      Int_t cycle = atoi((*row)[6]);
      const char* classname = (*row)[7];

      if (gDebug>4)
        cout << "  Reading keyid = " << keyid << " name = " << keyname << endl;

      if ((keyid>=sqlio::Ids_FirstKey) || (keyid==specialkeyid)) {
         if (doupdate) {
            TKeySQL* key = FindSQLKey(dir, keyid);

            if (key==0) {
               Error("StreamKeysForDirectory","Key with id %lld not exist in list", keyid);
               nkeys = -1; // this will finish execution
            } else
            if (key->IsKeyModified(keyname, keytitle, keydatime, cycle, classname))
               UpdateKeyData(key);

         } else {
            TKeySQL* key = new TKeySQL(dir, keyid, objid,
                                       keyname, keytitle,
                                       keydatime, cycle, classname);
            if (specialkey!=0)
               { *specialkey = key; nkeys = 1; }
            else
               dir->GetListOfKeys()->Add(key);
         }
      }
      delete row;
   }

   delete res;

   if (gDebug>4) {
      Info("StreamKeysForDirectory","dir = %s numread = %d",dir->GetName(), nkeys);
      dir->GetListOfKeys()->Print("*");
   }

   return nkeys;
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

      // read data corresponding to TSQLFile
      if (ok) {
         ReadSQLClassInfos();

         ReadStreamerInfo();

         ok = (ReadSpecialObject(sqlio::Ids_TSQLFile, this) != 0);
      }

      // read list of keys
      if (ok)
         ok = StreamKeysForDirectory(this, kFALSE)>=0;

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

   // Int_t lock = 0;

   #define ReadIntCfg(name, target)                        \
     if ((field.CompareTo(name, TString::kIgnoreCase)==0)) \
        target = value.Atoi(); else

   #define ReadBoolCfg(name, target)                        \
     if ((field.CompareTo(name, TString::kIgnoreCase)==0))  \
        target = value.CompareTo(sqlio::True, TString::kIgnoreCase)==0; else

   #define ReadStrCfg(name, target)                         \
     if ((field.CompareTo(name, TString::kIgnoreCase)==0))  \
        target = value; else

   TSQLRow* row = 0;

   while ((row = res->Next()) != 0) {

      TString field = row->GetField(0);
      TString value = row->GetField(1);

      delete row;

      ReadIntCfg(sqlio::cfg_Version, fSQLIOversion)
      ReadBoolCfg(sqlio::cfg_UseSufixes, fUseSuffixes)
      ReadIntCfg(sqlio::cfg_ArrayLimit, fArrayLimit)
      ReadStrCfg(sqlio::cfg_TablesType, fTablesType)
      ReadIntCfg(sqlio::cfg_UseTransactions, fUseTransactions)
      ReadIntCfg(sqlio::cfg_UseIndexes, fUseIndexes)
      ReadIntCfg(sqlio::cfg_ModifyCounter, fModifyCounter)
      // ReadIntCfg(sqlio::cfg_LockingMode, lock)
      {
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
   // Creates initial tables in database
   // This is table with configurations and table with keys
   // Function called once when first object is stored to the file.

   TString sqlcmd;

   const char* quote = SQLIdentifierQuote();
   const char* vquote = SQLValueQuote();

   if (SQLTestTable(sqlio::ConfigTable)) {
      sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlio::ConfigTable, quote);
      SQLQuery(sqlcmd.Data());
   }

   sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s)",
               quote, sqlio::ConfigTable, quote,
               quote, sqlio::CT_Field, quote, SQLSmallTextType(),
               quote, sqlio::CT_Value, quote, SQLSmallTextType());
   if ((fTablesType.Length()>0) && IsMySQL()) {
      sqlcmd +=" TYPE=";
      sqlcmd += fTablesType;
   }

   SQLQuery(sqlcmd.Data());

   #define WrintCfg(name, type, value)                              \
   {                                                                \
      sqlcmd.Form("INSERT INTO %s%s%s VALUES (%s%s%s, %s"type"%s)", \
                  quote, sqlio::ConfigTable, quote,                 \
                  vquote, name, vquote,                             \
                  vquote, value, vquote);                           \
      SQLQuery(sqlcmd.Data());                                      \
   }

   WrintCfg(sqlio::cfg_Version, "%d", fSQLIOversion);
   WrintCfg(sqlio::cfg_UseSufixes, "%s", fUseSuffixes ? sqlio::True : sqlio::False);
   WrintCfg(sqlio::cfg_ArrayLimit, "%d", fArrayLimit);
   WrintCfg(sqlio::cfg_TablesType, "%s", fTablesType.Data());
   WrintCfg(sqlio::cfg_UseTransactions, "%d", fUseTransactions);
   WrintCfg(sqlio::cfg_UseIndexes, "%d", fUseIndexes);
   WrintCfg(sqlio::cfg_ModifyCounter, "%d", fModifyCounter);
   WrintCfg(sqlio::cfg_LockingMode, "%d", kLockBusy);

   // from this moment on user cannot change configurations
   fCanChangeConfig = kFALSE;

   if (SQLTestTable(sqlio::KeysTable)) {
      sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlio::KeysTable, quote);
      SQLQuery(sqlcmd.Data());
   }

   sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s)",
               quote, sqlio::KeysTable, quote,
               quote, SQLKeyIdColumn(), quote, SQLIntType(),
               quote, SQLDirIdColumn(), quote, SQLIntType(),
               quote, SQLObjectIdColumn(), quote, SQLIntType(),
               quote, sqlio::KT_Name, quote, SQLSmallTextType(),
               quote, sqlio::KT_Title, quote, SQLSmallTextType(),
               quote, sqlio::KT_Datetime, quote, SQLDatetimeType(),
               quote, sqlio::KT_Cycle, quote, SQLIntType(),
               quote, sqlio::KT_Class, quote, SQLSmallTextType());

   if ((fTablesType.Length()>0) && IsMySQL()) {
      sqlcmd +=" TYPE=";
      sqlcmd += fTablesType;
   }

   SQLQuery(sqlcmd.Data());

   if (GetUseIndexes()>kIndexesNone) {
      sqlcmd.Form("CREATE UNIQUE INDEX %s%s%s ON %s%s%s (%s%s%s)",
                   quote, sqlio::KeysTableIndex, quote,
                   quote, sqlio::KeysTable, quote,
                   quote, SQLKeyIdColumn(), quote);
      SQLQuery(sqlcmd.Data());
   }
}

//______________________________________________________________________________
void TSQLFile::IncrementModifyCounter()
{
   // Update value of modify counter in config table
   // Modify counter used to indicate that something was changed in database.
   // It will be used when multiple instances of TSQLFile for the same data base
   // will be connected.

   if (!IsWritable()) {
      Error("IncrementModifyCounter","Cannot update tables without write accsess");
      return;
   }

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   const char* vquote = SQLValueQuote();

   sqlcmd.Form("UPDATE %s%s%s SET %s%s%s=%d WHERE %s%s%s=%s%s%s",
                quote, sqlio::ConfigTable, quote,
                quote, sqlio::CT_Value, quote, ++fModifyCounter,
                quote, sqlio::CT_Field, quote,
                vquote, sqlio::cfg_ModifyCounter, vquote);
   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
TString TSQLFile::MakeSelectQuery(TClass* cl)
{
   // Produce SELECT statement which can be used to get all data
   // of class cl in one SELECT statement
   // This statement also can be used to create VIEW by command like
   // mysql> CREATE VIEW TH1I_view AS $CLASSSELECT$
   // Where $CLASSSELECT$ argument should be produced by call
   //   f->MakeSelectQuery(TH1I::Class());
   // VIEWs supported by latest MySQL 5 and Oracle

   TString res = "";
   TSQLClassInfo* sqlinfo = FindSQLClassInfo(cl);
   if (sqlinfo==0) return res;

   TString columns, tables;
   Int_t tablecnt = 0;

   if (!ProduceClassSelectQuery(cl->GetStreamerInfo(), sqlinfo, columns, tables, tablecnt))
      return res;

   res.Form("SELECT %s FROM %s", columns.Data(), tables.Data());

   return res;
}

//______________________________________________________________________________
Bool_t TSQLFile::ProduceClassSelectQuery(TVirtualStreamerInfo* info,
                                         TSQLClassInfo* sqlinfo,
                                         TString& columns,
                                         TString& tables,
                                         Int_t& tablecnt)
{
   // used by MakeClassSelectQuery method to add columns from table of
   // class, specified by TVirtualStreamerInfo structure

   if ((info==0) || (sqlinfo==0)) return kFALSE;

   if (!sqlinfo->IsClassTableExist()) return kFALSE;

   const char* quote = SQLIdentifierQuote();

   TString table_syn;
   table_syn.Form("t%d", ++tablecnt);

   Bool_t start = tables.Length()==0;

   TString buf;

   if (start)
      buf.Form("%s AS %s", sqlinfo->GetClassTableName(), table_syn.Data());
   else
      buf.Form(" LEFT JOIN %s AS %s USING(%s%s%s)",
               sqlinfo->GetClassTableName(), table_syn.Data(),
               quote, SQLObjectIdColumn(), quote);

   tables += buf;

   if (start)
      columns.Form("%s.%s%s%s",table_syn.Data(), quote, SQLObjectIdColumn(), quote);

   if (info->GetClass()==TObject::Class()) {
      buf.Form(", %s.%s",table_syn.Data(), sqlio::TObjectUniqueId);
      columns+=buf;
      buf.Form(", %s.%s",table_syn.Data(), sqlio::TObjectBits);
      columns+=buf;
      buf.Form(", %s.%s",table_syn.Data(), sqlio::TObjectProcessId);
      columns+=buf;
      return kTRUE;
   }

   TIter iter(info->GetElements());
   TStreamerElement* elem = 0;

   while ((elem = (TStreamerElement*) iter()) != 0) {
      Int_t coltype = TSQLStructure::DefineElementColumnType(elem, this);
      TString colname = TSQLStructure::DefineElementColumnName(elem, this);

      buf = "";
      switch (coltype) {

         case TSQLStructure::kColObject:
         case TSQLStructure::kColObjectPtr:
         case TSQLStructure::kColTString:
         case TSQLStructure::kColSimple: {
            buf.Form(", %s.%s%s%s",table_syn.Data(), quote, colname.Data(), quote);
            columns+=buf;
            break;
         }

         case TSQLStructure::kColParent: {
            TClass* parentcl = elem->GetClassPointer();
            ProduceClassSelectQuery(parentcl->GetStreamerInfo(),
                                    FindSQLClassInfo(parentcl),
                                    columns, tables, tablecnt);
            break;
         }

         case TSQLStructure::kColSimpleArray: {
            for(Int_t n=0;n<elem->GetArrayLength();n++) {
               colname = TSQLStructure::DefineElementColumnName(elem, this, n);
               buf.Form(", %s.%s%s%s",table_syn.Data(), quote, colname.Data(), quote);
               columns+=buf;
            }
            break;
         }
      } // switch
   }

   return (columns.Length()>0) && (tables.Length()>0);
}

//______________________________________________________________________________
Bool_t TSQLFile::IsTablesExists()
{
   // Checks if main keys table is existing

   return SQLTestTable(sqlio::KeysTable) && SQLTestTable(sqlio::ConfigTable);
}

//______________________________________________________________________________
Bool_t TSQLFile::IsWriteAccess()
{
   // Checkis, if lock is free in configuration tables

   return GetLocking()==kLockFree;
}

//______________________________________________________________________________
void TSQLFile::SetLocking(Int_t mode)
{
   // Set locking mode for current database

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   const char* vquote = SQLValueQuote();

   sqlcmd.Form("UPDATE %s%s%s SET %s%s%s=%d WHERE %s%s%s=%s%s%s",
                quote, sqlio::ConfigTable, quote,
                quote, sqlio::CT_Value, quote, mode,
                quote, sqlio::CT_Field, quote,
                vquote, sqlio::cfg_LockingMode, vquote);
   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
Int_t TSQLFile::GetLocking()
{
   // Return current locking mode for that file

   const char* quote = SQLIdentifierQuote();
   const char* vquote = SQLValueQuote();

   TString sqlcmd;
   sqlcmd.Form("SELECT %s%s%s FROM %s%s%s WHERE %s%s%s=%s%s%s",
                quote, sqlio::CT_Value, quote,
                quote, sqlio::ConfigTable, quote,
                quote, sqlio::CT_Field, quote,
                vquote, sqlio::cfg_LockingMode, vquote);

   TSQLResult* res = SQLQuery(sqlcmd.Data(), 1);
   TSQLRow* row = (res==0) ? 0 : res->Next();
   TString field = (row==0) ? "" : row->GetField(0);
   delete row;
   delete res;

   if (field.Length()==0) return kLockFree;

   return field.Atoi();
}

//______________________________________________________________________________
Bool_t TSQLFile::IsReadAccess()
{
   // dummy, in future should check about read access to database

   return kTRUE;
}

//______________________________________________________________________________
TSQLResult* TSQLFile::SQLQuery(const char* cmd, Int_t flag, Bool_t* ok)
{
   // submits query to SQL server
   // if flag==0, result is not interesting and will be deleted
   // if flag==1, return result of submitted query
   // if flag==2, results is may be necessary for long time
   //             Oracle plugin do not support working with several TSQLResult
   //             objects, therefore explicit deep copy will be produced
   // If ok!=0, it will contains kTRUE is Query was successfull, otherwise kFALSE

   if (fLogFile!=0)
      *fLogFile << cmd << endl;

   if (ok!=0) *ok = kFALSE;

   if (fSQL==0) return 0;

   if (gDebug>2) Info("SQLQuery", "%s", cmd);

   fQuerisCounter++;

   if (flag==0) {
      Bool_t res = fSQL->Exec(cmd);
      if (ok!=0) *ok = res;
      return 0;
   }

   TSQLResult* res = fSQL->Query(cmd);
   if (ok!=0) *ok = res!=0;
   if (res==0) return 0;
//   if ((flag==2) && IsOracle())
//      res = new TSQLResultCopy(res);
   return res;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLCanStatement()
{
   // Test if DB support statement and number of open statements is not exceeded

   if (fSQL==0) return kFALSE;

   if (!fSQL->HasStatement()) return kFALSE;

   return kTRUE; // !IsOracle() || (fStmtCounter<15);
}

//______________________________________________________________________________
TSQLStatement* TSQLFile::SQLStatement(const char* cmd, Int_t bufsize)
{
   // Produces SQL statement for currently conected DB server

   if (fSQL==0) return 0;

   if (!fSQL->HasStatement()) return 0;

   if (gDebug>1)
      Info("SQLStatement", "%s", cmd);

   fStmtCounter++;
   fQuerisCounter++; // one statement counts as one query

   return fSQL->Statement(cmd, bufsize);
}

//______________________________________________________________________________
void TSQLFile::SQLDeleteStatement(TSQLStatement* stmt)
{
   // delete statement and decrease counter

   if (stmt==0) return;

   fStmtCounter--;

   delete stmt;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLApplyCommands(TObjArray* cmds)
{
   // supplies set of commands to server
   // Commands is stored as array of TObjString

   if ((cmds==0) || (fSQL==0)) return kFALSE;

   Bool_t ok = kTRUE;
   TIter iter(cmds);
   TObject* cmd= 0;
   while ((cmd=iter())!=0) {
      SQLQuery(cmd->GetName(),0,&ok);
      if(!ok) break;
   }

   return ok;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLTestTable(const char* tablename)
{
   // Test, if table of specified name exists

   if (fSQL==0) return kFALSE;

   if (fSQL->HasTable(tablename)) return kTRUE;

   TString buf(tablename);
   buf.ToLower();
   if (fSQL->HasTable(buf.Data())) return kTRUE;
   buf.ToUpper();
   return fSQL->HasTable(buf.Data());
}

//______________________________________________________________________________
Long64_t TSQLFile::SQLMaximumValue(const char* tablename, const char* columnname)
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

   Long64_t maxid = -1;
   if (row!=0)
      if (row->GetField(0)!=0)
         maxid = sqlio::atol64(row->GetField(0));

   delete row;
   delete res;

   if (gDebug>2)
      Info("SQLMaximumValue","Result = %lld",maxid);;

   return maxid;
}

//______________________________________________________________________________
void TSQLFile::SQLDeleteAllTables()
{
   // Delete all tables in database

   if (fSQL==0) return;

   TList* tables = fSQL->GetTablesList();
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
Bool_t TSQLFile::SQLStartTransaction()
{
   // Start SQL transaction.

   return fSQL ? fSQL->StartTransaction() : kFALSE;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLCommit()
{
   // Commit SQL transaction

   return fSQL ? fSQL->Commit() : kFALSE;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLRollback()
{
   // Rollback all SQL operations, done after start transaction

   return fSQL ? fSQL->Rollback() : kFALSE;
}

//______________________________________________________________________________
Int_t TSQLFile::SQLMaxIdentifierLength()
{
   // returns maximum allowed length of identifiers

   Int_t maxlen = fSQL==0 ? 32 : fSQL->GetMaxIdentifierLength();

   // lets exclude absolute ubnormal data
   if (maxlen<10) maxlen = 10;

   return maxlen;
}

//______________________________________________________________________________
void TSQLFile::DeleteKeyFromDB(Long64_t keyid)
{
// remove key with specified id from keys table
// also removes all objects data, related to this table

   if (!IsWritable() || (keyid<0) || (fSQL==0)) return;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();

   sqlcmd.Form("SELECT MIN(%s%s%s), MAX(%s%s%s) FROM %s%s%s WHERE %s%s%s=%lld",
               quote, SQLObjectIdColumn(), quote,
               quote, SQLObjectIdColumn(), quote,
               quote, sqlio::ObjectsTable, quote,
               quote, SQLKeyIdColumn(), quote, keyid);
   TSQLResult* res = SQLQuery(sqlcmd.Data(), 2);
   TSQLRow* row = res==0 ? 0 : res->Next();
   Long64_t minid(1), maxid(0);

   if ((row!=0) && (row->GetField(0)!=0) && (row->GetField(1)!=0)) {
      minid = sqlio::atol64(row->GetField(0));
      maxid = sqlio::atol64(row->GetField(1));
   }

   delete row;
   delete res;

   // can be that object tables does not include any entry this that keyid
   if (minid<=maxid) {
      TIter iter(fSQLClassInfos);
      TSQLClassInfo* info = 0;
      TString querymask, query;
      querymask.Form("DELETE FROM %s%s%s WHERE %s%s%s BETWEEN %lld AND %lld",
                     quote, "%s", quote,
                     quote, SQLObjectIdColumn(), quote,
                     minid, maxid);

      while ((info = (TSQLClassInfo*) iter()) !=0 ) {

         if (info->IsClassTableExist()) {
            query.Form(querymask.Data(), info->GetClassTableName());
            SQLQuery(query.Data());
         }

         if (info->IsRawTableExist()) {
            query.Form(querymask.Data(), info->GetRawTableName());
            SQLQuery(query.Data());
         }
      }
   }

   sqlcmd.Form("DELETE FROM %s%s%s WHERE %s%s%s=%lld", quote, sqlio::ObjectsTable, quote, quote, SQLKeyIdColumn(), quote, keyid);
   SQLQuery(sqlcmd.Data());

   sqlcmd.Form("DELETE FROM %s%s%s WHERE %s%s%s=%lld", quote, sqlio::KeysTable, quote, quote, SQLKeyIdColumn(), quote, keyid);
   SQLQuery(sqlcmd.Data());

   IncrementModifyCounter();
}

//______________________________________________________________________________
TKeySQL* TSQLFile::FindSQLKey(TDirectory* dir, Long64_t keyid)
{
   // Search for TKeySQL object with specified keyid

   if (dir==0) return 0;

   TIter next(dir->GetListOfKeys());
   TObject* obj = 0;

   while ((obj = next())!=0) {
      TKeySQL* key = dynamic_cast<TKeySQL*> (obj);
      if (key!=0)
         if (key->GetDBKeyId()==keyid) return key;
   }

   return 0;
}

//______________________________________________________________________________
Bool_t TSQLFile::WriteKeyData(TKeySQL* key)
{
   // add entry into keys table

   if ((fSQL==0) || (key==0)) return kFALSE;

   if (!IsTablesExists()) CreateBasicTables();

   TString sqlcmd;
   const char* valuequote = SQLValueQuote();
   const char* quote = SQLIdentifierQuote();

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%lld, %lld, %lld, %s%s%s, %s%s%s, %s%s%s, %d, %s%s%s)",
               quote, sqlio::KeysTable, quote,
               key->GetDBKeyId(), key->GetDBDirId(), key->GetDBObjId(),
               valuequote, key->GetName(), valuequote,
               valuequote, key->GetTitle(), valuequote,
               valuequote, key->GetDatime().AsSQLString(), valuequote,
               key->GetCycle(),
               valuequote, key->GetClassName(), valuequote);

   Bool_t ok = kTRUE;

   SQLQuery(sqlcmd.Data(), 0, &ok);

   if (ok) IncrementModifyCounter();

   return ok;
}

//______________________________________________________________________________
Bool_t TSQLFile::UpdateKeyData(TKeySQL* key)
{
   // updates (overwrites) key data in KeysTable

   if ((fSQL==0) || (key==0)) return kFALSE;

   TString sqlcmd;
   const char* valuequote = SQLValueQuote();
   const char* quote = SQLIdentifierQuote();

   TString keyname = key->GetName();
   TString keytitle = key->GetTitle();
   TString keydatime = key->GetDatime().AsSQLString();

   TSQLStructure::AddStrBrackets(keyname, valuequote);
   TSQLStructure::AddStrBrackets(keytitle, valuequote);
   TSQLStructure::AddStrBrackets(keydatime, valuequote);

   sqlcmd.Form("UPDATE %s%s%s SET %s%s%s=%s, %s%s%s=%s, %s%s%s=%s, %s%s%s=%d WHERE %s%s%s=%lld",
                quote, sqlio::KeysTable, quote,
                quote, sqlio::KT_Name, quote, keyname.Data(),
                quote, sqlio::KT_Title, quote, keytitle.Data(),
                quote, sqlio::KT_Datetime, quote, keydatime.Data(),
                quote, sqlio::KT_Cycle, quote, key->GetCycle(),
                quote, SQLKeyIdColumn(), quote, key->GetDBKeyId());

   Bool_t ok = kTRUE;

   SQLQuery(sqlcmd.Data(), 0, &ok);

   if (ok) IncrementModifyCounter();

   return ok;
}

//______________________________________________________________________________
Long64_t TSQLFile::DefineNextKeyId()
{
   // Returns next possible key identifier

   Long64_t max = -1;

   if (SQLTestTable(sqlio::KeysTable))
      max = SQLMaximumValue(sqlio::KeysTable, SQLKeyIdColumn());

   if (max<0) return sqlio::Ids_FirstKey;

   return max+1;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::FindSQLClassInfo(const char* clname, Int_t version)
{
   // return (if exists) TSQLClassInfo for specified class name and version

   if (fSQLClassInfos==0) return 0;

   TIter iter(fSQLClassInfos);
   TSQLClassInfo* info = 0;

   while ((info = (TSQLClassInfo*) iter()) !=0 ) {
      if (strcmp(info->GetName(), clname)==0)
         if (info->GetClassVersion()==version) return info;
   }
   return 0;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::FindSQLClassInfo(const TClass* cl)
{
   // return (if exists) TSQLClassInfo for specified class

   return FindSQLClassInfo(cl->GetName(), cl->GetClassVersion());
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::RequestSQLClassInfo(const char* clname, Int_t version)
{
   // search in database tables for specified class and return TSQLClassInfo object

   TSQLClassInfo* info = FindSQLClassInfo(clname, version);
   if (info!=0) return info;

   if (fSQL==0) return 0;

   Long64_t maxid = 0;

   if (fSQLClassInfos!=0) {
      TIter iter(fSQLClassInfos);
      info = 0;
      while ((info = (TSQLClassInfo*) iter()) !=0 ) {
         if (info->GetClassId()>maxid)
            maxid = info->GetClassId();
      }
   }

   info = new TSQLClassInfo(maxid+1, clname, version);

   info->SetClassTableName(DefineTableName(clname, version, kFALSE));
   info->SetRawTableName(DefineTableName(clname, version, kTRUE));

   if (fSQLClassInfos==0) fSQLClassInfos = new TList;
   fSQLClassInfos->Add(info);

   return info;
}

//______________________________________________________________________________
TString TSQLFile::DefineTableName(const char* clname, Int_t version, Bool_t rawtable)
{
   // proposes table name for class

   Int_t maxlen = SQLMaxIdentifierLength();

   TString res;

   const char *suffix = rawtable ? "_raw" : "_ver";

   res.Form("%s%s%d", clname, suffix, version);

   if ((res.Length() <= maxlen) && !HasTable(res.Data()))
      return res;

   TString scnt;

   Int_t len = strlen(clname);
   Int_t cnt = version;
   if (cnt>100) cnt = 0; // do not start with the biggest values

   do {
      scnt.Form("%d%s",cnt, suffix);
      Int_t numlen = scnt.Length();
      if (numlen>=maxlen-2) break;

      res = clname;

      if (len + numlen > maxlen)
         res.Resize(maxlen - numlen);

      res+=scnt;

      if (!HasTable(res.Data())) return res;

      cnt++;

   } while (cnt<10000);

   Error("DefineTableName","Cannot produce table name for class %s ver %d", clname, version);
   res.Form("%s%s%d", clname, suffix, version);

   return res;
}

//______________________________________________________________________________
Bool_t TSQLFile::HasTable(const char* name)
{
   // test if table name exists

   if (fSQLClassInfos==0) return kFALSE;

   TIter iter(fSQLClassInfos);
   TSQLClassInfo* info = 0;
   while ((info = (TSQLClassInfo*) iter()) !=0 ) {
      if (strcmp(info->GetClassTableName(), name)==0) return kTRUE;
      if (strcmp(info->GetRawTableName(), name)==0) return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
TSQLClassInfo* TSQLFile::RequestSQLClassInfo(const TClass* cl)
{
   // search in database tables for specified class and return TSQLClassInfo object

   return RequestSQLClassInfo(cl->GetName(), cl->GetClassVersion());
}

//______________________________________________________________________________
void TSQLFile::ReadSQLClassInfos()
{
   // Read all class infos from IdsTable

   if (fSQL==0) return;

   fIdsTableExists = SQLTestTable(sqlio::IdsTable);

   if (!fIdsTableExists) return;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();

   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s = %d ORDER BY %s%s%s",
                quote, sqlio::IdsTable, quote,
                quote, sqlio::IT_Type, quote, TSQLStructure::kIdTable,
                quote, sqlio::IT_TableID, quote);

   TSQLResult* res = SQLQuery(sqlcmd.Data(), 1);

   TSQLRow* row = 0;

   if (res!=0)
      while ((row = res->Next())!=0) {
         Long64_t tableid = sqlio::atol64(row->GetField(0));
         Int_t version = atoi(row->GetField(1));

         const char* classname = row->GetField(3);
         const char* classtable = row->GetField(4);

         TSQLClassInfo* info = new TSQLClassInfo(tableid, classname, version);
         info->SetClassTableName(classtable);

         if (fSQLClassInfos==0) fSQLClassInfos = new TList;
         fSQLClassInfos->Add(info);

         delete row;
      }
   delete res;


   TIter next(fSQLClassInfos);
   TSQLClassInfo* info = 0;

   while ((info = (TSQLClassInfo*) next()) != 0) {
      sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s = %lld ORDER BY %s%s%s",
                   quote, sqlio::IdsTable, quote,
                   quote, sqlio::IT_TableID, quote, info->GetClassId(),
                   quote, sqlio::IT_SubID, quote);
      res = SQLQuery(sqlcmd.Data(), 1);

      TObjArray* cols = 0;

      if (res!=0)
         while ((row = res->Next())!=0) {

            Int_t typ = atoi(row->GetField(2));

            const char* fullname = row->GetField(3);
            const char* sqlname = row->GetField(4);
            const char* info2 = row->GetField(5);

            if (typ==TSQLStructure::kIdColumn) {
                if (cols==0) cols = new TObjArray;
                cols->Add(new TSQLClassColumnInfo(fullname, sqlname, info2));
            }

            delete row;
         }

      delete res;

      info->SetColumns(cols);
   }

   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s = %d ORDER BY %s%s%s",
                quote, sqlio::IdsTable, quote,
                quote, sqlio::IT_Type, quote, TSQLStructure::kIdRawTable,
                quote, sqlio::IT_TableID, quote);

   res = SQLQuery(sqlcmd.Data(), 1);

   if (res!=0)
      while ((row = res->Next())!=0) {
         Long64_t tableid = sqlio::atol64(row->GetField(0));
         Int_t version = atoi(row->GetField(1));

         const char* classname = row->GetField(3);
         const char* rawtable = row->GetField(4);

         TSQLClassInfo* info2 = FindSQLClassInfo(classname, version);

         if (info2==0) {
            info2 = new TSQLClassInfo(tableid, classname, version);

            if (fSQLClassInfos==0) fSQLClassInfos = new TList;
            fSQLClassInfos->Add(info2);
         }

         info2->SetRawTableName(rawtable);
         info2->SetRawExist(kTRUE);

         delete row;
      }

   delete res;
}


//______________________________________________________________________________
void TSQLFile::AddIdEntry(Long64_t tableid, Int_t subid, Int_t type,
                          const char* name, const char* sqlname, const char* info)
{
   // Add entry into IdsTable, where all tables names and columns names are listed

   if ((fSQL==0) || !IsWritable()) return;

   TString sqlcmd;
   const char* valuequote = SQLValueQuote();
   const char* quote = SQLIdentifierQuote();

   if (!fIdsTableExists) {

      if (SQLTestTable(sqlio::IdsTable)) {
         sqlcmd.Form("DROP TABLE %s%s%s", quote, sqlio::IdsTable, quote);
         SQLQuery(sqlcmd.Data());
      }

      sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s)",
                  quote, sqlio::IdsTable, quote,
                  quote, sqlio::IT_TableID, quote, SQLIntType(),
                  quote, sqlio::IT_SubID, quote, SQLIntType(),
                  quote, sqlio::IT_Type, quote, SQLIntType(),
                  quote, sqlio::IT_FullName, quote, SQLSmallTextType(),
                  quote, sqlio::IT_SQLName, quote, SQLSmallTextType(),
                  quote, sqlio::IT_Info, quote, SQLSmallTextType());
      if ((fTablesType.Length()>0) && IsMySQL()) {
         sqlcmd +=" TYPE=";
         sqlcmd += fTablesType;
      }
      SQLQuery(sqlcmd.Data());

      fIdsTableExists = kTRUE;
   }

   sqlcmd.Form("INSERT INTO %s%s%s VALUES (%lld, %d, %d, %s%s%s, %s%s%s, %s%s%s)",
               quote, sqlio::IdsTable, quote,
               tableid, subid, type,
               valuequote, name, valuequote,
               valuequote, sqlname, valuequote,
               valuequote, info, valuequote);

   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
Bool_t TSQLFile::CreateClassTable(TSQLClassInfo* sqlinfo, TObjArray* colinfos)
{
   // Create normal class table if required

   if (sqlinfo==0) return kFALSE;

   // this is normal situation, when no extra column infos was created when not necessary
   if (colinfos==0) return sqlinfo->IsClassTableExist();

   if (sqlinfo->IsClassTableExist()) {
      if (colinfos!=0) {
         colinfos->Delete();
         delete colinfos;
         //Error("CreateClassTable","Why colinfos for table %s", sqlinfo->GetClassTableName());
      }
      return kTRUE;
   }

   if (gDebug>2)
      Info("CreateClassTable", "cl:%s", sqlinfo->GetName());

   const char* quote = SQLIdentifierQuote();

   AddIdEntry(sqlinfo->GetClassId(),
              sqlinfo->GetClassVersion(),
              TSQLStructure::kIdTable,
              sqlinfo->GetName(),
              sqlinfo->GetClassTableName(),
              "Main class table");

   TString sqlcmd;
   sqlcmd.Form("CREATE TABLE %s%s%s (",
                quote, sqlinfo->GetClassTableName(), quote);

   TIter iter(colinfos);
   TSQLClassColumnInfo* col;
   Bool_t first = kTRUE;
   Bool_t forcequote = IsOracle();
   Int_t colid = 0;
   while ((col=(TSQLClassColumnInfo*)iter())!=0) {
      if (!first) sqlcmd+=", "; else first = false;

      const char* colname = col->GetSQLName();
      if ((strpbrk(colname,"[:.]<>")!=0) || forcequote) {
         sqlcmd += quote;
         sqlcmd += colname;
         sqlcmd += quote;
         sqlcmd += " ";
      } else {
         sqlcmd += colname,
         sqlcmd += " ";
      }

      sqlcmd += col->GetSQLType();

      AddIdEntry(sqlinfo->GetClassId(),
                 colid++,
                 TSQLStructure::kIdColumn,
                 col->GetName(),
                 col->GetSQLName(),
                 col->GetSQLType());
   }
   sqlcmd += ")";

   if ((fTablesType.Length()>0)  && IsMySQL()) {
      sqlcmd +=" TYPE=";
      sqlcmd += fTablesType;
   }

   SQLQuery(sqlcmd.Data());

   sqlinfo->SetColumns(colinfos);

   if (GetUseIndexes()>kIndexesBasic) {

      TString indxname = sqlinfo->GetClassTableName();
      indxname.ReplaceAll("_ver","_i1x");

      sqlcmd.Form("CREATE UNIQUE INDEX %s%s_I1%s ON %s%s%s (%s%s%s)",
                  quote, indxname.Data(), quote,
                  quote, sqlinfo->GetClassTableName(), quote,
                  quote, SQLObjectIdColumn(), quote);
      SQLQuery(sqlcmd.Data());
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TSQLFile::CreateRawTable(TSQLClassInfo* sqlinfo)
{
   //create the raw table
   if (sqlinfo==0) return kFALSE;

   if (sqlinfo->IsRawTableExist()) return kTRUE;

   const char* quote = SQLIdentifierQuote();

   if (gDebug>2)
      Info("CreateRawTable", "%s", sqlinfo->GetName());

   TString sqlcmd;

   sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s %s, %s %s)",
               quote, sqlinfo->GetRawTableName(), quote,
               quote, SQLObjectIdColumn(), quote, SQLIntType(),
               quote, SQLRawIdColumn(), quote, SQLIntType(),
               sqlio::BT_Field, SQLSmallTextType(),
               sqlio::BT_Value, SQLSmallTextType());

   if ((fTablesType.Length()>0) && IsMySQL()) {
      sqlcmd +=" TYPE=";
      sqlcmd += fTablesType;
   }

   SQLQuery(sqlcmd.Data());
   sqlinfo->SetRawExist(kTRUE);

   if (GetUseIndexes()>kIndexesClass) {
      TString indxname = sqlinfo->GetClassTableName();
      indxname.ReplaceAll("_ver","_i2x");

      sqlcmd.Form("CREATE UNIQUE INDEX %s%s_I2%s ON %s%s%s (%s%s%s, %s%s%s)",
                  quote, indxname.Data(), quote,
                  quote, sqlinfo->GetRawTableName(), quote,
                  quote, SQLObjectIdColumn(), quote,
                  quote, SQLRawIdColumn(), quote);
      SQLQuery(sqlcmd.Data());
   }

   AddIdEntry(sqlinfo->GetClassId(),
              sqlinfo->GetClassVersion(),
              TSQLStructure::kIdRawTable,
              sqlinfo->GetName(),
              sqlinfo->GetRawTableName(),
              "Raw data class table");

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

   TString sqlcmd;
   sqlcmd.Form("CREATE TABLE %s (%s%s%s %s, %s%s%s %s, %s %s)",
               sqlio::StringsTable,
               quote, SQLObjectIdColumn(), quote, SQLIntType(),
               quote, SQLStrIdColumn(), quote, SQLIntType(),
               sqlio::ST_Value, SQLBigTextType());

   if (fTablesType.Length()>0) {
      sqlcmd +=" TYPE=";
      sqlcmd += fTablesType;
   }

   SQLQuery(sqlcmd.Data());

   return kTRUE;
}

//______________________________________________________________________________
TString TSQLFile::CodeLongString(Long64_t objid, Int_t strid)
{
   // produces id which will be placed in column instead of string itself
   TString res;
   res.Form("%s %lld %s %d %s", sqlio::LongStrPrefix, objid, sqlio::LongStrPrefix, strid, sqlio::LongStrPrefix);
   return res;
}

//______________________________________________________________________________
Int_t TSQLFile::IsLongStringCode(Long64_t objid, const char* value)
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

   Long64_t objid2 = sqlio::atol64(s_objid.Data());
   if (objid2!=objid) return 0;

   return atoi(s_strid.Data());
}

//______________________________________________________________________________
Bool_t TSQLFile::GetLongString(Long64_t objid, Int_t strid, TString& value)
{
   // returns value of string, extracted from special table,
   // where long strings are stored

   if (!SQLTestTable(sqlio::StringsTable)) return kFALSE;

   TString cmd;
   const char* quote = SQLIdentifierQuote();
   cmd.Form("SELECT %s FROM %s%s%s WHERE %s%s%s=%lld AND %s%s%s=%d",
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
Long64_t TSQLFile::VerifyObjectTable()
{
   // Checks that objects table is exists
   // If not, table will be created
   // Returns maximum value for existing objects id

   if (fSQL==0) return -1;

   Long64_t maxid = -1;

   if (gDebug>2)
      Info("VerifyObjectTable", "Checks if object table is there");

   if (SQLTestTable(sqlio::ObjectsTable))
      maxid = SQLMaximumValue(sqlio::ObjectsTable, SQLObjectIdColumn());
   else {
      TString sqlcmd;
      const char* quote = SQLIdentifierQuote();
      sqlcmd.Form("CREATE TABLE %s%s%s (%s%s%s %s, %s%s%s %s, %s%s%s %s, %s%s%s %s)",
                  quote, sqlio::ObjectsTable, quote,
                  quote, SQLKeyIdColumn(), quote, SQLIntType(),
                  quote, SQLObjectIdColumn(), quote, SQLIntType(),
                  quote, sqlio::OT_Class, quote, SQLSmallTextType(),
                  quote, sqlio::OT_Version, quote, SQLIntType());

      if ((fTablesType.Length()>0) && IsMySQL()) {
         sqlcmd +=" TYPE=";
         sqlcmd += fTablesType;
      }

      SQLQuery(sqlcmd.Data());

      if (GetUseIndexes()>kIndexesNone) {
         sqlcmd.Form("CREATE UNIQUE INDEX %s%s%s ON %s%s%s (%s%s%s)",
                      quote, sqlio::ObjectsTableIndex, quote,
                      quote, sqlio::ObjectsTable, quote,
                      quote, SQLObjectIdColumn(), quote);
         SQLQuery(sqlcmd.Data());
      }
   }

   return maxid;
}

//______________________________________________________________________________
Bool_t TSQLFile::SQLObjectInfo(Long64_t objid, TString& clname, Version_t &version)
{
   // Read from objects table data for specified objectid

   if (fSQL==0) return kFALSE;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT %s%s%s, %s%s%s FROM %s%s%s WHERE %s%s%s=%lld",
               quote, sqlio::OT_Class, quote,
               quote, sqlio::OT_Version, quote,
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
TObjArray* TSQLFile::SQLObjectsInfo(Long64_t keyid)
{
// Produce array of TSQLObjectInfo objects for all objects, belong to that key
// Array should be deleted by calling function afterwards
   if (fSQL==0) return 0;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT %s%s%s, %s%s%s, %s%s%s FROM %s%s%s WHERE %s%s%s=%lld ORDER BY %s%s%s",
               quote, SQLObjectIdColumn(), quote,
               quote, sqlio::OT_Class, quote,
               quote, sqlio::OT_Version, quote,
               quote, sqlio::ObjectsTable, quote,
               quote, SQLKeyIdColumn(), quote, keyid,
               quote, SQLObjectIdColumn(), quote);

   TObjArray* arr = 0;

   if (fLogFile!=0)
      *fLogFile << sqlcmd << endl;
   if (gDebug>2) Info("SQLObjectsInfo", "%s", sqlcmd.Data());
   fQuerisCounter++;

   TSQLStatement* stmt = SQLStatement(sqlcmd.Data(), 1000);

   if (stmt!=0) {
      stmt->Process();
      stmt->StoreResult();

      while (stmt->NextResultRow()) {
         Long64_t objid = stmt->GetLong64(0);
         const char* clname = stmt->GetString(1);
         Int_t version = stmt->GetInt(2);

         TSQLObjectInfo* info = new TSQLObjectInfo(objid, clname, version);
         if (arr==0) arr = new TObjArray();
         arr->Add(info);
      }

      delete stmt;
      return arr;
   }

   TSQLResult* res = SQLQuery(sqlcmd.Data(), 1);
   if (res==0) return 0;

   TSQLRow* row = 0;
   while ((row = res->Next()) != 0) {
      Long64_t objid = atoi(row->GetField(0));
      const char* clname = row->GetField(1);
      Int_t version = atoi(row->GetField(2));

      TSQLObjectInfo* info = new TSQLObjectInfo(objid, clname, version);
      if (arr==0) arr = new TObjArray();
      arr->Add(info);

      delete row;
   }
   delete res;
   return arr;
}

//______________________________________________________________________________
TSQLResult* TSQLFile::GetNormalClassData(Long64_t objid, TSQLClassInfo* sqlinfo)
{
// Method return request result for specified objid from normal classtable

   if (!sqlinfo->IsClassTableExist()) return 0;
   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s=%lld",
               quote, sqlinfo->GetClassTableName(), quote,
               quote, SQLObjectIdColumn(), quote, objid);
   return SQLQuery(sqlcmd.Data(), 2);
}

//______________________________________________________________________________
TSQLResult* TSQLFile::GetNormalClassDataAll(Long64_t minobjid, Long64_t maxobjid, TSQLClassInfo* sqlinfo)
{
   // return data for several objects from the range from normal class table

   if (!sqlinfo->IsClassTableExist()) return 0;
   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT * FROM %s%s%s WHERE %s%s%s BETWEEN %lld AND %lld ORDER BY %s%s%s",
               quote, sqlinfo->GetClassTableName(), quote,
               quote, SQLObjectIdColumn(), quote, minobjid, maxobjid,
               quote, SQLObjectIdColumn(), quote);
   return SQLQuery(sqlcmd.Data(), 2);
}

//______________________________________________________________________________
TSQLResult* TSQLFile::GetBlobClassData(Long64_t objid, TSQLClassInfo* sqlinfo)
{
//  Method return request results for specified objid from _streamer_ classtable

   if (!sqlinfo->IsRawTableExist()) return 0;
   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT %s, %s FROM %s%s%s WHERE %s%s%s=%lld ORDER BY %s%s%s",
               sqlio::BT_Field, sqlio::BT_Value,
               quote, sqlinfo->GetRawTableName(), quote,
               quote, SQLObjectIdColumn(), quote, objid,
               quote, SQLRawIdColumn(), quote);
   return SQLQuery(sqlcmd.Data(), 2);
}

//______________________________________________________________________________
TSQLStatement* TSQLFile::GetBlobClassDataStmt(Long64_t objid, TSQLClassInfo* sqlinfo)
{
//  Method return request results for specified objid from _streamer_ classtable
//  Data returned in form of statement, where direct access to values are possible

   if (!sqlinfo->IsRawTableExist()) return 0;

   TString sqlcmd;
   const char* quote = SQLIdentifierQuote();
   sqlcmd.Form("SELECT %s, %s FROM %s%s%s WHERE %s%s%s=%lld ORDER BY %s%s%s",
               sqlio::BT_Field, sqlio::BT_Value,
               quote, sqlinfo->GetRawTableName(), quote,
               quote, SQLObjectIdColumn(), quote, objid,
               quote, SQLRawIdColumn(), quote);

   if (fLogFile!=0)
      *fLogFile << sqlcmd << endl;
   if (gDebug>2) Info("BuildStatement", "%s", sqlcmd.Data());
   fQuerisCounter++;

   TSQLStatement* stmt = SQLStatement(sqlcmd.Data(), 1000);
   if (stmt==0) return 0;

   stmt->Process();

   stmt->StoreResult();

   return stmt;
}

//______________________________________________________________________________
Long64_t TSQLFile::StoreObjectInTables(Long64_t keyid, const void* obj, const TClass* cl)
{
   // Store object in database. Return stored object id or -1 if error

   if (fSQL==0) return -1;

   Long64_t objid = VerifyObjectTable();
   if (objid<=0) objid = 1; else objid++;

   TBufferSQL2 buffer(TBuffer::kWrite, this);

   TSQLStructure* s = buffer.SqlWriteAny(obj, cl, objid);

   if ((buffer.GetErrorFlag()>0) && s) {
      Error("StoreObjectInTables","Cannot convert object data to TSQLStructure");
      objid = -1;
   } else {
      TObjArray cmds;
      // here tables may be already created, therefore
      // it should be protected by transactions operations
      if (s && !s->ConvertToTables(this, keyid, &cmds)) {
         Error("StoreObjectInTables","Cannot convert to SQL statements");
         objid = -1;
      } else {
         Bool_t needcommit = kFALSE;

         if (GetUseTransactions()==kTransactionsAuto) {
            SQLStartTransaction();
            needcommit = kTRUE;
         }

         if (!SQLApplyCommands(&cmds)) {
            Error("StoreObject","Cannot correctly store object data in database");
            objid = -1;
            if (needcommit) SQLRollback();
         } else {
            if (needcommit) SQLCommit();
         }
      }
      cmds.Delete();
   }

   return objid;
}

//______________________________________________________________________________
const char* TSQLFile::SQLCompatibleType(Int_t typ) const
{
   // returns sql type name which is most closer to ROOT basic type
   // typ should be from TVirtualStreamerInfo:: constansts like TVirtualStreamerInfo::kInt

   return (typ<0) || (typ>18) ? 0 : fBasicTypes[typ];
}

//______________________________________________________________________________
const char* TSQLFile::SQLIntType() const
{
   // return SQL integer type

   return SQLCompatibleType(TVirtualStreamerInfo::kInt);
}

//______________________________________________________________________________
Long64_t TSQLFile::DirCreateEntry(TDirectory* dir)
{
   // Create entry for directory in database

   TDirectory* mother = dir->GetMotherDir();
   if (mother==0) mother = this;

   // key will be added to mother directory
   TKeySQL* key = new TKeySQL(mother, dir, dir->GetName(), dir->GetTitle());

   return key->GetDBKeyId();
}

//______________________________________________________________________________
Int_t TSQLFile::DirReadKeys(TDirectory* dir)
{
   // Read directory list of keys from database

   // First delete all old keys
   dir->GetListOfKeys()->Delete();

   if (gDebug>2)
      Info("DirReadKeys","dir = %s id = %lld", dir->GetName(), dir->GetSeekDir());

   return StreamKeysForDirectory(dir, kFALSE);
}

//______________________________________________________________________________
void TSQLFile::DirWriteKeys(TDirectory* dir)
{
   // Write directory keys list to database

   StreamKeysForDirectory(dir, kTRUE);
}

//______________________________________________________________________________
void TSQLFile::DirWriteHeader(TDirectory* dir)
{
   // Update dir header in the file

   TSQLClassInfo* sqlinfo = FindSQLClassInfo("TDirectory",TDirectoryFile::Class()->GetClassVersion());
   if (sqlinfo==0) return;

   // try to identify key with data for our directory
   TKeySQL* key = FindSQLKey(dir->GetMotherDir(), dir->GetSeekDir());
   if (key==0) return;

   const char* valuequote = SQLValueQuote();
   const char* quote = SQLIdentifierQuote();

   TString timeC = fDatimeC.AsSQLString();
   TSQLStructure::AddStrBrackets(timeC, valuequote);

   TString timeM = fDatimeM.AsSQLString();
   TSQLStructure::AddStrBrackets(timeM, valuequote);

   TString uuid = dir->GetUUID().AsString();
   TSQLStructure::AddStrBrackets(uuid, valuequote);

   TString sqlcmd;

   TString col1name = "CreateTime";
   TString col2name = "ModifyTime";
   TString col3name = "UUID";
   if (GetUseSuffixes()) {
      col1name+=sqlio::StrSuffix;
      col2name+=sqlio::StrSuffix;
      col3name+=sqlio::StrSuffix;
   }

   sqlcmd.Form("UPDATE %s%s%s SET %s%s%s=%s, %s%s%s=%s, %s%s%s=%s WHERE %s%s%s=%lld",
                quote, sqlinfo->GetClassTableName(), quote,
                quote, col1name.Data(), quote, timeC.Data(),
                quote, col2name.Data(), quote, timeM.Data(),
                quote, col3name.Data(), quote, uuid.Data(),
                quote, SQLObjectIdColumn(), quote, key->GetDBObjId());

   SQLQuery(sqlcmd.Data());
}

//______________________________________________________________________________
void TSQLFile::Streamer(TBuffer &b)
{
   // streamer for TSQLFile class
   // stores only data for TDirectory


   TString sbuf;

   if (b.IsReading()) {
      Version_t R__v = b.ReadVersion(0, 0);
      b.ClassBegin(TSQLFile::Class(), R__v);

      b.ClassMember("CreateTime","TString");
      sbuf.Streamer(b);
      TDatime timeC(sbuf.Data());
      fDatimeC = timeC;

      b.ClassMember("ModifyTime","TString");
      sbuf.Streamer(b);
      TDatime timeM(sbuf.Data());
      fDatimeM = timeM;

      b.ClassMember("UUID","TString");
      sbuf.Streamer(b);
      TUUID id(sbuf.Data());
      fUUID = id;

      b.ClassEnd(TSQLFile::Class());
   } else {

      b.WriteVersion(TSQLFile::Class());

      b.ClassBegin(TSQLFile::Class());

      b.ClassMember("CreateTime","TString");
      sbuf = fDatimeC.AsSQLString();
      sbuf.Streamer(b);

      b.ClassMember("ModifyTime","TString");
      fDatimeM.Set();
      sbuf = fDatimeM.AsSQLString();
      sbuf.Streamer(b);

      b.ClassMember("UUID","TString");
      sbuf = fUUID.AsString();
      sbuf.Streamer(b);

      b.ClassEnd(TSQLFile::Class());
   }
}
