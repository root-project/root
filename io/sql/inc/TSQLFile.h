// @(#)root/sql:$Id$
// Author: Sergey Linev  20/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLFile
#define ROOT_TSQLFile

#include "TFile.h"
#include <stdlib.h>

class TList;
class TStreamerElement;
class TVirtualStreamerInfo;

class TSQLServer;
class TSQLStatement;
class TSQLResult;
class TSQLRow;
class TKeySQL;
class TBufferSQL2;
class TSQLClassInfo;

class TSQLFile final : public TFile {

   friend class TBufferSQL2;
   friend class TKeySQL;
   friend class TSQLStructure;
   friend class TSQLTableData;
   friend class TSqlRegistry;
   friend class TSqlRawBuffer;
   friend class TSqlCmdsBuffer;

protected:
   enum ELockingKinds { kLockFree = 0, kLockBusy = 1 };

   // Interface to basic system I/O routines, suppressed
   Int_t SysOpen(const char *, Int_t, UInt_t) final { return 0; }
   Int_t SysClose(Int_t)  final { return 0; }
   Int_t SysRead(Int_t, void *, Int_t)  final { return 0; }
   Int_t SysWrite(Int_t, const void *, Int_t)  final { return 0; }
   Long64_t SysSeek(Int_t, Long64_t, Int_t)  final { return 0; }
   Int_t SysStat(Int_t, Long_t *, Long64_t *, Long_t *, Long_t *) final { return 0; }
   Int_t SysSync(Int_t)  final { return 0; }

   // Overwrite methods for directory I/O
   Long64_t DirCreateEntry(TDirectory *) final;
   Int_t DirReadKeys(TDirectory *) final;
   void DirWriteKeys(TDirectory *) final;
   void DirWriteHeader(TDirectory *) final;

   InfoListRet GetStreamerInfoListImpl(bool) final;

   // functions to manipulate basic tables (Configurations, Objects, Keys) in database
   void SaveToDatabase();
   Bool_t ReadConfigurations();
   Bool_t IsTablesExists();
   void InitSqlDatabase(Bool_t create);
   void CreateBasicTables();
   void IncrementModifyCounter();
   void SetLocking(Int_t mode);
   Int_t GetLocking();

   // function for read/write access infos
   Bool_t IsWriteAccess();
   Bool_t IsReadAccess();

   // generic sql functions
   TSQLResult *SQLQuery(const char *cmd, Int_t flag = 0, Bool_t *res = 0);
   Bool_t SQLCanStatement();
   TSQLStatement *SQLStatement(const char *cmd, Int_t bufsize = 1000);
   void SQLDeleteStatement(TSQLStatement *stmt);
   Bool_t SQLApplyCommands(TObjArray *cmds);
   Bool_t SQLTestTable(const char *tablename);
   Long64_t SQLMaximumValue(const char *tablename, const char *columnname);
   void SQLDeleteAllTables();
   Bool_t SQLStartTransaction();
   Bool_t SQLCommit();
   Bool_t SQLRollback();
   Int_t SQLMaxIdentifierLength();

   // operation with keys structures in database
   void DeleteKeyFromDB(Long64_t keyid);
   Bool_t WriteKeyData(TKeySQL *key);
   Bool_t UpdateKeyData(TKeySQL *key);
   TKeySQL *FindSQLKey(TDirectory *dir, Long64_t keyid);
   Long64_t DefineNextKeyId();
   Int_t StreamKeysForDirectory(TDirectory *dir, Bool_t doupdate, Long64_t specialkeyid = -1, TKeySQL **specialkey = 0);

   // handling SQL class info structures
   TSQLClassInfo *FindSQLClassInfo(const char *clname, Int_t version);
   TSQLClassInfo *FindSQLClassInfo(const TClass *cl);
   TSQLClassInfo *RequestSQLClassInfo(const char *clname, Int_t version);
   TSQLClassInfo *RequestSQLClassInfo(const TClass *cl);
   Bool_t CreateClassTable(TSQLClassInfo *sqlinfo, TObjArray *colinfos);
   Bool_t CreateRawTable(TSQLClassInfo *sqlinfo);

   Bool_t ProduceClassSelectQuery(TVirtualStreamerInfo *info, TSQLClassInfo *sqlinfo, TString &columns, TString &tables,
                                  Int_t &tablecnt);
   void AddIdEntry(Long64_t tableid, Int_t subid, Int_t type, const char *name, const char *sqlname, const char *info);
   void ReadSQLClassInfos();
   TString DefineTableName(const char *clname, Int_t version, Bool_t rawtable);
   Bool_t HasTable(const char *name);

   // operations with long string table
   TString CodeLongString(Long64_t objid, Int_t strid);
   Int_t IsLongStringCode(Long64_t objid, const char *value);
   Bool_t VerifyLongStringTable();
   Bool_t GetLongString(Long64_t objid, Int_t strid, TString &value);

   // operation with object tables in database
   Long64_t VerifyObjectTable();
   Bool_t SQLObjectInfo(Long64_t objid, TString &clname, Version_t &version);
   TObjArray *SQLObjectsInfo(Long64_t keyid);
   TSQLResult *GetNormalClassData(Long64_t objid, TSQLClassInfo *sqlinfo);
   TSQLResult *GetNormalClassDataAll(Long64_t minobjid, Long64_t maxobjid, TSQLClassInfo *sqlinfo);
   TSQLResult *GetBlobClassData(Long64_t objid, TSQLClassInfo *sqlinfo);
   TSQLStatement *GetBlobClassDataStmt(Long64_t objid, TSQLClassInfo *sqlinfo);
   Long64_t StoreObjectInTables(Long64_t keyid, const void *obj, const TClass *cl);
   Bool_t WriteSpecialObject(Long64_t keyid, TObject *obj, const char *name, const char *title);
   TObject *ReadSpecialObject(Long64_t keyid, TObject *obj = 0);

   // sql specific types
   const char *SQLCompatibleType(Int_t typ) const;
   const char *SQLIntType() const;
   const char *SQLSmallTextType() const { return fOtherTypes[0]; }
   Int_t SQLSmallTextTypeLimit() const { return atoi(fOtherTypes[1]); }
   const char *SQLBigTextType() const { return fOtherTypes[2]; }
   const char *SQLDatetimeType() const { return fOtherTypes[3]; }
   const char *SQLIdentifierQuote() const { return fOtherTypes[4]; }
   const char *SQLDirIdColumn() const { return fOtherTypes[5]; }
   const char *SQLKeyIdColumn() const { return fOtherTypes[6]; }
   const char *SQLObjectIdColumn() const { return fOtherTypes[7]; }
   const char *SQLRawIdColumn() const { return fOtherTypes[8]; }
   const char *SQLStrIdColumn() const { return fOtherTypes[9]; }
   const char *SQLNameSeparator() const { return fOtherTypes[10]; }
   const char *SQLValueQuote() const { return fOtherTypes[11]; }
   const char *SQLDefaultTableType() const { return fOtherTypes[12]; }

   TSQLServer *fSQL; ///<! interface to SQL database

   TList *fSQLClassInfos; ///<! list of SQL class infos

   Bool_t fUseSuffixes;     ///<! use suffixes in column names like fValue:Int_t or fObject:pointer
   Int_t fSQLIOversion;     ///<! version of SQL I/O which is stored in configurations
   Int_t fArrayLimit;       ///<! limit for array size. when array bigger, its content converted to raw format
   Bool_t fCanChangeConfig; ///<! variable indicates can be basic configuration changed or not
   TString fTablesType;     ///<! type, used in CREATE TABLE statements
   Int_t fUseTransactions;  ///<! use transaction statements for writing data into the tables
   Int_t fUseIndexes;       ///<! use indexes for tables: 0 - off, 1 - only for basic tables, 2  + normal class tables, 3 - all tables
   Int_t fModifyCounter;    ///<! indicates how many changes was done with database tables
   Int_t fQuerisCounter;    ///<! how many query was applied

   const char **fBasicTypes; ///<! pointer on list of basic types specific for currently connected SQL server
   const char **fOtherTypes; ///<! pointer on list of other SQL types like TEXT or blob

   TString fUserName; ///<! user name, used to access objects from database

   std::ofstream *fLogFile; ///<! log file with SQL statements

   Bool_t fIdsTableExists; ///<! indicate if IdsTable exists
   Int_t fStmtCounter;     ///<! count numbers of active statements

private:
   TSQLFile(const TSQLFile &);            //Files cannot be copied - not implemented
   void operator=(const TSQLFile &);      //Files cannot be copied - not implemented

public:
   enum ETransactionKinds { kTransactionsOff = 0, kTransactionsAuto = 1, kTransactionsUser = 2 };

   enum EIndexesKinds { kIndexesNone = 0, kIndexesBasic = 1, kIndexesClass = 2, kIndexesAll = 3 };

   TSQLFile();
   TSQLFile(const char *dbname, Option_t *option = "read", const char *user = "user", const char *pass = "pass");
   virtual ~TSQLFile();

   // configuration of SQL
   Bool_t GetUseSuffixes() const { return fUseSuffixes; }
   void SetUseSuffixes(Bool_t on = kTRUE);
   Int_t GetArrayLimit() const { return fArrayLimit; }
   void SetArrayLimit(Int_t limit = 20);
   void SkipArrayLimit() { SetArrayLimit(-1); }
   void SetTablesType(const char *table_type);
   const char *GetTablesType() const { return fTablesType.Data(); }
   void SetUseTransactions(Int_t mode = kTransactionsAuto);
   Int_t GetUseTransactions() const { return fUseTransactions; }
   void SetUseIndexes(Int_t use_type = kIndexesBasic);
   Int_t GetUseIndexes() const { return fUseIndexes; }
   Int_t GetQuerisCounter() const { return fQuerisCounter; }
   Int_t GetIOVersion() const { return fSQLIOversion; }

   TString MakeSelectQuery(TClass *cl);
   Bool_t StartTransaction();
   Bool_t Commit();
   Bool_t Rollback();

   // log file for SQL statements
   void StartLogFile(const char *fname); // *MENU*
   void StopLogFile();                   // *MENU*

   void Close(Option_t *option = "") final; // *MENU*
   TKey *CreateKey(TDirectory *mother, const TObject *obj, const char *name, Int_t bufsize)  final;
   TKey *CreateKey(TDirectory *mother, const void *obj, const TClass *cl, const char *name, Int_t bufsize) final;
   void DrawMap(const char * = "*", Option_t * = "") final {}
   void FillBuffer(char *&) final {}
   void Flush() final {}

   Long64_t GetEND() const  final { return 0; }
   Int_t GetErrno() const  final { return 0; }
   void ResetErrno() const  final {}

   const char *GetDataBaseName() const;
   Int_t GetNfree() const final { return 0; }
   Int_t GetNbytesInfo() const final{ return 0; }
   Int_t GetNbytesFree() const final { return 0; }
   Long64_t GetSeekFree() const final { return 0; }
   Long64_t GetSeekInfo() const final { return 0; }
   Long64_t GetSize() const final { return 0; }

   Bool_t IsOpen() const final;
   Bool_t IsMySQL() const;
   Bool_t IsOracle() const;
   Bool_t IsODBC() const;

   void MakeFree(Long64_t, Long64_t) final {}
   void MakeProject(const char *, const char * = "*", Option_t * = "new") final {} // *MENU*
   void Map(Option_t *) final {}                                                   //
   void Map() final {}                                                             //
   void Paint(Option_t * = "") final {}
   void Print(Option_t * = "") const final {}
   Bool_t ReadBuffer(char *, Int_t) final { return kFALSE; }
   Bool_t ReadBuffer(char *, Long64_t, Int_t) final { return kFALSE; }
   void ReadFree() final {}
   Int_t Recover() final { return 0; }
   Int_t ReOpen(Option_t *mode) final;
   void Seek(Long64_t, ERelativeTo = kBeg) final {}

   void SetEND(Long64_t) final {}
   Int_t Sizeof() const final { return 0; }

   Bool_t WriteBuffer(const char *, Int_t) final { return kFALSE; }
   Int_t Write(const char * = nullptr, Int_t = 0, Int_t = 0) final { return 0; }
   Int_t Write(const char * = nullptr, Int_t = 0, Int_t = 0) const final { return 0; }
   void WriteFree() final {}
   void WriteHeader() final;
   void WriteStreamerInfo() final;

   ClassDefOverride(TSQLFile, 1) // ROOT TFile interface to SQL database
};

#endif
