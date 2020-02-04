// @(#)root/odbc:$Id$
// Author: Sergey Linev   6/02/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TODBCServer
#define ROOT_TODBCServer

#include "TSQLServer.h"

#ifdef __CLING__
typedef void * SQLHENV;
typedef void * SQLHDBC;
typedef short  SQLRETURN;
#else
#ifdef WIN32
#include "windows.h"
#endif
#include <sql.h>
#endif

class TList;

class TODBCServer : public TSQLServer {

private:
   SQLHENV   fHenv;
   SQLHDBC   fHdbc;
   TString   fServerInfo;        // string with DBMS name and version like MySQL 4.1.11 or Oracle 10.01.0030
   TString   fUserId;

   Bool_t ExtractErrors(SQLRETURN retcode, const char* method);

   Bool_t EndTransaction(Bool_t commit);

   static TList* ListData(Bool_t isdrivers);

public:
   TODBCServer(const char* db, const char *uid, const char *pw);
   virtual ~TODBCServer();

   static TList* GetDrivers();
   static void PrintDrivers();
   static TList* GetDataSources();
   static void PrintDataSources();

   void        Close(Option_t *opt="") final;
   TSQLResult *Query(const char *sql) final;
   Bool_t      Exec(const char* sql) final;
   TSQLStatement *Statement(const char *sql, Int_t = 100) final;
   Bool_t      HasStatement() const final { return kTRUE; }
   Int_t       SelectDataBase(const char *dbname) final;
   TSQLResult *GetDataBases(const char *wild = nullptr) final;
   TSQLResult *GetTables(const char *dbname, const char *wild = nullptr) final;
   TList      *GetTablesList(const char* wild = nullptr) final;
   TSQLTableInfo* GetTableInfo(const char* tablename) final;
   TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = nullptr) final;
   Int_t       GetMaxIdentifierLength() final;
   Int_t       CreateDataBase(const char *dbname) final;
   Int_t       DropDataBase(const char *dbname) final;
   Int_t       Reload() final;
   Int_t       Shutdown() final;
   const char *ServerInfo() final;

   Bool_t      StartTransaction() final;
   Bool_t      Commit() final;
   Bool_t      Rollback() final;

   ClassDefOverride(TODBCServer,0)  // Connection to MySQL server
};

#endif
