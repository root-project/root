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

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#ifdef __CINT__
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

   void        Close(Option_t *opt="");
   TSQLResult *Query(const char *sql);
   Bool_t      Exec(const char* sql);
   TSQLStatement *Statement(const char *sql, Int_t = 100);
   Bool_t      HasStatement() const { return kTRUE; }
   Int_t       SelectDataBase(const char *dbname);
   TSQLResult *GetDataBases(const char *wild = 0);
   TSQLResult *GetTables(const char *dbname, const char *wild = 0);
   TList      *GetTablesList(const char* wild = 0);
   TSQLTableInfo* GetTableInfo(const char* tablename);
   TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   Int_t       GetMaxIdentifierLength();
   Int_t       CreateDataBase(const char *dbname);
   Int_t       DropDataBase(const char *dbname);
   Int_t       Reload();
   Int_t       Shutdown();
   const char *ServerInfo();

   Bool_t      StartTransaction();
   Bool_t      Commit();
   Bool_t      Rollback();

   ClassDef(TODBCServer,0)  // Connection to MySQL server
};

#endif
