// @(#)root/sqlite:$Id$
// Author: o.freyermuth <o.f@cern.ch>, 01/06/2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLiteServer
#define ROOT_TSQLiteServer

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#if !defined(__CINT__)
#include <sqlite3.h>
#else
struct sqlite3;
#endif



class TSQLiteServer : public TSQLServer {

 private:
   TString   fSrvInfo;   // Server info string
   sqlite3  *fSQLite;    // connection to SQLite DB

 public:
   TSQLiteServer(const char *db, const char *uid=NULL, const char *pw=NULL);
   ~TSQLiteServer();

   void           Close(Option_t *opt="");
   Bool_t         StartTransaction();
   TSQLResult    *Query(const char *sql);
   Bool_t         Exec(const char *sql);
   TSQLStatement *Statement(const char *sql, Int_t = 100);
   Bool_t         HasStatement() const;
   Int_t          SelectDataBase(const char *dbname);
   TSQLResult    *GetDataBases(const char *wild = 0);
   TSQLResult    *GetTables(const char *dbname, const char *wild = 0);
   TSQLResult    *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   TSQLTableInfo *GetTableInfo(const char* tablename);
   Int_t          CreateDataBase(const char *dbname);
   Int_t          DropDataBase(const char *dbname);
   Int_t          Reload();
   Int_t          Shutdown();
   const char    *ServerInfo();

   ClassDef(TSQLiteServer,0);  // Connection to SQLite DB
};

#endif
