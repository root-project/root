// @(#)root/mysql:$Name:  $:$Id: TMySQLServer.h,v 1.6 2006/06/02 14:02:03 brun Exp $
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMySQLServer
#define ROOT_TMySQLServer

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#if !defined(__CINT__)
#ifndef R__WIN32
#include <sys/time.h>
#endif
#include <mysql.h>
#else
struct MYSQL;
#endif



class TMySQLServer : public TSQLServer {

private:
   MYSQL  *fMySQL;    // connection to MySQL server

public:
   TMySQLServer(const char *db, const char *uid, const char *pw);
   ~TMySQLServer();

   void        Close(Option_t *opt="");
   TSQLResult *Query(const char *sql);
   Bool_t      Exec(const char* sql);
   TSQLStatement *Statement(const char *sql, Int_t = 100);
   Bool_t      IsSupportStatement() const;
   Int_t       SelectDataBase(const char *dbname);
   TSQLResult *GetDataBases(const char *wild = 0);
   TSQLResult *GetTables(const char *dbname, const char *wild = 0);
   TList      *GetTablesList(const char* wild = 0);
   TSQLTableInfo *GetTableInfo(const char* tablename);
   TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   Int_t       GetMaxIdentifierLength() { return 64; }
   Int_t       CreateDataBase(const char *dbname);
   Int_t       DropDataBase(const char *dbname);
   Int_t       Reload();
   Int_t       Shutdown();
   const char *ServerInfo();

   Bool_t      StartTransaction();
   Bool_t      Commit();
   Bool_t      Rollback();

   ClassDef(TMySQLServer,0)  // Connection to MySQL server
};

#endif
