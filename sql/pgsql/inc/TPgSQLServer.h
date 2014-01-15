// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPgSQLServer
#define ROOT_TPgSQLServer

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#include <map>
#include <string>

#if !defined(__CINT__)
#include <libpq-fe.h>
#else
struct PGconn;
#endif



class TPgSQLServer : public TSQLServer {

private:
   PGconn  *fPgSQL;    // connection to PgSQL server
   TString  fSrvInfo;  // Server info
   std::map<Int_t,std::string> fOidTypNameMap; // Map of oid to typname, used in GetTableInfo()
public:
   TPgSQLServer(const char *db, const char *uid, const char *pw);
   ~TPgSQLServer();

   void           Close(Option_t *opt="");
   TSQLResult    *Query(const char *sql);
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

   ClassDef(TPgSQLServer,0)  // Connection to PgSQL server
};

#endif
