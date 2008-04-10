// @(#)root/sapdb:$Id$
// Author: Mark Hemberger & Fons Rademakers   03/08/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSapDBServer
#define ROOT_TSapDBServer

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#if !defined(__CINT__)
#include <sys/time.h>
#include <WINDOWS.H>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#include <sql.h>
#include <sqlext.h>
#else
typedef long SQLHENV;
typedef long SQLHDBC;
typedef long SQLHSTMT;
#endif



class TSapDBServer : public TSQLServer {

private:
   SQLHDBC  fSapDB;     // connection to SapDB server
   SQLHENV  fEnv;       // environment for/of ODBC call
   SQLHSTMT fStmt;      // statement result set
   SQLHSTMT fStmtCnt;   // statement result set

   static Int_t printSQLError(SQLHDBC hdbc, SQLHSTMT hstmt);

public:
   TSapDBServer(const char *db, const char *uid, const char *pw);
   ~TSapDBServer();

   void        Close(Option_t *opt="");
   TSQLResult *Query(const char *sql);
   Int_t       SelectDataBase(const char *dbname);
   TSQLResult *GetDataBases(const char *wild = 0);
   TSQLResult *GetTables(const char *dbname, const char *wild = 0);
   TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   Int_t       CreateDataBase(const char *dbname);
   Int_t       DropDataBase(const char *dbname);
   Int_t       Reload();
   Int_t       Shutdown();
   const char *ServerInfo();

   ClassDef(TSapDBServer,0)  // Connection to SapDB server
};

#endif
