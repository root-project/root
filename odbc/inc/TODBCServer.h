// @(#)root/odbc:$Name:  $:$Id: TODBCServer.h,v 1.3 2001/12/11 14:19:51 rdm Exp $
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


class TODBCServer : public TSQLServer {

private:
   SQLHENV   fHenv;
   SQLHDBC   fHdbc;
   TString   fInfo;

   Bool_t ExtractErrors(SQLRETURN retcode, const char* method);

public:
   TODBCServer(const char* db, const char *uid, const char *pw);
   virtual ~TODBCServer();

   void        Close(Option_t *opt="");
   TSQLResult *Query(const char *sql);
   TSQLStatement *Statement(const char *sql, Int_t = 100);
   Int_t       SelectDataBase(const char *dbname);
   TSQLResult *GetDataBases(const char *wild = 0);
   TSQLResult *GetTables(const char *dbname, const char *wild = 0);
   TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   Int_t       CreateDataBase(const char *dbname);
   Int_t       DropDataBase(const char *dbname);
   Int_t       Reload();
   Int_t       Shutdown();
   const char *ServerInfo();

   ClassDef(TODBCServer,0)  // Connection to MySQL server
};

#endif
