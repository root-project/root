// @(#)root/mysql:$Id$
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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMySQLServer                                                         //
//                                                                      //
// MySQL server plugin implementing the TSQLServer interface.           //
//                                                                      //
// To open a connection to a server use the static method Connect().    //
// The db argument of Connect() is of the form:                         //
//    mysql://<host>[:<port>][/<database>], e.g.                        //
// mysql://pcroot.cern.ch:3456/test                                     //
//                                                                      //
// As an example of connecting to mysql we assume that the server is    //
// running on the local host and that you have access to a database     //
// named "test" by connecting using an account that has a username and  //
// password of "tuser" and "tpass". You can set up this account         //
// by using the "mysql" program to connect to the server as the MySQL   //
// root user and issuing the following statement:                       //
//                                                                      //
// mysql> GRANT ALL ON test.* TO 'tuser'@'localhost' IDENTIFIED BY 'tpass';
//                                                                      //
// If the test database does not exist, create it with this statement:  //
//                                                                      //
// mysql> CREATE DATABASE test;                                         //
//                                                                      //
// If you want to use a different server host, username, password,      //
// or database name, just substitute the appropriate values.            //
// To connect do:                                                       //
//                                                                      //
// TSQLServer *db = TSQLServer::Connect("mysql://localhost/test", "tuser", "tpass");
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TSQLServer
#include "TSQLServer.h"
#endif

#include <mysql.h>

class TMySQLServer : public TSQLServer {

protected:
   MYSQL     *fMySQL;    // connection to MySQL server
   TString    fInfo;     // server info string

public:
   TMySQLServer(const char *db, const char *uid, const char *pw);
   ~TMySQLServer();

   void           Close(Option_t *opt="");
   TSQLResult    *Query(const char *sql);
   Bool_t         Exec(const char* sql);
   TSQLStatement *Statement(const char *sql, Int_t = 100);
   Bool_t         HasStatement() const;
   Int_t          SelectDataBase(const char *dbname);
   TSQLResult    *GetDataBases(const char *wild = 0);
   TSQLResult    *GetTables(const char *dbname, const char *wild = 0);
   TList         *GetTablesList(const char* wild = 0);
   TSQLTableInfo *GetTableInfo(const char* tablename);
   TSQLResult    *GetColumns(const char *dbname, const char *table, const char *wild = 0);
   Int_t          GetMaxIdentifierLength() { return 64; }
   Int_t          CreateDataBase(const char *dbname);
   Int_t          DropDataBase(const char *dbname);
   Int_t          Reload();
   Int_t          Shutdown();
   const char    *ServerInfo();

   Bool_t         StartTransaction();
   Bool_t         Commit();
   Bool_t         Rollback();

   Bool_t         PingVerify();
   Int_t          Ping();

   ClassDef(TMySQLServer,0)  // Connection to MySQL server
};

#endif
