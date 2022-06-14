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

#include "TSQLServer.h"

#include <mysql.h>

class TMySQLServer : public TSQLServer {

protected:
   MYSQL     *fMySQL{nullptr};   // connection to MySQL server
   TString    fInfo;             // server info string

public:
   TMySQLServer(const char *db, const char *uid, const char *pw);
   ~TMySQLServer();

   void           Close(Option_t *opt="") final;
   TSQLResult    *Query(const char *sql) final;
   Bool_t         Exec(const char* sql) final;
   TSQLStatement *Statement(const char *sql, Int_t = 100) final;
   Bool_t         HasStatement() const final;
   Int_t          SelectDataBase(const char *dbname) final;
   TSQLResult    *GetDataBases(const char *wild = nullptr) final;
   TSQLResult    *GetTables(const char *dbname, const char *wild = nullptr) final;
   TList         *GetTablesList(const char* wild = nullptr) final;
   TSQLTableInfo *GetTableInfo(const char* tablename) final;
   TSQLResult    *GetColumns(const char *dbname, const char *table, const char *wild = nullptr) final;
   Int_t          GetMaxIdentifierLength() final { return 64; }
   Int_t          CreateDataBase(const char *dbname) final;
   Int_t          DropDataBase(const char *dbname) final;
   Int_t          Reload() final;
   Int_t          Shutdown() final;
   const char    *ServerInfo() final;

   Bool_t         StartTransaction() final;
   Bool_t         Commit() final;
   Bool_t         Rollback() final;

   Bool_t         PingVerify() final;
   Int_t          Ping() final;

   ClassDefOverride(TMySQLServer,0)  // Connection to MySQL server
};

#endif
