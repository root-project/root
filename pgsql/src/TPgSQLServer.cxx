// @(#)root/pgsql:$Name:$:$Id:$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPgSQLServer.h"
#include "TPgSQLResult.h"
#include "TUrl.h"


ClassImp(TPgSQLServer)

//______________________________________________________________________________
TPgSQLServer::TPgSQLServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a PgSQL DB server. The db arguments should be
   // of the form "pgsql://<host>[:<port>][/<database>]", e.g.:
   // "pgsql://pcroot.cern.ch:3456/test". The uid is the username and pw
   // the password that should be used for the connection.

   fPgSQL = 0;

   TUrl url(db);

   if (!url.IsValid()) {
      Error("TPgSQLServer", "malformed db argument %s", db);
      MakeZombie();
      return;
   }

   if (strncmp(url.GetProtocol(), "pgsql", 5)) {
      Error("TPgSQLServer", "protocol in db argument should be pgsql it is %s",
            url.GetProtocol());
      MakeZombie();
      return;
   }

   const char *dbase = 0;
   if (strcmp(url.GetFile(), "/"))
      dbase = url.GetFile()+1;   //skip leading /

   fPgSQL = PQsetdbLogin(url.GetHost(), (const char *) url.GetPort(), 0, 0, dbase, uid, pw);

   if (PQstatus(fPgSQL) != CONNECTION_BAD) {
      fType = "PgSQL";
      fHost = url.GetHost();
      fDB   = dbase;
      fPort = url.GetPort();
   } else {
      Error("TPgSQLServer", "connection to %s failed", url.GetHost());
      MakeZombie();
   }
}

//______________________________________________________________________________
TPgSQLServer::~TPgSQLServer()
{
   // Close connection to PgSQL DB server.

   if (IsConnected())
      Close();
}

//______________________________________________________________________________
void TPgSQLServer::Close(Option_t *)
{
   // Close connection to PgSQL DB server.

   if (!fPgSQL)
      return;

   PQfinish(fPgSQL);
   fPort = -1;
}

//______________________________________________________________________________
TSQLResult *TPgSQLServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("Query", "not connected");
      return 0;
   }

   PGresult *res = PQexec(fPgSQL, sql);

   if ((PQresultStatus(res) != PGRES_COMMAND_OK) &&
       (PQresultStatus(res) != PGRES_TUPLES_OK)) {
      Error("Query", PQresultErrorMessage(res));
      PQclear(res);
      return 0;
   }

   return new TPgSQLResult(res);
}

//______________________________________________________________________________
Int_t TPgSQLServer::SelectDataBase(const char *dbname)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.

   if (!IsConnected()) {
      Error("SelectDataBase", "not connected");
      return 0;
   }
   Error("SelectDataBase", "not implemented");
   return 0;
}

//______________________________________________________________________________
TSQLResult *TPgSQLServer::GetDataBases(const char *wild)
{
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return 0;
   }

   PGresult *res = PQexec(fPgSQL, "SELECT pg_database.datname FROM pg_database");
   return new TPgSQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TPgSQLServer::GetTables(const char *dbname, const char *wild)
{
   // List all tables in the specified database. Wild is for wildcarding
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetTables", "not connected");
      return 0;
   }

   if (SelectDataBase(dbname) != 0) {
      Error("GetTables", "no such database %s", dbname);
      return 0;
   }

   PGresult *res = PQexec(fPgSQL, "SELECT relname FROM pg_class");
   return new TPgSQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TPgSQLServer::GetColumns(const char *dbname, const char *table,
                                     const char *wild)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   if (!IsConnected()) {
      Error("GetColumns", "not connected");
      return 0;
   }

   if (SelectDataBase(dbname) != 0) {
      Error("GetColumns", "no such database %s", dbname);
      return 0;
   }

   char *sql;
   if (wild)
      sql = Form("SELECT RELNAME FROM %s LIKE %s", table, wild);
   else
      sql = Form("SELECT RELNAME FROM %s", table);

   //return Query(sql);
   Error("GetColumns", "not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TPgSQLServer::CreateDataBase(const char *dbname)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.

   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return -1;
   }
   char *sql;
   sql = Form("CREATE DATABASE %s", dbname);
   PGresult *res = PQexec(fPgSQL, sql);
   PQclear(res);
   return 0;
}

//______________________________________________________________________________
Int_t TPgSQLServer::DropDataBase(const char *dbname)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.

   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return -1;
   }
   char *sql;
   sql = Form("DROP DATABASE %s", dbname);
   PGresult *res = PQexec(fPgSQL, sql);
   PQclear(res);
   return 0;
}

//______________________________________________________________________________
Int_t TPgSQLServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.

   if (!IsConnected()) {
      Error("Reload", "not connected");
      return 0;
   }
   Error("Reload", "not implemented");
   return 0;
}

//______________________________________________________________________________
Int_t TPgSQLServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.

   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return 0;
   }
   Error("Shutdown", "not implemented");
   return 0;
}

//______________________________________________________________________________
const char *TPgSQLServer::ServerInfo()
{
   // Return server info.

   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return 0;
   }
   Error("ServerInfo", "not implemented");
   return 0;
}
