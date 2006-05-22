// @(#)root/mysql:$Name:  $:$Id: TMySQLServer.cxx,v 1.9 2006/05/16 10:59:35 rdm Exp $
// Author: Fons Rademakers   15/02/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TMySQLServer.h"
#include "TMySQLResult.h"
#include "TMySQLStatement.h"
#include "TUrl.h"


ClassImp(TMySQLServer)

//______________________________________________________________________________
TMySQLServer::TMySQLServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a MySQL DB server. The db arguments should be
   // of the form "mysql://<host>[:<port>][/<database>]", e.g.:
   // "mysql://pcroot.cern.ch:3456/test". The uid is the username and pw
   // the password that should be used for the connection.

   fMySQL = 0;

   TUrl url(db);

   if (!url.IsValid()) {
      TString errmsg("malformed db argument ");
      errmsg+=db;
      SetError(-1, errmsg.Data(), "TMySQLServer");
      MakeZombie();
      return;
   }

   if (strncmp(url.GetProtocol(), "mysql", 5)) {
      SetError(-1, "protocol in db argument should be mysql://", "TMySQLServer");
      MakeZombie();
      return;
   }

   const char* dbase = url.GetFile();
   if (dbase!=0)
     if (*dbase=='/') dbase++; //skip leading "/" if appears
   
   fMySQL = new MYSQL;
   mysql_init(fMySQL);

   if (mysql_real_connect(fMySQL, url.GetHost(), uid, pw, dbase,
                          url.GetPort(), 0, 0)) {
      fType = "MySQL";
      fHost = url.GetHost();
      fDB   = dbase;
      fPort = url.GetPort();
   } else {
      SetError(mysql_errno(fMySQL), mysql_error(fMySQL), "TMySQLServer");
      MakeZombie();
   }
}

//______________________________________________________________________________
TMySQLServer::~TMySQLServer()
{
   // Close connection to MySQL DB server.

   if (IsConnected())
      Close();
   delete fMySQL;
}

// Reset error and check that server connected
#define CheckConnect(method, res)                       \
   {                                                    \
      ClearError();                                     \
      if (!IsConnected()) {                             \
         SetError(-1,"MySQL server is not connected",method); \
         return res;                                    \
      }                                                 \
   }


// check last mysql error code
#define CheckErrNo(method, force, res)                  \
   {                                                    \
      unsigned int errno = mysql_errno(fMySQL);         \
      if ((errno!=0) || force) {                        \
         const char* errmsg = mysql_error(fMySQL);      \
         if (errno==0) { errno = 11111; errmsg = "MySQL error"; } \
         SetError(errno, errmsg, method);               \
         return res;                                    \
      }                                                 \
   }


//______________________________________________________________________________
void TMySQLServer::Close(Option_t *)
{
   // Close connection to MySQL DB server.

   ClearError();

   if (!fMySQL)
      return;

   mysql_close(fMySQL);
//   CheckErrNo("Close", kFALSE, );
   fPort = -1;
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   CheckConnect("Query", 0);

   if (mysql_query(fMySQL, sql) != 0) 
      CheckErrNo("Query",kTRUE,0);

   MYSQL_RES *res = mysql_store_result(fMySQL);
   CheckErrNo("Query", kFALSE, 0);
   
   return new TMySQLResult(res);
}

//______________________________________________________________________________
Int_t TMySQLServer::SelectDataBase(const char *dbname)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.

   CheckConnect("SelectDataBase", -1);

   Int_t res = mysql_select_db(fMySQL, dbname);
   if (res==0) fDB = dbname;
          else CheckErrNo("SelectDataBase", kTRUE, res);  
          
   return res;
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetDataBases(const char *wild)
{
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   CheckConnect("GetDataBases", 0);

   MYSQL_RES *res = mysql_list_dbs(fMySQL, wild);
   
   CheckErrNo("GetDataBases", kFALSE, 0);  
   
   return new TMySQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetTables(const char *dbname, const char *wild)
{
   // List all tables in the specified database. Wild is for wildcarding
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   CheckConnect("GetTables", 0);

   if (SelectDataBase(dbname) != 0) return 0;

   MYSQL_RES *res = mysql_list_tables(fMySQL, wild);
   
   CheckErrNo("GetTables", kFALSE, 0);  
   
   return new TMySQLResult(res);
}

//______________________________________________________________________________
TSQLResult *TMySQLServer::GetColumns(const char *dbname, const char *table,
                                     const char *wild)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   CheckConnect("GetColumns", 0);

   if (SelectDataBase(dbname) != 0) return 0;

   TString sql;
   if (wild)
      sql.Form("SHOW COLUMNS FROM %s LIKE '%s'", table, wild);
   else
      sql.Form("SHOW COLUMNS FROM %s", table);

   return Query(sql.Data());
}

//______________________________________________________________________________
Int_t TMySQLServer::CreateDataBase(const char *dbname)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.

   CheckConnect("CreateDataBase", -1);
   
   Int_t res = mysql_query(fMySQL, Form("CREATE DATABASE %s",dbname));

   CheckErrNo("CreateDataBase", kFALSE, res);

   return res;
}

//______________________________________________________________________________
Int_t TMySQLServer::DropDataBase(const char *dbname)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.

   CheckConnect("DropDataBase", -1);

   Int_t res = mysql_query(fMySQL, Form("DROP DATABASE %s",dbname));

   CheckErrNo("DropDataBase", kFALSE, res);

   return res;
}

//______________________________________________________________________________
Int_t TMySQLServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.

   CheckConnect("Reload", -1);

   Int_t res = mysql_reload(fMySQL);

   CheckErrNo("Reload", kFALSE, res);

   return res;
}

//______________________________________________________________________________
Int_t TMySQLServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.

   CheckConnect("Shutdown", -1);
   
   Int_t res;
   
#if MYSQL_VERSION_ID >= 50001 || \
    (MYSQL_VERSION_ID < 50000 && MYSQL_VERSION_ID >= 40103)
   res = mysql_shutdown(fMySQL, SHUTDOWN_DEFAULT);
#else
   res = mysql_shutdown(fMySQL);
#endif

   CheckErrNo("Shutdown", kFALSE, res);

   return res;
}

//______________________________________________________________________________
const char *TMySQLServer::ServerInfo()
{
   // Return server info.

   CheckConnect("ServerInfo", 0);
   
   const char* res = mysql_get_server_info(fMySQL);

   CheckErrNo("ServerInfo", kFALSE, res);
   
   return res;
}


//______________________________________________________________________________
TSQLStatement* TMySQLServer::Statement(const char *sql, Int_t)
{
   // Produce TMySQLStatement 
    
   CheckConnect("Statement", 0);

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Statement");
      return 0;
   }

   MYSQL_STMT *stmt = mysql_stmt_init(fMySQL);
   if (!stmt) 
      CheckErrNo("Statement", kTRUE, 0); 
    
   if (mysql_stmt_prepare(stmt, sql, strlen(sql))) {
      mysql_stmt_close(stmt);
      CheckErrNo("Statement", kTRUE, 0); 
   }

   return new TMySQLStatement(stmt);
}

//______________________________________________________________________________
Bool_t TMySQLServer::StartTransaction()
{
   // Start transaction 
    
   CheckConnect("StartTransaction", kFALSE);
   
   return TSQLServer::StartTransaction();
   
//   return Commit();
}

//______________________________________________________________________________
Bool_t TMySQLServer::Commit()
{
   // Commit changes

   CheckConnect("Commit", kFALSE);
   
   if (mysql_commit(fMySQL))
      CheckErrNo("Commit", kTRUE, kFALSE); 
     
   return kTRUE;
   
}

//______________________________________________________________________________
Bool_t TMySQLServer::Rollback()
{
   // Rollback changes

   CheckConnect("Rollback", kFALSE);
   
   if (mysql_rollback(fMySQL))
      CheckErrNo("Rollback", kTRUE, kFALSE); 
      
   return kTRUE;
}
