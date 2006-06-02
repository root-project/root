// @(#)root/mysql:$Name:  $:$Id: TMySQLServer.cxx,v 1.10 2006/05/22 08:55:30 brun Exp $
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
#include "TSQLColumnInfo.h"
#include "TSQLTableInfo.h"
#include "TSQLRow.h"
#include "TUrl.h"
#include "TList.h"
#include "TObjString.h"


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

   if (mysql_query(fMySQL, sql)) 
      CheckErrNo("Query",kTRUE,0);

   MYSQL_RES *res = mysql_store_result(fMySQL);
   CheckErrNo("Query", kFALSE, 0);
   
   return new TMySQLResult(res);
}

//______________________________________________________________________________
Bool_t TMySQLServer::Exec(const char* sql)
{
   // Execute SQL command which does not produce any result sets
   // Returns kTRUE is successfull

   CheckConnect("Exec", kFALSE);

   if (mysql_query(fMySQL, sql)) 
      CheckErrNo("Exec",kTRUE,kFALSE);

   return !IsError();
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
TList* TMySQLServer::GetTablesList(const char* wild)
{
   // Return list of tables with specified wildcard
   
   CheckConnect("GetTablesList", 0);
   
   MYSQL_RES *res = mysql_list_tables(fMySQL, wild);
   
   CheckErrNo("GetTablesList", kFALSE, 0);
   
   MYSQL_ROW row = mysql_fetch_row(res);
   
   TList* lst = 0;
   
   while (row!=0) {
      CheckErrNo("GetTablesList", kFALSE, lst);
      
      const char* tablename = row[0]; 
      
      if (tablename!=0) {
         if (lst==0) {
            lst = new TList();
            lst->SetOwner(kTRUE);   
         } 
         lst->Add(new TObjString(tablename));   
      }
       
      row = mysql_fetch_row(res); 
   }
   
   mysql_free_result(res);

   return lst;
}

//______________________________________________________________________________
TSQLTableInfo* TMySQLServer::GetTableInfo(const char* tablename)
{
   // Produces SQL table info
   // Object must be deleted by user
   
   CheckConnect("GetTableInfo", 0);
   
   if ((tablename==0) || (*tablename==0)) return 0;

   TString sql;
   sql.Form("SELECT * FROM `%s` LIMIT 1", tablename);

   if (mysql_query(fMySQL, sql.Data()) != 0) 
      CheckErrNo("GetTableInfo", kTRUE, 0);

   MYSQL_RES *res = mysql_store_result(fMySQL);
   CheckErrNo("GetTableInfo", kFALSE, 0);

   unsigned int numfields = mysql_num_fields(res);

   MYSQL_FIELD* fields = mysql_fetch_fields(res);
   
   sql.Form("SHOW COLUMNS FROM `%s`", tablename);
   TSQLStatement* stmt = Statement(sql.Data());

   if ((stmt!=0) && !stmt->Process()) {
      delete stmt;
      stmt = 0;
   }
   
   if (stmt==0) {
      mysql_free_result(res);
      return 0;
   }
   
   TList* lst = 0;
   
   unsigned int nfield = 0;

   stmt->StoreResult();  

   while (stmt->NextResultRow()) {
      const char* column_name = stmt->GetString(0);
      const char* type_name = stmt->GetString(1);
      
      if ((nfield>=numfields) ||
          (strcmp(column_name, fields[nfield].name)!=0))
       {
         SetError(-1,"missmatch in column names","GetTableInfo");
         break;
      }
      
      Int_t sqltype = kSQL_NONE;
      
      Int_t data_size = -1;    // size in bytes
      Int_t data_length = -1;  // declaration like VARCHAR(n) or NUMERIC(n)
      Int_t data_scale = -1;   // second argument in declaration
      Int_t data_sign = -1; // signed type or not
      
      if (IS_NUM(fields[nfield].type))
         if (fields[nfield].flags & UNSIGNED_FLAG)
            data_sign = 0;
         else
            data_sign = 1;    

      Bool_t Nullable = (fields[nfield].flags & NOT_NULL_FLAG) == 0;

      data_length = fields[nfield].length;
      if (data_length==0) data_length = -1;

      switch (fields[nfield].type) {
         case MYSQL_TYPE_TINY:
         case MYSQL_TYPE_SHORT:
         case MYSQL_TYPE_LONG:
         case MYSQL_TYPE_INT24:
         case MYSQL_TYPE_LONGLONG:
            sqltype = kSQL_INTEGER;
            break;
         case MYSQL_TYPE_DECIMAL:
            sqltype = kSQL_NUMERIC;
            data_scale = fields[nfield].decimals;
            break;
         case MYSQL_TYPE_FLOAT:
            sqltype = kSQL_FLOAT;
            break;
         case MYSQL_TYPE_DOUBLE:
            sqltype = kSQL_DOUBLE;
            break;
         case MYSQL_TYPE_TIMESTAMP:
            sqltype = kSQL_TIMESTAMP;
            break;
         case MYSQL_TYPE_DATE:
         case MYSQL_TYPE_TIME:
         case MYSQL_TYPE_DATETIME:
         case MYSQL_TYPE_YEAR:
            break;
         case MYSQL_TYPE_STRING:
            if (fields[nfield].charsetnr==63)
               sqltype = kSQL_BINARY;
            else
               sqltype = kSQL_CHAR;
            data_size = data_length;
            break;
         case MYSQL_TYPE_VAR_STRING:
            if (fields[nfield].charsetnr==63)
               sqltype = kSQL_BINARY;
            else
               sqltype = kSQL_VARCHAR;
            data_size = data_length;   
            break;
         case MYSQL_TYPE_BLOB:
            if (fields[nfield].charsetnr==63)
               sqltype = kSQL_BINARY;
            else
               sqltype = kSQL_VARCHAR;
            data_length = fields[nfield].max_length;
            if (data_length==0) data_length = -1;
            data_size = data_length;
            break;
         case MYSQL_TYPE_SET:
         case MYSQL_TYPE_ENUM:
         case MYSQL_TYPE_GEOMETRY:
         case MYSQL_TYPE_NULL:
            break;
         default:
            if (IS_NUM(fields[nfield].type))
               sqltype = kSQL_NUMERIC; 
      }
      
      if (lst==0) lst = new TList;
      lst->Add(new TSQLColumnInfo(column_name, 
                                  type_name, 
                                  Nullable,
                                  sqltype,
                                  data_size,
                                  data_length,
                                  data_scale,
                                  data_sign));
      
      nfield++;
   }
   
   mysql_free_result(res);
   delete stmt;

   sql.Form("SHOW TABLE STATUS LIKE '%s'", tablename);

   TSQLTableInfo* info = 0;

   TSQLResult* stats = Query(sql.Data());
   
   if (stats!=0) {
      TSQLRow* row = 0;
      
      while ((row = stats->Next()) != 0) {
         if (strcmp(row->GetField(0), tablename)!=0) {
            delete row;
            continue;  
         }
         const char* comments = 0;
         const char* engine = 0;
         const char* create_time = 0;
         const char* update_time = 0;
         
         for (int n=1;n<stats->GetFieldCount();n++) {
            TString fname = stats->GetFieldName(n);
            fname.ToLower();
            if (fname=="engine") engine = row->GetField(n); else
            if (fname=="comment") comments = row->GetField(n); else
            if (fname=="create_time") create_time = row->GetField(n); else
            if (fname=="update_time") update_time = row->GetField(n);
         }
         
         info = new TSQLTableInfo(tablename, 
                                  lst,
                                  comments,
                                  engine,
                                  create_time,
                                  update_time);
         
         delete row;
         break;
      }
      delete stats;    
   }
   
   if (info==0)
      info = new TSQLTableInfo(tablename, lst);
   
   return info;
}


/*
//______________________________________________________________________________
TSQLTableInfo* TMySQLServer::GetTableInfo(const char* tablename)
{
   // Produces SQL table info
   // Object must be deleted by user
   
   CheckConnect("GetTableInfo", 0);
   
   if ((tablename==0) || (*tablename==0)) return 0;

   TString sql;
   sql.Form("SHOW COLUMNS FROM `%s`", tablename);
   
   TSQLStatement* stmt = Statement(sql.Data(), 10);
   if (stmt==0) return 0;
   
   if (!stmt->Process()) {
      delete stmt;
      return 0;
   }

   TList* lst = 0;

   stmt->StoreResult();  
   
   while (stmt->NextResultRow()) {
      const char* columnname = stmt->GetString(0);
      const char* sqltype = stmt->GetString(1);
      const char* nstr = stmt->GetString(2);
      
      Bool_t IsNullable = kFALSE;
      if (nstr!=0)
         IsNullable = (strcmp(nstr,"YES")==0) || (strcmp(nstr,"yes")==0);
      
      if (lst==0) lst = new TList;
      
      lst->Add(new TSQLColumnInfo(columnname, sqltype, IsNullable));
   }
   
   delete stmt;
   
   return new TSQLTableInfo(tablename, lst);
}

*/

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
