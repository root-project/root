// @(#)root/oracle:$Name:  $:$Id: TOracleServer.cxx,v 1.10 2006/05/22 08:55:30 brun Exp $
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TOracleServer.h"
#include "TOracleResult.h"
#include "TOracleStatement.h"
#include "TSQLColumnInfo.h"
#include "TSQLTableInfo.h"
#include "TUrl.h"
#include "TList.h"
#include "TObjString.h"

ClassImp(TOracleServer)

//______________________________________________________________________________
TOracleServer::TOracleServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a Oracle DB server. The db arguments should be
   // of the form "oracle://connection_identifier][/<database>]", e.g.:
   // "oracle://cmscald.fnal.gov/test". The uid is the username and pw
   // the password that should be used for the connection.

   fEnv = 0;
   fConn = 0;

   TUrl url(db);

   if (!url.IsValid()) {
      TString errmsg = "Malformed db argument ";
      errmsg+=db;
      SetError(-1, errmsg.Data(), "TOracleServer");
      MakeZombie();
      return;
   }

   if (strncmp(url.GetProtocol(), "oracle", 6)) {
      SetError(-1, "protocol in db argument should be oracle://", "TOracleServer");
      MakeZombie();
      return;
   }

   const char *conn_str = url.GetFile();
   if (conn_str!=0)
     if (*conn_str == '/') conn_str++; //skip leading "/" if appears

   try {
      fEnv = Environment::createEnvironment();
      fConn = fEnv->createConnection(uid, pw, conn_str);

      fType = "Oracle";
      fHost = url.GetHost();
      fDB   = conn_str;
      fPort = url.GetPort();
      fPort = (fPort) ? fPort : 1521;
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "TOracleServer");
      MakeZombie();
   }
}

//______________________________________________________________________________
TOracleServer::~TOracleServer()
{
   // Close connection to Oracle DB server.

   if (IsConnected())
      Close();
}


//______________________________________________________________________________
void TOracleServer::Close(Option_t *)
{
   // Close connection to Oracle DB server.

   ClearError();

   try {
      if (fConn)
         fEnv->terminateConnection(fConn);
      if (fEnv)
         Environment::terminateEnvironment(fEnv);
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Close");
      //MakeZombie();
   }

   fPort = -1;
}

//______________________________________________________________________________
TSQLStatement *TOracleServer::Statement(const char *sql, Int_t niter)
{
   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","Statement");
      return 0;
   }

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Statement");
      return 0;
   }

   try {
      oracle::occi::Statement *stmt = fConn->createStatement(sql);

      return new TOracleStatement(fConn, stmt, niter);

   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Statement");
   }

   return 0;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","Query");
      return 0;
   }

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Query");
      return 0;
   }

   try {
      oracle::occi::Statement *stmt = fConn->createStatement();

      // NOTE: before special COUNT query was executed to define number of 
      // rows in result set. Now it is not requried, while TOracleResult class
      // will automatically fetch all rows from resultset when 
      // GetRowCount() will be called first time. 
      // It is better do not use GetRowCount() to avoid unnecessary memory usage.
      
      stmt->setSQL(sql);
      stmt->setPrefetchRowCount(1000);
      stmt->setPrefetchMemorySize(1000000);
      stmt->execute();

      TOracleResult *res = new TOracleResult(fConn, stmt);
      return res;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Query");
   }

   return 0;
}

//______________________________________________________________________________
Bool_t TOracleServer::Exec(const char* sql)
{
   // Execute sql command wich does not produce any result set.
   // Return kTRUE if succesfull

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","Exec");
      return 0;
   }

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Exec");
      return 0;
   }

   try {
      oracle::occi::Statement *stmt = fConn->createStatement(sql);
      stmt->execute();
      delete stmt;
      return kTRUE;
   } catch (SQLException &oraex)  {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Exec");
   }

   return kFALSE;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetTables(const char *dbname, const char * /*wild*/)
{
   // List all tables in the specified database. Wild is for wildcarding
   // "t%" list all tables starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   // In Oracle 9 and above, table is accessed in schema.table format.
   // GetTables returns tables in all schemas accessible for the user.
   // Assumption: table ALL_OBJECTS is accessible for the user, which is true in Oracle 10g
   // The returned TSQLResult has two columns: schema_name, table_name
   // "dbname": if specified, return table list of this schema, or return all tables
   // "wild" is not used in this implementation

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","GetTables");
      return 0;
   }

   TString sqlstr("SELECT owner, object_name FROM ALL_OBJECTS WHERE object_type='TABLE'");
   if (dbname)
      sqlstr = sqlstr + " AND owner='" + dbname + "'";

   return Query(sqlstr.Data());
}

//______________________________________________________________________________
TList* TOracleServer::GetTablesList(const char* wild)
{
   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","GetTablesList");
      return 0;
   }
   
   TString cmd("SELECT table_name FROM user_tables");
   if ((wild!=0) && (*wild!=0)) 
      cmd+=Form(" WHERE table_name LIKE '%s'", wild);

   TSQLStatement* stmt = Statement(cmd);
   if (stmt==0) return 0;
   
   TList* lst = 0;
   
   if (stmt->Process()) {
      stmt->StoreResult();
      while (stmt->NextResultRow()) {
         const char* tablename = stmt->GetString(0); 
         if (tablename==0) continue;
         if (lst==0) {
            lst = new TList;
            lst->SetOwner(kTRUE);   
         }
         lst->Add(new TObjString(tablename));
      }
   }
   
   delete stmt;
   
   return lst;
}

//______________________________________________________________________________
TSQLTableInfo *TOracleServer::GetTableInfo(const char* tablename)
{
   // Produces SQL table info
   // Object must be deleted by user

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","GetTableInfo");
      return 0;
   }
   
   if ((tablename==0) || (*tablename==0)) return 0;

   TString sql;
   sql.Form("SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, NULLABLE, CHAR_COL_DECL_LENGTH FROM user_tab_columns WHERE table_name = '%s'", tablename);
   
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
      TString data_type = stmt->GetString(1);
      Int_t data_length = stmt->GetInt(2);      // this is size in bytes
      Int_t data_precision = stmt->GetInt(3);    
      Int_t data_scale = stmt->GetInt(4);
      const char* nstr = stmt->GetString(5);
      Int_t char_col_decl_length = stmt->GetInt(6);
      Int_t data_sign = -1; // no info about sign
      
      Int_t sqltype = kSQL_NONE;
      
      if (data_type=="NUMBER") {
         sqltype = kSQL_NUMERIC;
         if (data_precision<=0) {
            data_precision = -1;
            data_scale = -1;
         } else
         if (data_scale<=0)
            data_scale = -1;
         data_sign = 1;
      } else 
      
      if (data_type=="CHAR") {
         sqltype = kSQL_CHAR;
         data_precision = char_col_decl_length;
         data_scale = -1;
      } else 
      
      if ((data_type=="VARCHAR") || (data_type=="VARCHAR2")) {
         sqltype = kSQL_VARCHAR;
         data_precision = char_col_decl_length;
         data_scale = -1;
      } else
      
      if (data_type=="FLOAT") {
         sqltype = kSQL_FLOAT;
         data_scale = -1;
         if (data_precision==126) data_precision = -1;
         data_sign = 1;
      } else 

      if (data_type=="LONG") {
         sqltype = kSQL_VARCHAR;
         data_length = 0x7fffffff; // size of LONG 2^31-1
         data_precision = -1;
         data_scale = -1;
      } else
      
      if (data_type.Contains("TIMESTAMP")) {
         sqltype = kSQL_TIMESTAMP;
         data_precision = -1;
      }
      
      Bool_t IsNullable = kFALSE;
      if (nstr!=0)
         IsNullable = (*nstr=='Y') || (*nstr=='y');
      
      TSQLColumnInfo* info = 
         new TSQLColumnInfo(columnname, 
                            data_type, 
                            IsNullable,
                            sqltype,
                            data_length,
                            data_precision,
                            data_scale,
                            data_sign);
                            
      if (lst==0) lst = new TList;
      lst->Add(info);
   }
   
   delete stmt;
   
   return new TSQLTableInfo(tablename, lst);
}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetColumns(const char * /*dbname*/, const char *table,
                                      const char * /*wild*/)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","GetColumns");
      return 0;
   }

//  make no sense, while method is not implemented
//   if (SelectDataBase(dbname) != 0) {
//      SetError(-1, "Database is not connected","GetColumns");
//      return 0;
//   }

   return new TOracleResult(fConn, table);
}

//______________________________________________________________________________
Int_t TOracleServer::SelectDataBase(const char * /*dbname*/)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","SelectDataBases");
      return -1;
   }

   // do nothing and return success code
   return 0;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::GetDataBases(const char * /*wild*/)
{
   // List all available databases. Wild is for wildcarding "t%" list all
   // databases starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) 
      SetError(-1, "Database is not connected","GetDataBases");

   return 0;
}

//______________________________________________________________________________
Int_t TOracleServer::CreateDataBase(const char * /*dbname*/)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) 
      SetError(-1, "Database is not connected","CreateDataBase");
      
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::DropDataBase(const char * /*dbname*/)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) 
      SetError(-1, "Database is not connected","DropDataBase");

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) 
      SetError(-1, "Database is not connected","Reload");
   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) 
      SetError(-1, "Database is not connected","Shutdown");
   
   return -1;
}

//______________________________________________________________________________
const char *TOracleServer::ServerInfo()
{
   // Return server info.
   // NOT IMPLEMENTED.

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","ServerInfo");
      return 0;
   }
   
   return "Oracle";
}

//______________________________________________________________________________
Bool_t TOracleServer::StartTransaction()
{
   // Call Commit() to submit all chanes, done before.
   // Commit() ot Rollback() must be used to complete submitted actions or cancel them
   
   return Commit();
}

//______________________________________________________________________________
Bool_t TOracleServer::Commit()
{
   // Commits all changes made since the previous Commit() or Rollback()
   // Return kTRUE if OK
   
   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","Commit");
      return kFALSE;
   }

   try {
      fConn->commit();
      return kTRUE;
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Commit");
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleServer::Rollback()
{
   // Drops all changes made since the previous Commit() or Rollback()
   // Return kTRUE if OK

   ClearError();

   if (!IsConnected()) {
      SetError(-1, "Database is not connected","Rollback");
      return kFALSE;
   }
   
   try {
      fConn->rollback();
      return kTRUE;
   } catch (SQLException &oraex) {
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), "Rollback");
   }

   return kFALSE;
}
