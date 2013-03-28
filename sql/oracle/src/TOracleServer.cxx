// @(#)root/oracle:$Id$
// Author: Yan Liu and Shaowen Wang   23/11/04

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TOracleServer                                                        //
//                                                                      //
// This class implements an OCCI interface to Oracle data bases.        //
// It uses the instantclient10 software available from Oracle.          //
// To install this client software do:                                  //
// 1) Download Instant Client Packages (4 files) from:                  //
//     http://www.oracle.com/technology/software/tech/oci/instantclient/index.html
// 2) Unzip the files into instantclient10_2 (Mac OS X example here):   //
//     unzip instantclient-basic-macosx-10.2.0.4.zip                    //
//     unzip instantclient-sqlplus-macosx-10.2.0.4.zip                  //
//     unzip instantclient-sdk-macosx-10.2.0.4.zip                      //
//     unzip instantclient-jdbc-macosx-10.2.0.4.zip                     //
// 3) Create two symbolic links for the files that have the version     //
//    appended:                                                         //
//      ln -s libclntsh.dylib.10.1 libclntsh.dylib                      //
//      ln -s libocci.dylib.10.1 libocci.dylib                          //
// 4) Add instantclient10_1 directory to your (DY)LD_LIBRARY_PATH       //
//    in your .profile:                                                 //
//      export (DY)LD_LIBRARY_PATH="<pathto>/instantclient10_2"         //
//    Use DY only on Mac OS X.                                          //
// 5) If you also want to use the sqlplus command line app add also     //
//      export SQLPATH="<pathto>/instantclient10_2"                     //
// 6) If you want to connect to a remote db server you will also need   //
//    to create a tnsname.ora file which describes the local_name for   //
//    the remote db servers (at CERN most public machines have this     //
//    file in /etc). If it is not in /etc create TNS_ADMIN:             //
//      export TNS_ADMIN="<path-to-dir-containing-tnsname.ora>"         //
// 7) Test it our with the sqlplus command line app:                    //
//      sqlplus [username][/password]@<local_name>                      //
//    or                                                                //
//      sqlplus [username][/password]@//[hostname][:port][/database]    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TOracleServer.h"
#include "TOracleResult.h"
#include "TOracleStatement.h"
#include "TSQLColumnInfo.h"
#include "TSQLTableInfo.h"
#include "TUrl.h"
#include "TList.h"
#include "TObjString.h"

ClassImp(TOracleServer)

using namespace std;
using namespace oracle::occi;

const char* TOracleServer::fgDatimeFormat = "MM/DD/YYYY, HH24:MI:SS";


// Reset error and check that server connected
#define CheckConnect(method, res)                       \
      ClearError();                                     \
      if (!IsConnected()) {                             \
         SetError(-1,"Oracle database is not connected",method); \
         return res;                                    \
      }

// catch Oracle exception after try block
#define CatchError(method)                           \
   catch (SQLException &oraex) {                     \
      SetError(oraex.getErrorCode(), oraex.getMessage().c_str(), method); \
   }

//______________________________________________________________________________
TOracleServer::TOracleServer(const char *db, const char *uid, const char *pw)
{
   // Open a connection to a Oracle DB server. The db arguments should be
   // of the form "oracle://connection_identifier[/<database>]", e.g.:
   // "oracle://cmscald.fnal.gov/test". The uid is the username and pw
   // the password that should be used for the connection.

   fEnv = 0;
   fConn = 0;

   if (gDebug>0) {
      // this code is necessary to guarantee, that libclntsh.so will be
      // linked to libOracle.so.
      sword  major_version(0), minor_version(0), update_num(0), patch_num(0), port_update_num(0);
      OCIClientVersion(&major_version, &minor_version, &update_num, &patch_num, &port_update_num);
      Info("TOracleServer","Oracle Call Interface version %u.%u.%u.%u.%u",
            (unsigned) major_version, (unsigned) minor_version, (unsigned) update_num, (unsigned) patch_num, (unsigned) port_update_num);
   }

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
      // found out whether to use objet mode
      TString options = url.GetOptions();
      Int_t pos = options.Index("ObjectMode");
      // create environment accordingly
      if (pos != kNPOS) {
        fEnv = Environment::createEnvironment(Environment::OBJECT);
      } else {
        fEnv = Environment::createEnvironment();
      }
      fConn = fEnv->createConnection(uid, pw, conn_str ? conn_str : "");

      fType = "Oracle";
      fHost = url.GetHost();
      fDB   = conn_str;
      fPort = url.GetPort();
      fPort = (fPort>0) ? fPort : 1521;
      return;

   } CatchError("TOracleServer")

   MakeZombie();
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
   } CatchError("Close")

   fPort = -1;
}

//______________________________________________________________________________
TSQLStatement *TOracleServer::Statement(const char *sql, Int_t niter)
{
   CheckConnect("Statement",0);

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Statement");
      return 0;
   }

   try {
      oracle::occi::Statement *stmt = fConn->createStatement(sql);

      Blob parblob(fConn);

      return new TOracleStatement(fEnv, fConn, stmt, niter, fErrorOut);

   } CatchError("Statement")

   return 0;
}

//______________________________________________________________________________
TSQLResult *TOracleServer::Query(const char *sql)
{
   // Execute SQL command. Result object must be deleted by the user.
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.

   CheckConnect("Query",0);

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
   } CatchError("Query")

   return 0;
}

//______________________________________________________________________________
Bool_t TOracleServer::Exec(const char* sql)
{
   // Execute sql command wich does not produce any result set.
   // Return kTRUE if successful

   CheckConnect("Exec", kFALSE);

   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Exec");
      return kFALSE;
   }

   oracle::occi::Statement *stmt = 0;

   Bool_t res = kFALSE;

   try {
      stmt = fConn->createStatement(sql);
      stmt->execute();
      res = kTRUE;
   } CatchError("Exec")

   try {
      fConn->terminateStatement(stmt);
   } CatchError("Exec")

   return res;
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

   CheckConnect("GetTables",0);

   TString sqlstr("SELECT object_name,owner FROM ALL_OBJECTS WHERE object_type='TABLE'");
   if (dbname && dbname[0])
      sqlstr = sqlstr + " AND owner='" + dbname + "'";

   return Query(sqlstr.Data());
}

//______________________________________________________________________________
TList* TOracleServer::GetTablesList(const char* wild)
{
   CheckConnect("GetTablesList",0);

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

   CheckConnect("GetTableInfo",0);

   if ((tablename==0) || (*tablename==0)) return 0;

   TString table(tablename);
   table.ToUpper();
   TString sql;
   sql.Form("SELECT COLUMN_NAME, DATA_TYPE, DATA_LENGTH, DATA_PRECISION, DATA_SCALE, NULLABLE, CHAR_COL_DECL_LENGTH FROM user_tab_columns WHERE table_name = '%s' ORDER BY COLUMN_ID", table.Data());

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

      if (data_type=="BINARY_FLOAT") {
         sqltype = kSQL_FLOAT;
         data_scale = -1;
         data_precision = -1;
         data_sign = 1;
      } else

      if (data_type=="BINARY_DOUBLE") {
         sqltype = kSQL_DOUBLE;
         data_scale = -1;
         data_precision = -1;
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
TSQLResult *TOracleServer::GetColumns(const char * /*dbname*/, const char *tablename,
                                      const char * wild)
{
   // List all columns in specified table in the specified database.
   // Wild is for wildcarding "t%" list all columns starting with "t".
   // Returns a pointer to a TSQLResult object if successful, 0 otherwise.
   // The result object must be deleted by the user.

   CheckConnect("GetColumns",0);

//  make no sense, while method is not implemented
//   if (SelectDataBase(dbname) != 0) {
//      SetError(-1, "Database is not connected","GetColumns");
//      return 0;
//   }

   TString sql;
   TString table(tablename);
   table.ToUpper();
   if (wild && wild[0])
      sql.Form("select COLUMN_NAME, concat(concat(concat(data_type,'('),data_length),')') \"Type\" FROM user_tab_columns WHERE table_name like '%s' ORDER BY COLUMN_ID", wild);
   else
      sql.Form("select COLUMN_NAME, concat(concat(concat(data_type,'('),data_length),')') \"Type\" FROM user_tab_columns WHERE table_name = '%s' ORDER BY COLUMN_ID", table.Data()); 
   return Query(sql);
}

//______________________________________________________________________________
Int_t TOracleServer::SelectDataBase(const char * /*dbname*/)
{
   // Select a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

   CheckConnect("SelectDataBase", -1);

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

   CheckConnect("GetDataBases",0);

   return 0;
}

//______________________________________________________________________________
Int_t TOracleServer::CreateDataBase(const char * /*dbname*/)
{
   // Create a database. Returns 0 if successful, non-zero otherwise.
   // NOT IMPLEMENTED.

   CheckConnect("CreateDataBase",-1);

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::DropDataBase(const char * /*dbname*/)
{
   // Drop (i.e. delete) a database. Returns 0 if successful, non-zero
   // otherwise.
   // NOT IMPLEMENTED.

   CheckConnect("DropDataBase",-1);

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Reload()
{
   // Reload permission tables. Returns 0 if successful, non-zero
   // otherwise. User must have reload permissions.
   // NOT IMPLEMENTED.

   CheckConnect("Reload", -1);

   return -1;
}

//______________________________________________________________________________
Int_t TOracleServer::Shutdown()
{
   // Shutdown the database server. Returns 0 if successful, non-zero
   // otherwise. User must have shutdown permissions.
   // NOT IMPLEMENTED.

   CheckConnect("Shutdown", -1);

   return -1;
}

//______________________________________________________________________________
const char *TOracleServer::ServerInfo()
{
   // Return Oracle server version info.

   CheckConnect("ServerInfo", 0);

   fInfo = "Oracle";
   TSQLStatement* stmt = Statement("select * from v$version");
   if (stmt!=0) {
       stmt->EnableErrorOutput(kFALSE);
       if (stmt->Process()) {
          fInfo = "";
          stmt->StoreResult();
          while (stmt->NextResultRow()) {
             if (fInfo.Length()>0) fInfo += "\n";
             fInfo += stmt->GetString(0);
          }
       }
       delete stmt;
   }

   return fInfo.Data();
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

   CheckConnect("Commit", kFALSE);

   try {
      fConn->commit();
      return kTRUE;
   } CatchError("Commit")

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TOracleServer::Rollback()
{
   // Drops all changes made since the previous Commit() or Rollback()
   // Return kTRUE if OK

   CheckConnect("Rollback", kFALSE);

   try {
      fConn->rollback();
      return kTRUE;
   } CatchError("Rollback")

   return kFALSE;
}

//______________________________________________________________________________
void TOracleServer::SetDatimeFormat(const char* fmt)
{
   // set format for converting timestamps or date field into string
   // default value is "MM/DD/YYYY, HH24:MI:SS"

   if (fmt==0) fmt = "MM/DD/YYYY, HH24:MI:SS";
   fgDatimeFormat = fmt;
}

//______________________________________________________________________________
const char* TOracleServer::GetDatimeFormat()
{
   // return value of actul convertion format from timestamps or date to string

   return fgDatimeFormat;
}
