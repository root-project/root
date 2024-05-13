// @(#)root/pgsql:$Id$
// Author: g.p.ciceri <gp.ciceri@acm.org> 01/06/2001

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPgSQLServer.h"
#include "TPgSQLResult.h"
#include "TPgSQLStatement.h"

#include "TSQLColumnInfo.h"
#include "TSQLTableInfo.h"
#include "TSQLRow.h"
#include "TUrl.h"
#include "TList.h"

#include <pg_config.h> // to get PG_VERSION_NUM

#define pgsql_success(x) (((x) == PGRES_EMPTY_QUERY) \
                        || ((x) == PGRES_COMMAND_OK) \
                        || ((x) == PGRES_TUPLES_OK))

#include <libpq-fe.h>

////////////////////////////////////////////////////////////////////////////////
/// PluginManager generator function

TSQLServer* ROOT_Plugin_TPgSQLServer(const char* db, const char* uid, const char* pw) {
   return new TPgSQLServer(db, uid, pw);
}


ClassImp(TPgSQLServer);

////////////////////////////////////////////////////////////////////////////////
/// Open a connection to a PgSQL DB server. The db arguments should be
/// of the form "pgsql://<host>[:<port>][/<database>]", e.g.:
/// "pgsql://pcroot.cern.ch:3456/test". The uid is the username and pw
/// the password that should be used for the connection.

TPgSQLServer::TPgSQLServer(const char *db, const char *uid, const char *pw)
{
   fPgSQL = nullptr;
   fSrvInfo = "";

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

   const char *dbase = url.GetFile();

   if (url.GetPort()) {
      TString port;
      port += url.GetPort();
      fPgSQL = PQsetdbLogin(url.GetHost(), port.Data(), nullptr, nullptr, dbase, uid, pw);
   } else {
      fPgSQL = PQsetdbLogin(url.GetHost(), nullptr, nullptr, nullptr, dbase, uid, pw);
   }

   if (PQstatus(fPgSQL) != CONNECTION_BAD) {
      fType = "PgSQL";
      fHost = url.GetHost();
      fDB   = dbase;
      fPort = url.GetPort();

      // Populate server-info
      fSrvInfo = "postgres ";
      static const char *sql = "select setting from pg_settings where name='server_version'";
      PGresult *res = PQexec(fPgSQL, sql);
      int stat = PQresultStatus(res);
      if (stat == PGRES_TUPLES_OK && PQntuples(res)) {
         char *vers = PQgetvalue(res,0,0);
         fSrvInfo += vers;
         PQclear(res);
      } else {
         fSrvInfo += "unknown version number";
      }
   } else {
      Error("TPgSQLServer", "connection to %s failed", url.GetHost());
      MakeZombie();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Close connection to PgSQL DB server.

TPgSQLServer::~TPgSQLServer()
{
   if (IsConnected())
      Close();
}

////////////////////////////////////////////////////////////////////////////////
/// Close connection to PgSQL DB server.

void TPgSQLServer::Close(Option_t *)
{
   if (!fPgSQL)
      return;

   PQfinish(fPgSQL);
   fPort = -1;
}

////////////////////////////////////////////////////////////////////////////////
/// Execute SQL command. Result object must be deleted by the user.
/// Returns a pointer to a TSQLResult object if successful, 0 otherwise.
/// The result object must be deleted by the user.

TSQLResult *TPgSQLServer::Query(const char *sql)
{
   if (!IsConnected()) {
      Error("Query", "not connected");
      return nullptr;
   }

   PGresult *res = PQexec(fPgSQL, sql);

   if ((PQresultStatus(res) != PGRES_COMMAND_OK) &&
       (PQresultStatus(res) != PGRES_TUPLES_OK)) {
      Error("Query", "%s",PQresultErrorMessage(res));
      PQclear(res);
      return nullptr;
   }

   return new TPgSQLResult(res);
}

////////////////////////////////////////////////////////////////////////////////
/// Select a database. Returns 0 if successful, non-zero otherwise.

Int_t TPgSQLServer::SelectDataBase(const char *dbname)
{
   TString usr;
   TString pwd;
   TString port;
   TString opts;

   if (!IsConnected()) {
      Error("SelectDataBase", "not connected");
      return -1;
   }

   if (dbname == fDB) {
      return 0;
   } else {
      usr = PQuser(fPgSQL);
      pwd = PQpass(fPgSQL);
      port = PQport(fPgSQL);
      opts = PQoptions(fPgSQL);

      Close();
      fPgSQL = PQsetdbLogin(fHost.Data(), port.Data(),
                            opts.Data(), nullptr, dbname,
                            usr.Data(), pwd.Data());

      if (PQstatus(fPgSQL) == CONNECTION_OK) {
         fDB=dbname;
         fPort=port.Atoi();
      } else {
         Error("SelectDataBase", "%s",PQerrorMessage(fPgSQL));
         return -1;
      }
   }
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// List all available databases. Wild is for wildcarding "t%" list all
/// databases starting with "t".
/// Returns a pointer to a TSQLResult object if successful, 0 otherwise.
/// The result object must be deleted by the user.

TSQLResult *TPgSQLServer::GetDataBases(const char *wild)
{
   if (!IsConnected()) {
      Error("GetDataBases", "not connected");
      return nullptr;
   }

   TString sql = "SELECT pg_database.datname FROM pg_database";
   if (wild && *wild)
      sql += TString::Format(" WHERE pg_database.datname LIKE '%s'", wild);

   return Query(sql.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// List all tables in the specified database. Wild is for wildcarding
/// "t%" list all tables starting with "t".
/// Returns a pointer to a TSQLResult object if successful, 0 otherwise.
/// The result object must be deleted by the user.

TSQLResult *TPgSQLServer::GetTables(const char *dbname, const char *wild)
{
   if (!IsConnected()) {
      Error("GetTables", "not connected");
      return nullptr;
   }

   if (SelectDataBase(dbname) != 0) {
      Error("GetTables", "no such database %s", dbname);
      return nullptr;
   }

   TString sql = "SELECT relname FROM pg_class where relkind='r'";
   if (wild && *wild)
      sql += TString::Format(" AND relname LIKE '%s'", wild);

   return Query(sql.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// List all columns in specified table in the specified database.
/// Wild is for wildcarding "t%" list all columns starting with "t".
/// Returns a pointer to a TSQLResult object if successful, 0 otherwise.
/// The result object must be deleted by the user.

TSQLResult *TPgSQLServer::GetColumns(const char *dbname, const char *table,
                                     const char *wild)
{
   if (!IsConnected()) {
      Error("GetColumns", "not connected");
      return nullptr;
   }

   if (SelectDataBase(dbname) != 0) {
      Error("GetColumns", "no such database %s", dbname);
      return nullptr;
   }

   TString sql;
   if (wild && *wild)
      sql.Form("select a.attname,t.typname,a.attnotnull \
                from pg_attribute a, pg_class c, pg_type t \
                where c.oid=a.attrelid and c.relname='%s' and \
                a.atttypid=t.oid and a.attnum>0 \
                and a.attname like '%s' order by a.attnum ", table, wild);
   else
      sql.Form("select a.attname,t.typname,a.attnotnull \
                from pg_attribute a, pg_class c, pg_type t \
                where c.oid=a.attrelid and c.relname='%s' and \
                a.atttypid=t.oid and a.attnum>0 order by a.attnum", table);

   return Query(sql.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Create a database. Returns 0 if successful, non-zero otherwise.

Int_t TPgSQLServer::CreateDataBase(const char *dbname)
{
   if (!IsConnected()) {
      Error("CreateDataBase", "not connected");
      return -1;
   }
   TString sql;
   sql.Form("CREATE DATABASE %s", dbname);
   PGresult *res = PQexec(fPgSQL, sql.Data());
   PQclear(res);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Drop (i.e. delete) a database. Returns 0 if successful, non-zero
/// otherwise.

Int_t TPgSQLServer::DropDataBase(const char *dbname)
{
   if (!IsConnected()) {
      Error("DropDataBase", "not connected");
      return -1;
   }
   TString sql;
   sql.Form("DROP DATABASE %s", dbname);
   PGresult *res = PQexec(fPgSQL, sql.Data());
   PQclear(res);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Reload permission tables. Returns 0 if successful, non-zero
/// otherwise. User must have reload permissions.

Int_t TPgSQLServer::Reload()
{
   if (!IsConnected()) {
      Error("Reload", "not connected");
      return -1;
   }

   Error("Reload", "not implemented");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Shutdown the database server. Returns 0 if successful, non-zero
/// otherwise. User must have shutdown permissions.

Int_t TPgSQLServer::Shutdown()
{
   if (!IsConnected()) {
      Error("Shutdown", "not connected");
      return -1;
   }

   Error("Shutdown", "not implemented");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return server info.

const char *TPgSQLServer::ServerInfo()
{
   if (!IsConnected()) {
      Error("ServerInfo", "not connected");
      return nullptr;
   }

   return fSrvInfo.Data();
}

////////////////////////////////////////////////////////////////////////////////
/// PG_VERSION_NUM conveniently only started being #%defined at 8.2.3
/// which is the first version of libpq which explicitly supports prepared
/// statements

Bool_t TPgSQLServer::HasStatement() const
{
#ifdef PG_VERSION_NUM
   return kTRUE;
#else
   return kFALSE;
#endif
}

////////////////////////////////////////////////////////////////////////////////
/// Produce TPgSQLStatement.

#ifdef PG_VERSION_NUM
TSQLStatement* TPgSQLServer::Statement(const char *sql, Int_t)
#else
TSQLStatement* TPgSQLServer::Statement(const char *, Int_t)
#endif
{
#ifdef PG_VERSION_NUM
   if (!sql || !*sql) {
      SetError(-1, "no query string specified","Statement");
      return nullptr;
   }

   PgSQL_Stmt_t *stmt = new PgSQL_Stmt_t;
   if (!stmt){
      SetError(-1, "cannot allocate PgSQL_Stmt_t", "Statement");
      return nullptr;
   }
   stmt->fConn = fPgSQL;
   stmt->fRes  = PQprepare(fPgSQL, "preparedstmt", sql, 0, (const Oid*)0);

   ExecStatusType stat = PQresultStatus(stmt->fRes);
   if (pgsql_success(stat)) {
      fErrorOut = stat;
      return new TPgSQLStatement(stmt, fErrorOut);
   } else {
      SetError(stat, PQresultErrorMessage(stmt->fRes), "Statement");
      stmt->fConn = nullptr;
      delete stmt;
      return nullptr;
   }
#else
   Error("Statement", "not implemented for pgsql < 8.2");
#endif
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Produce TSQLTableInfo.

TSQLTableInfo *TPgSQLServer::GetTableInfo(const char* tablename)
{
   if (!IsConnected()) {
      Error("GetColumns", "not connected");
      return nullptr;
   }

   // Check table name
   if (!tablename || (*tablename==0))
      return nullptr;

   // Query first row ( works same way as MySQL)
   PGresult *res = PQexec(fPgSQL, TString::Format("SELECT * FROM %s LIMIT 1;", tablename));

   if ((PQresultStatus(res) != PGRES_COMMAND_OK) &&
       (PQresultStatus(res) != PGRES_TUPLES_OK)) {
      Error("Query", "%s",PQresultErrorMessage(res));
      PQclear(res);
      return nullptr;
   }

   if (fOidTypNameMap.empty()) {
      // Oid-TypNameMap empty, populate it, stays valid at least for connection
      // lifetime.
      PGresult *res_type = PQexec(fPgSQL, "SELECT OID, TYPNAME FROM PG_TYPE;");

      if ((PQresultStatus(res_type) != PGRES_COMMAND_OK) &&
          (PQresultStatus(res_type) != PGRES_TUPLES_OK)) {
         Error("Query", "%s", PQresultErrorMessage(res_type));
         PQclear(res);
         PQclear(res_type);
         return nullptr;
      }

      Int_t nOids = PQntuples(res_type);
      for (Int_t oid=0; oid<nOids; oid++) {
         Int_t tOid;
         char* oidString  = PQgetvalue(res_type, oid, 0);
         char* typeString = PQgetvalue(res_type, oid, 1);
         if (sscanf(oidString, "%10d", &tOid) != 1) {
            Error("GetTableInfo", "Bad non-numeric oid '%s' for type '%s'", oidString, typeString);
         }
         fOidTypNameMap[tOid]=std::string(typeString);
      }
      PQclear(res_type);
   }

   TList * lst = nullptr;

   Int_t nfields  = PQnfields(res);

   for (Int_t col=0;col<nfields;col++){
      Int_t sqltype = kSQL_NONE;
      Int_t data_size = -1;    // size in bytes
      Int_t data_length = -1;  // declaration like VARCHAR(n) or NUMERIC(n)
      Int_t data_scale = -1;   // second argument in declaration
      Int_t data_sign = -1;    // signed type or not
      Bool_t nullable = 0;

      const char* column_name = PQfname(res,col);
      const char* type_name;
      int   imod     = PQfmod(res,col);
      //int   isize    = PQfsize(res,col);

      int oid_code = PQftype(res,col);

      // Search for oid in map
      std::map<Int_t,std::string>::iterator lookupOid = fOidTypNameMap.find(oid_code);
      if (lookupOid == fOidTypNameMap.end()) {
         // Not found.
         //default
         sqltype = kSQL_NUMERIC;
         type_name = "NUMERIC";
         data_size=-1;
      } else if (lookupOid->second == "int2"){
         sqltype = kSQL_INTEGER;
         type_name = "INT";
         data_size=2;
      } else if (lookupOid->second == "int4"){
         sqltype = kSQL_INTEGER;
         type_name = "INT";
         data_size=4;
      } else if (lookupOid->second == "int8"){
         sqltype = kSQL_INTEGER;
         type_name = "INT";
         data_size=8;
      } else if (lookupOid->second == "float4"){
         sqltype = kSQL_FLOAT;
         type_name = "FLOAT";
         data_size=4;
      } else if (lookupOid->second == "float8"){
         sqltype = kSQL_DOUBLE;
         type_name = "DOUBLE";
         data_size=8;
      } else if (lookupOid->second == "bool"){
         sqltype = kSQL_INTEGER;
         type_name = "INT";
         data_size=1;
      } else if (lookupOid->second == "char"){
         sqltype = kSQL_CHAR;
         type_name = "CHAR";
         data_size=1;
      } else if (lookupOid->second == "varchar"){
         sqltype = kSQL_VARCHAR;
         type_name = "VARCHAR";
         data_size=imod;
      } else if (lookupOid->second == "text"){
         sqltype = kSQL_VARCHAR;
         type_name = "VARCHAR";
         data_size=imod;
      } else if (lookupOid->second == "name"){
         sqltype = kSQL_VARCHAR;
         type_name = "VARCHAR";
         data_size=imod;
      } else if (lookupOid->second == "date"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "time"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "timetz"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "timestamp"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "timestamptz"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "interval"){
         sqltype = kSQL_TIMESTAMP;
         type_name = "TIMESTAMP";
         data_size=8;
      } else if (lookupOid->second == "bytea"){
         sqltype = kSQL_BINARY;
         type_name = "BINARY";
         data_size=-1;
      } else if (lookupOid->second == ""){
         sqltype = kSQL_NONE;
         type_name = "UNKNOWN";
         data_size=-1;
      } else{
         //default
         sqltype = kSQL_NUMERIC;
         type_name = "NUMERIC";
         data_size=-1;
      }

      if (!lst)
         lst = new TList;

      lst->Add(new TSQLColumnInfo(column_name,
                                  type_name,
                                  nullable,
                                  sqltype,
                                  data_size,
                                  data_length,
                                  data_scale,
                                  data_sign));
   } //! ( cols)

   PQclear(res);
   return (new TSQLTableInfo(tablename,lst));
}
