// @(#)root/net:$Id$
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLServer
#define ROOT_TSQLServer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLServer                                                           //
//                                                                      //
// Abstract base class defining interface to a SQL server.              //
//                                                                      //
// To open a connection to a server use the static method Connect().    //
// The db argument of Connect() is of the form:                         //
//    <dbms>://<host>[:<port>][/<database>], e.g.                       //
// mysql://pcroot.cern.ch:3456/test, oracle://srv1.cern.ch/main, ...    //
// Depending on the <dbms> specified an appropriate plugin library      //
// will be loaded which will provide the real interface.                //
//                                                                      //
// Related classes are TSQLStatement, TSQLResult and TSQLRow.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TSQLResult;
class TSQLStatement;
class TSQLTableInfo;
class TList;

class TSQLServer : public TObject {

protected:
   TString   fType;       // type of DBMS (MySQL, Oracle, SysBase, ...)
   TString   fHost;       // host to which we are connected
   TString   fDB;         // currently selected DB
   Int_t     fPort;       // port to which we are connected
   Int_t     fErrorCode;  // error code of last operation
   TString   fErrorMsg;   // error message of last operation
   Bool_t    fErrorOut;   // enable error output

   TSQLServer()
     : fType(), fHost(), fDB(), fPort(-1), fErrorCode(0),
     fErrorMsg(), fErrorOut(kTRUE) { ClearError(); }

   void                ClearError();
   void                SetError(Int_t code, const char* msg, const char* method = 0);

   static const char* fgFloatFmt;          //!  printf argument for floats and doubles, either "%f" or "%e" or "%10f" and so on

public:
   enum ESQLDataTypes {  // data types, recognised by TSQLServer and other classes, extrction from ODBC
      kSQL_NONE = -1,      // data type unknown
      kSQL_CHAR = 1,       // CHAR(n) - string with fixed length n
      kSQL_VARCHAR = 2,    // VARCHAR(n) - string with variable length upto n
      kSQL_INTEGER = 3,    // INTEGER, INT - integer value
      kSQL_FLOAT = 4,      // FLOAT - float value
      kSQL_DOUBLE = 5,     // DOUBLE - double value
      kSQL_NUMERIC = 6,    // NUMERIC - numeric values with length and precion
      kSQL_BINARY = 7,     // BLOB - binary data
      kSQL_TIMESTAMP = 8   // TIMESTAMP -
   };

   virtual ~TSQLServer() { }

   virtual void        Close(Option_t *option="") = 0;
   virtual TSQLResult *Query(const char *sql) = 0;
   virtual Bool_t      Exec(const char* sql);
   virtual TSQLStatement *Statement(const char*, Int_t = 100)
                           { AbstractMethod("Statement"); return 0; }
   virtual Bool_t      HasStatement() const { return kFALSE; }
   virtual Int_t       SelectDataBase(const char *dbname) = 0;
   virtual TSQLResult *GetDataBases(const char *wild = 0) = 0;
   virtual TSQLResult *GetTables(const char *dbname, const char *wild = 0) = 0;
   virtual TList      *GetTablesList(const char* wild = 0);
   virtual Bool_t      HasTable(const char* tablename);
   virtual TSQLTableInfo *GetTableInfo(const char* tablename);
   virtual TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0) = 0;
   virtual Int_t       GetMaxIdentifierLength() { return 20; }
   virtual Int_t       CreateDataBase(const char *dbname) = 0;
   virtual Int_t       DropDataBase(const char *dbname) = 0;
   virtual Int_t       Reload() = 0;
   virtual Int_t       Shutdown() = 0;
   virtual const char *ServerInfo() = 0;
   virtual Bool_t      IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }
   const char         *GetDBMS() const { return fType.Data(); }
   const char         *GetDB() const { return fDB.Data(); }
   const char         *GetHost() const { return fHost.Data(); }
   Int_t               GetPort() const { return fPort; }

   virtual Bool_t      IsError() const { return GetErrorCode()!=0; }
   virtual Int_t       GetErrorCode() const;
   virtual const char* GetErrorMsg() const;
   virtual void        EnableErrorOutput(Bool_t on = kTRUE) { fErrorOut = on; }

   virtual Bool_t      StartTransaction();
   virtual Bool_t      Commit();
   virtual Bool_t      Rollback();

   virtual Bool_t      PingVerify() { return kFALSE; }
   virtual Int_t       Ping() { return -9999; }

   static TSQLServer *Connect(const char *db, const char *uid, const char *pw);

   static    void     SetFloatFormat(const char* fmt = "%e");
   static const char* GetFloatFormat();

   ClassDef(TSQLServer,0)  // Connection to SQL server
};

#endif
