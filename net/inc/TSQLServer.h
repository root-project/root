// @(#)root/net:$Name$:$Id$
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
// Related classes are TSQLResult and TSQLRow.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class  TSQLResult;


class TSQLServer : public TObject {

protected:
   TString   fType;    // type of DBMS (MySQL, Oracle, SysBase, ...)
   TString   fHost;    // host to which we are connected
   TString   fDB;      // currently selected DB
   Int_t     fPort;    // port to which we are connected

   TSQLServer() { fPort = -1; }

public:
   virtual ~TSQLServer() { }

   virtual void        Close(Option_t *option="") = 0;
   virtual TSQLResult *Query(const char *sql) = 0;
   virtual Int_t       SelectDataBase(const char *dbname) = 0;
   virtual TSQLResult *GetDataBases(const char *wild = 0) = 0;
   virtual TSQLResult *GetTables(const char *dbname, const char *wild = 0) = 0;
   virtual TSQLResult *GetColumns(const char *dbname, const char *table, const char *wild = 0) = 0;
   virtual Int_t       CreateDataBase(const char *dbname) = 0;
   virtual Int_t       DropDataBase(const char *dbname) = 0;
   virtual Int_t       Reload() = 0;
   virtual Int_t       Shutdown() = 0;
   virtual const char *ServerInfo() = 0;
   Bool_t              IsConnected() const { return fPort == -1 ? kFALSE : kTRUE; }
   const char         *GetDBMS() const { return fType.Data(); }
   const char         *GetHost() const { return fHost.Data(); }
   Int_t               GetPort() const { return fPort; }

   static TSQLServer *Connect(const char *db, const char *uid, const char *pw);

   ClassDef(TSQLServer,0)  // Connection to SQL server
};

#endif
