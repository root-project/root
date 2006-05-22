// @(#)root/net:$Name:  $:$Id: TSQLServer.cxx,v 1.9 2005/07/12 15:57:08 rdm Exp $
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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

#include "TSQLServer.h"
#include "TSQLResult.h"
#include "TROOT.h"
#include "TPluginManager.h"
#include "TVirtualMutex.h"

ClassImp(TSQLServer)

//______________________________________________________________________________
TSQLServer *TSQLServer::Connect(const char *db, const char *uid, const char *pw)
{
   // The db should be of the form:  <dbms>://<host>[:<port>][/<database>],
   // e.g.:  mysql://pcroot.cern.ch:3456/test, oracle://srv1.cern.ch/main,
   // pgsql://... or sapdb://...
   // The uid is the username and pw the password that should be used for
   // the connection. Depending on the <dbms> the shared library (plugin)
   // for the selected system will be loaded. When the connection could not
   // be opened 0 is returned.

   TPluginHandler *h;
   TSQLServer *serv = 0;

   if ((h = gROOT->GetPluginManager()->FindHandler("TSQLServer", db))) {
      if (h->LoadPlugin() == -1)
         return 0;
      serv = (TSQLServer *) h->ExecPlugin(3, db, uid, pw);
   }

   if (serv && serv->IsZombie()) {
      delete serv;
      serv = 0;
   }

   return serv;
}

//______________________________________________________________________________
Int_t TSQLServer::GetErrorCode() const
{
   // returns error code of last operation
   // if res==0, no error
   // Each specific implementation of TSQLServer provides its own error coding
   
   return fErrorCode;
}

//______________________________________________________________________________
const char* TSQLServer::GetErrorMsg() const
{
   //  returns error message of last operation
   // if no errors, return 0
   // Each specific implementation of TSQLServer provides its own error messages
   
   return GetErrorCode()==0 ? 0 : fErrorMsg.Data();
}

//______________________________________________________________________________
void TSQLServer::ClearError()
{
   // reset error fields
   
   fErrorCode = 0;
   fErrorMsg = "";
}

//______________________________________________________________________________
void TSQLServer::SetError(Int_t code, const char* msg, const char* method)
{
   // set new values for error fields
   // if method is specified, displays error message
   
   fErrorCode = code;
   fErrorMsg = msg;
   if ((method!=0) && fErrorOut)
      Error(method,"Code: %d  Msg: %s", code, msg);
}

//______________________________________________________________________________
Bool_t TSQLServer::StartTransaction()
{
   // submit "START TRANSACTION" query to database
   // return kTRUE, if succesfull
   
   TSQLResult* res = Query("START TRANSACTION");
   if (res!=0) delete res;
   return !IsError();
}

//______________________________________________________________________________
Bool_t TSQLServer::Commit()
{
   // submit "COMMIT" query to database
   // return kTRUE, if succesfull

   TSQLResult* res = Query("COMMIT");
   if (res!=0) delete res;
   return !IsError();
}

//______________________________________________________________________________
Bool_t TSQLServer::Rollback()
{
   // submit "ROLLBACK" query to database
   // return kTRUE, if succesfull

   TSQLResult* res = Query("ROLLBACK");
   if (res!=0) delete res;
   return !IsError();
}
