// @(#)root/net:$Name:  $:$Id: TSQLServer.cxx,v 1.3 2001/08/24 16:34:18 rdm Exp $
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
#include "TROOT.h"
#include "TPluginManager.h"


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
      serv = (TSQLServer *) gROOT->ProcessLineFast(Form(
             "new %s(\"%s\", \"%s\", \"%s\")", h->GetClass(), db, uid, pw));
   }
   return serv;
}


