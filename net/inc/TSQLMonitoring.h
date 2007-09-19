// @(#)root/net:$Id$
// Author: J.F. Grosse-Oetringhaus, G.Ganis

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLMonitoring
#define ROOT_TSQLMonitoring

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLMonitoringWriter                                                 //
//                                                                      //
// SQL implementation of TVirtualMonitoringWriter.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TVirtualMonitoring
#include "TVirtualMonitoring.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TSQLServer;


class TSQLMonitoringWriter : public TVirtualMonitoringWriter {

private:
   TSQLServer  *fDB;              // SQL database where to write
   TString      fTable;           // SQL table name

   TSQLMonitoringWriter(const TSQLMonitoringWriter&);            // not implemented
   TSQLMonitoringWriter& operator=(const TSQLMonitoringWriter&); // not implemented

public:
   TSQLMonitoringWriter(const char *serv, const char *user, const char *pass, const char *table);
   virtual ~TSQLMonitoringWriter();

   Bool_t SendParameters(TList *values, const char * /*identifier*/);

   ClassDef(TSQLMonitoringWriter, 0)   // Sending monitoring data to a SQL DB
};

#endif
