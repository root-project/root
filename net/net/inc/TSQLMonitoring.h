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

#include "TVirtualMonitoring.h"
#include "TString.h"


class TSQLServer;


class TSQLMonitoringWriter : public TVirtualMonitoringWriter {

private:
   TSQLServer  *fDB;              // SQL database where to write
   TString      fTable;           // SQL table name

   Long64_t     fMaxBulkSize;     // Max packet size for insertions

   Bool_t       fVerbose;         // Verbosity toggle

   TSQLMonitoringWriter(const TSQLMonitoringWriter&) = delete;
   TSQLMonitoringWriter& operator=(const TSQLMonitoringWriter&) = delete;

public:
   TSQLMonitoringWriter(const char *serv, const char *user, const char *pass, const char *table);
   virtual ~TSQLMonitoringWriter();

   Bool_t SendParameters(TList *values, const char * /*identifier*/) override;

   void Verbose(Bool_t onoff) override { fVerbose = onoff; }

   ClassDefOverride(TSQLMonitoringWriter, 0)   // Sending monitoring data to a SQL DB
};

#endif
