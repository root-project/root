// @(#)root/proofplayer:$Name:  $:$Id:$
// Author: J.F. Grosse-Oetringhaus, G.Ganis

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSQLWriter
#define ROOT_TSQLWriter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLWriter                                                           //
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


class TSQLWriter : public TVirtualMonitoringWriter {

private:
   TSQLServer  *fDB;              // SQL database where to write
   TString      fTable;           // SQL table name

   TSQLWriter(const TSQLWriter&);            // not implemented
   TSQLWriter& operator=(const TSQLWriter&); // not implemented

public:
   TSQLWriter(const char *serv, const char *user, const char *pass, const char *table);
   virtual ~TSQLWriter();

   Bool_t SendParameters(TList *values, const char * /*identifier*/);

   ClassDef(TSQLWriter, 0)   // Interface to SQL Monitoring
};

#endif
