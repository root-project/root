// @(#)root/proofplayer:$Id$
// Author: G.Ganis July 2011

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofMonSenderSQL
#define ROOT_TProofMonSenderSQL

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMonSenderSQL                                                   //
//                                                                      //
// TProofMonSender implementation for SQL writers.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofMonSender
#include "TProofMonSender.h"
#endif

class TDSet;
class TList;
class TPerfStat;
class TVirtualMonitoringWriter;

class TProofMonSenderSQL : public TProofMonSender {

private:
   TVirtualMonitoringWriter *fWriter; // Writer instance connect to backend
   TString                   fDSetSendOpts; // Opts for posting dataset table
   TString                   fFilesSendOpts; // Opts for posting files table

public:

   TProofMonSenderSQL(const char *serv, const char *user, const char *pass,
                      const char *table = "proof.proofquerylog",
                      const char *dstab = 0, const char *filestab = 0);
   virtual ~TProofMonSenderSQL();

   // Summary record
   Int_t SendSummary(TList *, const char *);

   // Information about the dataset(s) processed
   Int_t SendDataSetInfo(TDSet *, TList *, const char *, const char *);

   // Detailed information about files
   Int_t SendFileInfo(TDSet *, TList *, const char *, const char *);

   ClassDef(TProofMonSenderSQL, 0); // Interface for PROOF monitoring
};

#endif
