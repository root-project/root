// @(#)root/proofplayer:$Id$
// Author: G.Ganis July 2011

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofMonSenderML
#define ROOT_TProofMonSenderML

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofMonSenderML                                                    //
//                                                                      //
// TProofMonSender implementation for the MonaLisa writer.              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TProofMonSender
#include "TProofMonSender.h"
#endif

class TDSet;
class TList;
class TPerfStat;
class TVirtualMonitoringWriter;

class TProofMonSenderML : public TProofMonSender {

private:
   TVirtualMonitoringWriter *fWriter; // Writer instance connect to backend

public:

   TProofMonSenderML(const char *serv, const char *tag, const char *id = 0,
                     const char *subid = 0, const char *opt = "");
   virtual ~TProofMonSenderML();
   
   // Summary record
   Int_t SendSummary(TList *, const char *);

   // Information about the dataset(s) processed
   Int_t SendDataSetInfo(TDSet *, TList *, const char *, const char *);

   // Detailed infoirmation about files
   Int_t SendFileInfo(TDSet *, TList *, const char *, const char *);
   
   ClassDef(TProofMonSenderML, 0); // Interface for PROOF monitoring
};

#endif
