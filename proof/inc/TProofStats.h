// @(#)root/proof:$Name:  $:$Id: TProofStats.h,v 1.1 2004/05/18 11:32:49 rdm Exp $
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofStats
#define ROOT_TProofStats

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofStats                                                          //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing                                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TTimeStamp
#include "TTimeStamp.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TTree;
class TH1D;
class TH2D;
class TList;


class TProofEvent : public TObject {

public:
   enum EEventType { kUnDefined, kPacket, kStart, kStop, kFile };

   EEventType  fType;
   TString     fSlaveName;
   TString     fNodeName;
   TString     fFileName;
   Int_t       fSlave;
   Long64_t    fEventsProcessed;
   Double_t    fLatency;
   Double_t    fProcTime;
   Double_t    fCpuTime;
   TTimeStamp  fTimeStamp;
   Bool_t      fIsStart;

   TProofEvent()
     : fType(kUnDefined), fSlave(0), fEventsProcessed(0), fLatency(0.0),
       fProcTime(0.0), fCpuTime(0.0), fIsStart(kFALSE)
   {
   }

   virtual ~TProofEvent() {}

   ClassDef(TProofEvent,1) // Class holding TProof Event Info
};


class TProofStats : public TObject {

private:
   TTree         *fTrace;        //!TTree with PROOF trace events
   TProofEvent   *fProofEvent;   //!TProofEvent used to fill tree
   TH1D          *fPacketsHist;  //!histogram of packets processed per slave
   TH1D          *fEventsHist;   //!histogram of events processed per slave
   TH1D          *fNodeHist;     //!histogram of slaves per file serving node
   TH2D          *fLatencyHist;  //!histogram of latency due to packet requests
   TH2D          *fProcTimeHist; //!histogram of real time spent processing packets
   TH2D          *fCpuTimeHist;  //!histogram of cpu time spent processing packets

public:
   TProofStats(Int_t nslaves, TList *output, Bool_t doHist, Bool_t doTrace);
   virtual ~TProofStats() {}

   void SimpleEvent(TProofEvent::EEventType type);
   void PacketEvent(Int_t slave, const char *slavename, const char *filename,
                    Long64_t eventsprocessed, Double_t latency, Double_t proctime, Double_t cputime);
   void FileEvent(Int_t slave, const char *slavename, const char *nodename, const char *filename,
                  Bool_t isStart);
   
   
   ClassDef(TProofStats,0)  // Class for collecting PROOF statistics
};

#endif
