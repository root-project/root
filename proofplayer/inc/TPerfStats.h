// @(#)root/proofplayer:$Name:  $:$Id: TPerfStats.h,v 1.4 2006/11/15 17:45:54 rdm Exp $
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPerfStats
#define ROOT_TPerfStats

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPerfStats                                                           //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing.                                                   //
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
#ifndef ROOT_TVirtualPerfStats
#include "TVirtualPerfStats.h"
#endif


class TTree;
class TH1D;
class TH2D;
class TList;


class TPerfEvent : public TObject {

public:
   TString                       fEvtNode;   // node on which the event was generated
   TTimeStamp                    fTimeStamp; // time offset from start of run
   TVirtualPerfStats::EEventType fType;
   TString                       fSlaveName;
   TString                       fNodeName;
   TString                       fFileName;
   TString                       fFileClass;
   TString                       fSlave;
   Long64_t                      fEventsProcessed;
   Long64_t                      fBytesRead;
   Long64_t                      fLen;
   Double_t                      fLatency;
   Double_t                      fProcTime;
   Double_t                      fCpuTime;
   Bool_t                        fIsStart;
   Bool_t                        fIsOk;

   TPerfEvent(TTimeStamp *offset = 0);
   virtual ~TPerfEvent() {}

   Bool_t   IsSortable() const { return kTRUE; }
   Int_t    Compare(const TObject *obj) const;
   void     Print(Option_t *option="") const;

   ClassDef(TPerfEvent,3) // Class holding TProof Event Info
};


class TPerfStats : public TVirtualPerfStats {

private:
   TTree         *fTrace;        //!TTree with trace events
   TTimeStamp     fTzero;        //!start time of this run
   TPerfEvent    *fPerfEvent;    //!TPerfEvent used to fill tree
   TH1D          *fPacketsHist;  //!histogram of packets processed per slave
   TH1D          *fEventsHist;   //!histogram of events processed per slave
   TH1D          *fNodeHist;     //!histogram of slaves per file serving node
   TH2D          *fLatencyHist;  //!histogram of latency due to packet requests
   TH2D          *fProcTimeHist; //!histogram of real time spent processing packets
   TH2D          *fCpuTimeHist;  //!histogram of cpu time spent processing packets
   Long64_t       fBytesRead;    //!track bytes read of main file

   Bool_t         fDoHist;       //!Fill histos
   Bool_t         fDoTrace;      //!Trace details in master
   Bool_t         fDoTraceRate;  //!Trace processing rate in master
   Bool_t         fDoSlaveTrace; //!Full tracing in workers

   TPerfStats(TList *input, TList *output);

public:
   virtual ~TPerfStats() {}

   void SimpleEvent(EEventType type);
   void PacketEvent(const char *slave, const char *slavename, const char *filename,
                    Long64_t eventsprocessed, Double_t latency,
                    Double_t proctime, Double_t cputime, Long64_t bytesRead);
   void FileEvent(const char *slave, const char *slavename, const char *nodename, const char *filename,
                  Bool_t isStart);

   void FileOpenEvent(TFile *file, const char *filename, Double_t proctime);
   void FileReadEvent(TFile *file, Int_t len, Double_t proctime);
   void RateEvent(Double_t proctime, Double_t deltatime,
                  Long64_t eventsprocessed, Long64_t bytesRead);
   void SetBytesRead(Long64_t num);
   Long64_t GetBytesRead() const;

   static void Start(TList *input, TList *output);
   static void Stop();
   static void Setup(TList *input);

   ClassDef(TPerfStats,0)  // Class for collecting PROOF statistics
};


#endif
