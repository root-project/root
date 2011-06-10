// @(#)root/base:$Id$
// Author: Kristjan Gulbrandsen   11/05/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVirtualPerfStats
#define ROOT_TVirtualPerfStats

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualPerfStats                                                    //
//                                                                      //
// Provides the interface for the PROOF internal performance measurment //
// and event tracing.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TFile;


class TVirtualPerfStats : public TObject {

public:
   virtual ~TVirtualPerfStats() {}

   enum EEventType {
      kUnDefined,
      kPacket,       //info of single packet processing
      kStart,        //begin of run
      kStop,         //end of run
      kFile,         //file started/finished in packetizer
      kFileOpen,     //opening data file statistics
      kFileRead,     //reading data file event
      kRate,         //processing {evt, MB} rates
      kNumEventType  //number of entries, must be last
   };

   virtual void SimpleEvent(EEventType type) = 0;

   virtual void PacketEvent(const char *slave, const char *slavename, const char *filename,
                            Long64_t eventsprocessed, Double_t latency,
                            Double_t proctime, Double_t cputime,
                            Long64_t bytesRead) = 0;

   virtual void FileEvent(const char *slave, const char *slavename, const char *nodename,
                          const char *filename, Bool_t isStart) = 0;

   virtual void FileOpenEvent(TFile *file, const char *filename, Double_t start) = 0;

   virtual void FileReadEvent(TFile *file, Int_t len, Double_t start) = 0;

   virtual void FileUnzipEvent(TFile *file, Long64_t pos, Double_t start, Int_t complen, Int_t objlen) = 0;

   virtual void RateEvent(Double_t proctime, Double_t deltatime,
                          Long64_t eventsprocessed, Long64_t bytesRead) = 0;

   virtual void SetBytesRead(Long64_t num) = 0;
   virtual Long64_t GetBytesRead() const = 0;
   virtual void SetNumEvents(Long64_t num) = 0;
   virtual Long64_t GetNumEvents() const = 0;

   static const char *EventType(EEventType type);

   ClassDef(TVirtualPerfStats,0)  // ABC for collecting PROOF statistics
};


R__EXTERN TVirtualPerfStats *gPerfStats;


#endif
