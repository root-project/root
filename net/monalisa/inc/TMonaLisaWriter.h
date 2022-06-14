// @(#)root/monalisa:$Id$
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMonaLisaWriter
#define ROOT_TMonaLisaWriter

#include "TVirtualMonitoring.h"
#include "TStopwatch.h"

#ifndef __CINT__
#include <ApMon.h>
#else
struct ApMon;
#endif

#include <time.h>
#include <map>

class MonitoredTFileInfo;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMonaLisaWriter                                                      //
//                                                                      //
// Class defining interface to MonaLisa Monitoring Services in ROOT.    //
// The TMonaLisaWriter object is used to send monitoring information to //
// a MonaLisa server using the ML ApMon package (libapmoncpp.so/UDP     //
// packets). The MonaLisa ApMon library for C++ can be downloaded at    //
// http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,     //
// current version:                                                     //
//http://monalisa.cacr.caltech.edu/download/apmon/ApMon_cpp-2.2.0.tar.gz//
//                                                                      //
// The ROOT implementation is primary optimized for process/job         //
// monitoring, although all other generic MonaLisa ApMon functionality  //
// can be exploited through the ApMon class directly via                //
// dynamic_cast<TMonaLisaWriter*>(gMonitoringWriter)->GetApMon().       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TMonaLisaValue : public TNamed {

private:
   Double_t fValue;  // double monitor value

   TMonaLisaValue(const TMonaLisaValue&); // Not implented
   TMonaLisaValue& operator=(const TMonaLisaValue&); // Not implented

public:
   TMonaLisaValue(const char *name, Double_t value)
      : TNamed(name, ""), fValue(value) { }
   virtual ~TMonaLisaValue() { }

   Double_t  GetValue() const { return fValue; }
   Double_t *GetValuePtr() { return &fValue; }

   ClassDef(TMonaLisaValue, 1)  // Interface to MonaLisa Monitoring Values
};


class TMonaLisaText : public TNamed {

public:
   TMonaLisaText(const char *name, const char *text) : TNamed(name, text) { }
   virtual ~TMonaLisaText() { }

   const char *GetText() const { return GetTitle(); }

   ClassDef(TMonaLisaText, 1)   // Interface to MonaLisa Monitoring Text
};


class TMonaLisaWriter : public TVirtualMonitoringWriter {

private:
   ApMon     *fApmon;            //! connection to MonaLisa
   TString    fJobId;            //! job id
   TString    fSubJobId;         //! sub job id
   TString    fHostname;         //! hostname of MonaLisa server
   Int_t      fPid;              //! process id
   Bool_t     fInitialized;      // true if initialized
   Bool_t     fVerbose;          // verbocity
   Double_t   fLastRWSendTime;     // timestamp of the last send command for file reads/writes
   Double_t   fLastFCloseSendTime; // In order not to flood ML servers
   time_t     fLastProgressTime; // timestamp of the last send command for player process

   std::map<UInt_t,  MonitoredTFileInfo *>   //!
             *fMonInfoRepo;      //! repo to gather per-file-instance mon info;
                                 // ROOT should really have something like this

   Int_t      fReportInterval;   // interval after which to send the latest value

   TStopwatch fStopwatch;        // cpu and time measurement for job and proc status
   TStopwatch fFileStopwatch;     // time measurements for data access throughputs

   TMonaLisaWriter(const TMonaLisaWriter&); // Not implemented
   TMonaLisaWriter& operator=(const TMonaLisaWriter&); // Not implemented

   void Init(const char *monserver, const char *montag, const char *monid,
             const char *monsubid, const char *option);

   Bool_t SendFileCheckpoint(TFile *file);
public:
   TMonaLisaWriter(const char *monserver, const char *montag, const char *monid = 0,
                   const char *monsubid = 0, const char *option = "");

   virtual ~TMonaLisaWriter();

   ApMon *GetApMon() const { return fApmon; }

   virtual Bool_t SendParameters(TList *valuelist, const char *identifier = 0);
   virtual Bool_t SendInfoTime();
   virtual Bool_t SendInfoUser(const char *user = 0);
   virtual Bool_t SendInfoDescription(const char *jobtag);
   virtual Bool_t SendInfoStatus(const char *status);

   virtual Bool_t SendFileCloseEvent(TFile *file);

   // An Open might have several phases, and the timings might be interesting
   // to report
   // The info is only gathered, and sent when forcesend=kTRUE
   virtual Bool_t SendFileOpenProgress(TFile *file, TList *openphases, const char *openphasename,
                               Bool_t forcesend = kFALSE);

   virtual Bool_t SendFileReadProgress(TFile *file);
   virtual Bool_t SendFileWriteProgress(TFile *file);

   virtual Bool_t SendProcessingStatus(const char *status, Bool_t restarttimer=kFALSE);
   virtual Bool_t SendProcessingProgress(Double_t nevent, Double_t nbytes, Bool_t force=kFALSE);
   virtual void   SetLogLevel(const char *loglevel = "WARNING");
   virtual void   Verbose(Bool_t onoff) { fVerbose = onoff; }

   void   Print(Option_t *option = "") const;

   ClassDef(TMonaLisaWriter, 1)   // Interface to MonaLisa Monitoring
};

#endif
