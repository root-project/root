// @(#)root/monalisa:$Name:  $:$Id: TMonaLisaWriter.h,v 1.2 2006/10/05 16:10:22 rdm Exp $
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

#ifndef ROOT_TVirtualMonitoring
#include "TVirtualMonitoring.h"
#endif
#ifndef ROOT_TStopwatch
#include "TStopwatch.h"
#endif

#ifndef __CINT__
#include <ApMon.h>
#else
struct ApMon;
#endif

#include <time.h>

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
   time_t     fLastSendTime;     // timestamp of the last send command for file reads
   time_t     fLastProgressTime; // timestamp of the last send command for player process
   time_t     fReportInterval;   // interval after which to send the latest value
   TStopwatch fStopwatch;        // cpu time measurement

public:
   TMonaLisaWriter(const char *monid = 0, const char *montag = 0,
                   const char *monserver = 0);

   TMonaLisaWriter(const char *monid , const char* monsubid , const char *montag,
                   const char *monserver );

   void Init(const char *monid , const char* monsubid , const char *montag,
                   const char *monserver );

   virtual ~TMonaLisaWriter();

   ApMon *GetApMon() const { return fApmon; }

   Bool_t SendParameters(TList *valuelist);
   Bool_t SendInfoTime();
   Bool_t SendInfoUser(const char *user = 0);
   Bool_t SendInfoDescription(const char *jobtag);
   Bool_t SendInfoStatus(const char *status);
   Bool_t SendFileReadProgress(TFile *file, Bool_t force=kFALSE);
   Bool_t SendProcessingStatus(const char *status, Bool_t restarttimer=kFALSE);
   Bool_t SendProcessingProgress(Double_t nevent, Double_t nbytes, Bool_t force=kFALSE);
   void   SetLogLevel(const char *loglevel = "WARNING");
   void   Verbose(Bool_t onoff) { fVerbose = onoff; }

   void   Print(Option_t *option = "") const;

   ClassDef(TMonaLisaWriter, 1)   // Interface to MonaLisa Monitoring
};

#endif
