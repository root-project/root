// @(#)root/monalisa:$Id$
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
// http://monalisa.cacr.caltech.edu/download/apmon/ApMon_c-2.2.0.tar.gz //
//                                                                      //
// The ROOT implementation is primary optimized for process/job         //
// monitoring, although all other generic MonaLisa ApMon functionality  //
// can be exploited through the ApMon class directly via                //
// dynamic_cast<TMonaLisaWriter*>(gMonitoringWriter)->GetApMon().       //
//                                                                      //
// Additions/modifications by Fabrizio Furano 10/04/2008                //
// - The implementation of TFile throughput and info sending was        //
// just sending 'regular' samples about the activity of the single TFile//
// instance that happened to trigger an activity in the right moment.   //
// - Now TMonaLisaWriter keeps internally track of every activity       //
// and regularly sends summaries valid for all the files which had      //
// activity in the last time interval.                                  //
// - Additionally, it's now finalized the infrastructure able to measure//
// and keep track of the file Open latency. A packet is sent for each   //
// successful Open, sending the measures of the latencies for the       //
// various phases of the open. Currently exploited fully by TAlienFile  //
// and TXNetFile. Easy to report from other TFiles too.                 //
// - Now, the hook for the Close() func triggers sending of a packet    //
// containing various information about the performance related to that //
// file only.                                                           //
// - Added support also for performance monitoring when writing         //
//////////////////////////////////////////////////////////////////////////

#include <cmath>
#include <pthread.h>
#include "TMonaLisaWriter.h"
#include "TSystem.h"
#include "TGrid.h"
#include "TFile.h"
#include "TUrl.h"
#include "TStopwatch.h"
#include "Riostream.h"
#include "TParameter.h"
#include "THashList.h"


ClassImp(TMonaLisaWriter);


// Information which is kept about an alive instance of TFile
class MonitoredTFileInfo: public TObject {
private:
   TFile       *fileinst;
public:
   MonitoredTFileInfo(TFile *file, Double_t timenow): TObject(), fileinst(file) {
      if (file->InheritsFrom("TXNetFile"))
         fFileClassName = "TXNetFile";
      else
         fFileClassName = file->ClassName();

      fLastBytesRead = 0;
      fLastBytesWritten = 0;

      fTempReadBytes = 0;
      fTempWrittenBytes = 0;

      fLastResetTime = timenow;
      fCreationTime = timenow;

      fKillme = kFALSE;
   }

   Double_t    fCreationTime;

   TString     fFileClassName;

   Long64_t    fLastBytesRead;
   Long64_t    fTempReadBytes;
   Long64_t    fLastBytesWritten;
   Long64_t    fTempWrittenBytes;

   Double_t    fLastResetTime;

   Bool_t      fKillme;   // tells to remove the instance after the next computation step

   void        GetThroughputs(Long64_t &readthr, Long64_t &writethr, Double_t timenow, Double_t prectime) {
      readthr = -1;
      writethr = -1;
      Double_t t = std::min(prectime, fLastResetTime);

      Int_t mselapsed = std::round(std::floor(((timenow - t) * 1000)));
      mselapsed = std::max(mselapsed, 1);

      readthr = fTempReadBytes / mselapsed * 1000;
      writethr = fTempWrittenBytes / mselapsed * 1000;
   }

   void        UpdateFileStatus(TFile *file) {
      fTempReadBytes = file->GetBytesRead() - fLastBytesRead;
      fTempWrittenBytes = file->GetBytesWritten() - fLastBytesWritten;
   }

   void        ResetFileStatus(TFile *file, Double_t timenow) {
      if (fKillme) return;
      fLastBytesRead = file->GetBytesRead();
      fLastBytesWritten = file->GetBytesWritten();
      fTempReadBytes = 0;
      fTempWrittenBytes = 0;
      fLastResetTime = timenow;
   }

   void        ResetFileStatus(Double_t timenow) {
      if (fKillme) return;
      ResetFileStatus(fileinst, timenow);
   }

};




// Helper used to build up ongoing throughput summaries
class MonitoredTFileSummary: public TNamed {
public:
   MonitoredTFileSummary(TString &fileclassname): TNamed(fileclassname, fileclassname) {
      fBytesRead = 0;
      fBytesWritten = 0;
      fReadThroughput = 0;
      fWriteThroughput = 0;
   }

   Long64_t    fBytesRead;
   Long64_t    fBytesWritten;
   Long64_t    fReadThroughput;
   Long64_t    fWriteThroughput;

   void        Update(MonitoredTFileInfo *mi, Double_t timenow, Double_t prectime) {
      Long64_t rth, wth;
      mi->GetThroughputs(rth, wth, timenow, prectime);

      fBytesRead += mi->fTempReadBytes;
      fBytesWritten += mi->fTempWrittenBytes;

      if (rth > 0) fReadThroughput += rth;
      if (wth > 0) fWriteThroughput += wth;
   }

};



////////////////////////////////////////////////////////////////////////////////
/// Create MonaLisa write object.

TMonaLisaWriter::TMonaLisaWriter(const char *monserver, const char *montag,
                                 const char *monid, const char *monsubid,
                                 const char *option)
{
   fMonInfoRepo = new std::map<UInt_t, MonitoredTFileInfo *>;

   Init(monserver, montag, monid, monsubid, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Creates a TMonaLisaWriter object to send monitoring information to a
/// MonaLisa server using the MonaLisa ApMon package (libapmoncpp.so/UDP
/// packets). The MonaLisa ApMon library for C++ can be downloaded at
/// http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,
/// current version:
/// http://monalisa.cacr.caltech.edu/download/apmon/ApMon_cpp-2.0.6.tar.gz
///
/// The ROOT implementation is primary optimized for process/job monitoring,
/// although all other generic MonaLisa ApMon functionality can be exploited
/// through the ApMon class directly (gMonitoringWriter->GetApMon()).
///
/// Monitoring information in MonaLisa is structured in the following tree
/// structure:
/// `<farmname>`
///    |
///    ---> `<nodename1>`
///              |
///              ---> `<key1>` - `<value1>`
///              ---> `<key2>` - `<value2>`
///    ---> `<nodename2>`
///              |
///              ---> `<key3>` - `<value3>`
///              ---> `<key4>` - `<value4>`
///
/// The parameter monid is equivalent to the MonaLisa node name, for the
/// case of process monitoring it can be just an identifier to classify
/// the type of jobs e.g. "PROOF_PROCESSING".
/// If monid is not specified, TMonaLisaWriter tries to set it in this order
/// from environment variables:
/// - PROOF_JOB_ID
/// - GRID_JOB_ID
/// - LCG_JOB_ID
/// - ALIEN_MASTERJOB_ID
/// - ALIEN_PROC_ID
///
/// The parameter montag is equivalent to the MonaLisa farm name, for the
/// case of process monitoring it can be a process identifier e.g. a PROOF
/// session ID.
///
/// The parameter monserver specifies the server to whom to send the
/// monitoring UDP packets. If not specified, the hostname (the port is
/// a default one) is specified in the environment variable APMON_CONFIG.
///
/// To use TMonaLisaWriter, libMonaLisa.so has to be loaded.
///
/// According to the fact, that the deepness of the MonaLisa naming scheme
/// is only 3 (`<farm>``<node>``<value>`), a special naming scheme is used for
/// process monitoring. There is a high-level method to send progress
/// information of Tree analysis (# of events, datasize).
/// To distinguish individual nodes running the processing, part of the
/// information is kept in the `<value>` parameter of ML.
/// `<value>` is named as:
///    `<site-name>`:`<host-name>`:`<pid>`:`<valuetag>`
/// `<site-name>` is taken from an environment variable in the following order:
/// - PROOF_SITE
/// - GRID_SITE
/// - ALIEN_SITE
/// - default 'none'
/// `<host-name>` is taken from gSystem->Hostname()
/// `<pid>` is the process ID of the ROOT process
///
/// Example of use for Process Monitoring:
///   new TMonaLisaWriter("BATCH_ANALYSIS","AnalysisLoop-00001","lxplus050.cern.ch");
/// Once when you create an analysis task, execute
///   gMonitoringWriter->SendInfoUser("myname");
///   gMonitoringWriter->SendInfoDescription("My first Higgs analysis");
///   gMonitoringWriter->SendInfoTime();
///   gMonitoringWriter->SendInfoStatus("Submitted");
///
/// On each node executing a subtask, you can set the status of this subtask:
///   gMonitoringWriter->SendProcessingStatus("Started");
/// During the processing of your analysis you can send progress updates:
///   gMonitoringWriter->SendProcessProgress(100,1000000); <= 100 events, 1MB processed
///   ....
///   gMonitoringWriter-SendProcessingStatus("Finished");
///   delete gMonitoringWriter; gMonitoringWriter=0;
///
/// Example of use for any Generic Monitoring information:
///   TList *valuelist = new TList();
///   valuelist->SetOwner(kTRUE);
///   // append a text object
///   TMonaLisaText *valtext = new TMonaLisaText("decaychannel","K->eeg");
///   valuelist->Add(valtext);
///   // append a double value
///   TMonaLisaValue* valdouble = new TMonaLisaValue("n-gamma",5);
///   valuelist->Add(valdouble);
///   Bool_t success = SendParameters(valuelist);
///   delete valuelist;
///
/// option:
/// "global": gMonitoringWriter is initialized with this instance

void TMonaLisaWriter::Init(const char *monserver, const char *montag, const char *monid,
                           const char *monsubid, const char *option)
{
   SetName(montag);
   SetTitle(montag);

   fVerbose = kFALSE;           // no verbosity as default

   fFileStopwatch.Start(kTRUE);
   fLastRWSendTime = fFileStopwatch.RealTime();
   fLastFCloseSendTime = fFileStopwatch.RealTime();
   fLastProgressTime = time(0);
   fFileStopwatch.Continue();

   fReportInterval = 120; // default interval is 120, to prevent flooding
   if (gSystem->Getenv("APMON_INTERVAL")) {
      fReportInterval = atoi(gSystem->Getenv("APMON_INTERVAL"));
      if (fReportInterval < 1)
         fReportInterval =1;
      Info("TMonaLisaWriter","Setting APMON Report Interval to %d seconds",fReportInterval);
   }

   char *apmon_config[1] =
      { ((monserver == 0) ? (char *) gSystem->Getenv("APMON_CONFIG") : (char *) monserver) };
   if (apmon_config[0] == 0) {
      Error("TMonaLisaWriter",
            "Disabling apmon monitoring since env variable APMON_CONFIG was not found and the monitoring server is not specified in the constructor!");
      fInitialized = kFALSE;
      return;
   }

   try {
      fApmon = new ApMon(1, apmon_config);
      fApmon->setConfRecheck(false);
      fApmon->setJobMonitoring(false);
      //((ApMon*)fApmon)->setSysMonitoring(false);
      //((ApMon*)fApmon)->setGenMonitoring(false);
   } catch (runtime_error &e) {
      Error("TMonaLisaWriter", "Error initializing ApMon: %s", e.what());
      Error("TMonaLisaWriter", "Disabling apmon.");
      fInitialized = kFALSE;
      return;
   }

   TString clustername="ROOT_";

   if (montag == 0) {
     if (gSystem->Getenv("PROOF_SITE")) {
       clustername+=(gSystem->Getenv("PROOF_SITE"));
     } else if (gSystem->Getenv("GRID_SITE")) {
       clustername+=(gSystem->Getenv("GRID_SITE"));
     } else if (gSystem->Getenv("LCG_SITE")) {
       clustername+=(gSystem->Getenv("LCG_SITE"));
     } else if (gSystem->Getenv("ALIEN_SITE")) {
       clustername+=(gSystem->Getenv("ALIEN_SITE"));
     } else {
       clustername += TString("none");
     }
     SetName(clustername);
     SetTitle(clustername);
   } else {
       SetName(clustername+TString(montag));
       SetTitle(clustername+TString(montag));
   }

   fHostname = gSystem->HostName();
   fPid = gSystem->GetPid();

   if (monid == 0) {
      if (gSystem->Getenv("PROOF_QUERY_ID"))
         fJobId = gSystem->Getenv("PROOF_QUERY_ID");
      else if (gSystem->Getenv("GRID_JOB_ID"))
         fJobId = gSystem->Getenv("GRID_JOB_ID");
      else if (gSystem->Getenv("LCG_JOB_ID"))
         fJobId = gSystem->Getenv("LCG_JOB_ID");
      else if (gSystem->Getenv("ALIEN_MASTERJOBID"))
         fJobId = gSystem->Getenv("ALIEN_MASTERJOBID");
      else if (gSystem->Getenv("ALIEN_PROC_ID"))
         fJobId = gSystem->Getenv("ALIEN_PROC_ID");
      else
         fJobId = "-no-job-id";
   } else {
      fJobId = monid;
   }

   if (monsubid == 0) {
     if (gSystem->Getenv("PROOF_PROC_ID")) {
       fSubJobId = gSystem->Getenv("PROOF_PROC_ID");
     } else if (gSystem->Getenv("ALIEN_PROC_ID")) {
       fSubJobId = gSystem->Getenv("ALIEN_PROC_ID");
     } else {
       fSubJobId = fJobId;
     }
   } else {
     fSubJobId = monsubid;
   }


   if (fVerbose)
      Info("Initialized for ML Server <%s> - Setting ClusterID <%s> JobID <%s> SubID <%s>\n",
           apmon_config[0], fName.Data() ,fJobId.Data(),fSubJobId.Data());

   fInitialized = kTRUE;

   TString optionStr(option);
   if (optionStr.Contains("global"))
      gMonitoringWriter = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Cleanup.

TMonaLisaWriter::~TMonaLisaWriter()
{
   if (fMonInfoRepo) {

      std::map<UInt_t, MonitoredTFileInfo *>::iterator iter = fMonInfoRepo->begin();
      while (iter != fMonInfoRepo->end()) {
         delete iter->second;
         ++iter;
      }

      fMonInfoRepo->clear();
      delete fMonInfoRepo;
      fMonInfoRepo = 0;
   }

   if (gMonitoringWriter == this)
      gMonitoringWriter = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Sends a `<status>` text to MonaLisa following the process scheme:
///    `<site>` --> `<jobid>` --> 'status' = `<status>`
/// Used to set a global status for a groupjob, e.g.
/// a master-job or the general status of PROOF processing.

Bool_t TMonaLisaWriter::SendInfoStatus(const char *status)
{
   if (!fInitialized) {
      Error("SendInfoStatus", "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText("status", status);
   valuelist->Add(valtext);

   // send it to monalisa
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Sends the `<user>` text to MonaLisa following the process scheme:
///    `<site>` --> `<jobid>` --> 'user' = `<user>`

Bool_t TMonaLisaWriter::SendInfoUser(const char *user)
{
   if (!fInitialized) {
      Error("TMonaLisaWriter",
            "Monitoring initialization has failed - you can't send to MonaLisa!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   const char *localuser;
   if (user) {
      localuser = user;
   } else {
      if (gGrid) {
         localuser = gGrid->GetUser();
      } else {
         localuser = "unknown";
      }
   }

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText("user", localuser);
   valuelist->Add(valtext);

   // send it to monalisa
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Sends the description `<jobtag>` following the processing scheme:
///    `<site>` --> `<jobid>` --> 'jobname' = `<jobtag>`

Bool_t TMonaLisaWriter::SendInfoDescription(const char *jobtag)
{
   if (!fInitialized) {
      Error("SendInfoDescription",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText("jobname", jobtag);
   valuelist->Add(valtext);

   // send it to monalisag
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Sends the current time to MonaLisa following the processing scheme
///    `<site>` --> `<jobid>` --> 'time' = >unixtimestamp<

Bool_t TMonaLisaWriter::SendInfoTime()
{
   if (!fInitialized) {
      Error("SendInfoTime", "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   TString valtime = (Int_t) time(0);

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText("time", valtime);
   valuelist->Add(valtext);

   // send it to monalisa
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Send the procesing status 'status' to MonaLisa following the
/// processing scheme:
///    `<site>` --> `<jobid>` --> 'status' = `<status>`
/// Used, to set the processing status of individual subtaks e.g. the
/// status of a batch (sub-)job or the status of a PROOF slave
/// participating in query `<jobid>`

Bool_t TMonaLisaWriter::SendProcessingStatus(const char *status, Bool_t restarttimer)
{
   if (restarttimer) {
      fStopwatch.Start(kTRUE);
   }

   if (!fInitialized) {
      Error("TMonaLisaWriter",
            "Monitoring initialization has failed - you can't send to MonaLisa!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText("status", status);
   valuelist->Add(valtext);

   TMonaLisaText *valhost = new TMonaLisaText("hostname",fHostname);
   valuelist->Add(valhost);

   TMonaLisaText *valsid = new TMonaLisaText("subid", fSubJobId.Data());
   valuelist->Add(valsid);

   // send it to monalisa
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Send the procesing progress to MonaLisa.

Bool_t TMonaLisaWriter::SendProcessingProgress(Double_t nevent, Double_t nbytes, Bool_t force)
{
   if (!force && (time(0)-fLastProgressTime) < fReportInterval) {
     // if the progress is not forced, we send maximum < fReportInterval per second!
     return kFALSE;
   }

   if (!fInitialized) {
      Error("SendProcessingProgress",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaValue *valevent =    new TMonaLisaValue("events", nevent);
   TMonaLisaValue *valbyte =     new TMonaLisaValue("processedbytes", nbytes);
   TMonaLisaValue *valrealtime = new TMonaLisaValue("realtime",fStopwatch.RealTime());
   TMonaLisaValue *valcputime =  new TMonaLisaValue("cputime",fStopwatch.CpuTime());

   ProcInfo_t pinfo;
   gSystem->GetProcInfo(&pinfo);
   Double_t totmem = (Double_t)(pinfo.fMemVirtual) * 1024.;
   Double_t rssmem = (Double_t)(pinfo.fMemResident) * 1024.;
   Double_t shdmem = 0.;

   TMonaLisaValue *valtotmem = new TMonaLisaValue("totmem",totmem);
   TMonaLisaValue *valrssmem = new TMonaLisaValue("rssmem",rssmem);
   TMonaLisaValue *valshdmem = new TMonaLisaValue("shdmem",shdmem);

   TMonaLisaText *valsid = new TMonaLisaText("subid", fSubJobId.Data());
   valuelist->Add(valsid);
   valuelist->Add(valevent);
   valuelist->Add(valbyte);
   valuelist->Add(valrealtime);
   valuelist->Add(valcputime);
   valuelist->Add(valtotmem);
   valuelist->Add(valrssmem);
   valuelist->Add(valshdmem);

   TString      strevents="";
   strevents += nevent;
   TString      strbytes="";
   strbytes  += nbytes;
   TString      strcpu="";
   strcpu    += fStopwatch.CpuTime();
   TString      strreal="";
   strreal   += fStopwatch.RealTime();
   TString      strtotmem="";
   strtotmem += totmem;
   TString      strrssmem="";
   strrssmem += rssmem;
   TString      strshdmem="";
   strshdmem += shdmem;

   fStopwatch.Continue();

   TMonaLisaText *textevent   = new TMonaLisaText("events_str", strevents.Data());
   TMonaLisaText *textbyte    = new TMonaLisaText("processedbytes_str", strbytes.Data());
   TMonaLisaText *textreal    = new TMonaLisaText("realtime_str", strreal.Data());
   TMonaLisaText *textcpu     = new TMonaLisaText("cputime_str", strcpu.Data());
   TMonaLisaText *texttotmem  = new TMonaLisaText("totmem_str", strtotmem.Data());
   TMonaLisaText *textrssmem  = new TMonaLisaText("rssmem_str", strrssmem.Data());
   TMonaLisaText *textshdmem  = new TMonaLisaText("shdmem_str", strshdmem.Data());
   valuelist->Add(textevent);
   valuelist->Add(textbyte);
   valuelist->Add(textcpu);
   valuelist->Add(textreal);
   valuelist->Add(texttotmem);
   valuelist->Add(textrssmem);
   valuelist->Add(textshdmem);

   TMonaLisaText *valhost = new TMonaLisaText("hostname",fHostname);
   valuelist->Add(valhost);

   // send it to monalisa
   success = SendParameters(valuelist);
   fLastProgressTime = time(0);
   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Send the fileopen progress to MonaLisa.
/// If openphases=0 it means that the information is to be stored
/// in a temp space, since there is not yet an object where to attach it to.
/// This is typical in the static Open calls.
/// The temp openphases are put into a list as soon as one is specified.
///
/// If thisopenphasename=0 it means that the stored phases (temp and object)
/// have to be cleared.

Bool_t TMonaLisaWriter::SendFileOpenProgress(TFile *file, TList *openphases,
                                             const char *openphasename,
                                             Bool_t forcesend)
{
   if (!fInitialized) {
      Error("SendFileOpenProgress",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   // Create the list, if not yet done
   if (!fTmpOpenPhases && !openphases) {
      fTmpOpenPhases = new TList;
      fTmpOpenPhases->SetOwner();
   }

   if (!openphasename) {
      // This means "reset my phases"
      fTmpOpenPhases->Clear();
      return kTRUE;
   }

   // Take a measurement
   TParameter<Double_t> *nfo = new TParameter<Double_t>(openphasename, fFileStopwatch.RealTime());
   fFileStopwatch.Continue();

   if (!openphases) {
      fTmpOpenPhases->Add(nfo);
   } else {
      // Move info temporarly saved to object list
      TIter nxt(fTmpOpenPhases);
      TParameter<Double_t> *nf = 0;
      while ((nf = (TParameter<Double_t> *)nxt()))
         openphases->Add(nf);
      // Add this measurement
      openphases->Add(nfo);
      // Reset the temporary list
      if (fTmpOpenPhases) {
         fTmpOpenPhases->SetOwner(0);
         fTmpOpenPhases->Clear();
      }

   }

   if (!forcesend) return kTRUE;
   if (!file) return kTRUE;

   TList *op = openphases ? openphases : fTmpOpenPhases;

   Bool_t success = kFALSE;


   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object

   TMonaLisaText *valhost = new TMonaLisaText("hostname",fHostname);
   valuelist->Add(valhost);
   TMonaLisaText *valsid = new TMonaLisaText("subid", fSubJobId.Data());
   valuelist->Add(valsid);
   TMonaLisaText *valdest = new TMonaLisaText("destname", file->GetEndpointUrl()->GetHost());
   valuelist->Add(valdest);

   TMonaLisaValue *valfid = new TMonaLisaValue("fileid", file->GetFileCounter());
   valuelist->Add(valfid);
   TString strfid = Form("%lld", file->GetFileCounter());
   TMonaLisaText *valstrfid = new TMonaLisaText("fileid_str", strfid.Data());
   valuelist->Add(valstrfid);

   Int_t kk = 1;
   TIter nxt(op);
   TParameter<Double_t> *nf1 = 0;
   TParameter<Double_t> *nf0 = (TParameter<Double_t> *)nxt();
   while ((nf1 = (TParameter<Double_t> *)nxt())) {
      TString s = Form("openphase%d_%s", kk, nf0->GetName());
      TMonaLisaValue *v = new TMonaLisaValue(s.Data(), nf1->GetVal() - nf0->GetVal());
      valuelist->Add(v);
      // Go to next
      nf0 = nf1;
      kk++;
   }

   // Now send how much time was elapsed in total
   nf0 = (TParameter<Double_t> *)op->First();
   nf1 = (TParameter<Double_t> *)op->Last();
   TMonaLisaValue *valtottime =
      new TMonaLisaValue("total_open_time", nf1->GetVal() - nf0->GetVal());
   valuelist->Add(valtottime);

   // send it to monalisa
   success = SendParameters(valuelist);
   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TMonaLisaWriter::SendFileCloseEvent(TFile *file) {
   if (!fInitialized) {
      Error("SendFileCloseEvent",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;
   Double_t timenow = fFileStopwatch.RealTime();
   fFileStopwatch.Continue();

   MonitoredTFileInfo *mi = 0;
   std::map<UInt_t, MonitoredTFileInfo *>::iterator iter = fMonInfoRepo->find(file->GetUniqueID());
   if (iter != fMonInfoRepo->end()) mi = iter->second;

   Double_t timelapsed = 0.0;
   if (mi) timelapsed = timenow - mi->fCreationTime;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   TString valname;
   TString pfx = file->ClassName();
   if (file->InheritsFrom("TXNetFile"))
      pfx = "TXNetFile";

   pfx += "_";

   // The info to be sent is the one relative to the specific file

   TMonaLisaText *valdest = new TMonaLisaText("destname",file->GetEndpointUrl()->GetHost());
   valuelist->Add(valdest);
   TMonaLisaValue *valfid = new TMonaLisaValue("fileid",file->GetFileCounter());
   valuelist->Add(valfid);
   TString strfid="";
   strfid+=file->GetFileCounter();
   TMonaLisaText *valstrfid = new TMonaLisaText("fileid_str",strfid.Data());
   valuelist->Add(valstrfid);

   valname = pfx;
   valname += "readbytes";
   TMonaLisaValue *valread = new TMonaLisaValue(valname, file->GetBytesRead());
   valuelist->Add(valread);

//    TString strbytes_r="";
//    strbytes_r += file->GetBytesRead();
//    TMonaLisaText *valstrread = new TMonaLisaText("readbytes_str", strbytes_r.Data());
//    valuelist->Add(valstrread);

   valname = pfx;
   valname += "writtenbytes";
   TMonaLisaValue *valwrite = new TMonaLisaValue(valname, file->GetBytesWritten());
   valuelist->Add(valwrite);

//    TString strbytes_w="";
//    strbytes_w += file->GetBytesWritten();
//    TMonaLisaText *valstrwrite = new TMonaLisaText("writtenbytes_str", strbytes_w.Data());
//    valuelist->Add(valstrwrite);

   int thput;
   if (timelapsed > 0.001) {
      Int_t selapsed = std::round(std::floor(timelapsed * 1000));

      thput = file->GetBytesRead() / selapsed * 1000;
      valname = pfx;
      valname += "filethrpt_rd";
      TMonaLisaValue *valreadthavg = new TMonaLisaValue(valname, thput);
      valuelist->Add(valreadthavg);

      thput = file->GetBytesWritten() / selapsed * 1000;
      valname = pfx;
      valname += "filethrpt_wr";
      TMonaLisaValue *valwritethavg = new TMonaLisaValue(valname, thput);
      valuelist->Add(valwritethavg);
   }

   // And the specific file summary has to be removed from the repo
   if (mi) {
      mi->UpdateFileStatus(file);
      mi->fKillme = kTRUE;
   }

   // send it to monalisa
   success = SendParameters(valuelist);


   delete valuelist;
   return success;
}
////////////////////////////////////////////////////////////////////////////////

Bool_t TMonaLisaWriter::SendFileReadProgress(TFile *file) {
   return SendFileCheckpoint(file);
}
////////////////////////////////////////////////////////////////////////////////

Bool_t TMonaLisaWriter::SendFileWriteProgress(TFile *file) {
   return SendFileCheckpoint(file);
}
////////////////////////////////////////////////////////////////////////////////

Bool_t TMonaLisaWriter::SendFileCheckpoint(TFile *file)
{
   if (!fInitialized) {
      Error("SendFileCheckpoint",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   if (!file->IsOpen()) return kTRUE;

   // We cannot handle this kind of ongoing averaged monitoring for a file which has not an unique id. Sorry.
   // This seems to affect raw files, for which only the Close() info is available
   // Removing this check causes a mess for non-raw and raw TFiles, because
   // the UUID in the Init phase has not yet been set, and the traffic
   // reported during that phase is reported wrongly, causing various leaks and troubles
   // TFiles without an unique id can be monitored only in their Open/Close event
   if (!file->TestBit(kHasUUID)) return kTRUE;

   Double_t timenow = fFileStopwatch.RealTime();
   fFileStopwatch.Continue();

   // This info has to be gathered in any case.

   // Check if an MonitoredTFileInfo instance is already available
   // If not, create one
   MonitoredTFileInfo *mi = 0;
   std::map<UInt_t, MonitoredTFileInfo *>::iterator iter = fMonInfoRepo->find(file->GetUniqueID());
   if (iter != fMonInfoRepo->end()) mi = iter->second;
   if (!mi) {

      mi = new MonitoredTFileInfo(file, timenow);
      if (mi) fMonInfoRepo->insert( make_pair( file->GetUniqueID(), mi ) );
   }

   // And now we get those partial values
   if (mi) mi->UpdateFileStatus(file);

   // Send the fileread progress to MonaLisa only if required or convenient
   if ( timenow - fLastRWSendTime < fReportInterval) {
      // if the progress is not forced, we send maximum 1 every fReportInterval seconds!
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   TString valname;

   // We send only a little throughput summary info
   // Instead we send the info for the actual file passed

   TMonaLisaText *valhost = new TMonaLisaText("hostname",fHostname);
   valuelist->Add(valhost);
   TMonaLisaText *valsid = new TMonaLisaText("subid", fSubJobId.Data());
   valuelist->Add(valsid);

   // First of all, we create an internal summary, sorted by kind of TFile used
   THashList summary;
   summary.SetOwner(kTRUE);

   iter = fMonInfoRepo->begin();
   if (iter != fMonInfoRepo->end()) mi = iter->second;
   else mi = 0;

   while (mi) {
      MonitoredTFileSummary *sum = static_cast<MonitoredTFileSummary *>(summary.FindObject(mi->fFileClassName));
      if (!sum) {
         sum = new MonitoredTFileSummary(mi->fFileClassName);
         if (sum) summary.AddLast(sum);
      }

      if (sum) {
         sum->Update(mi, timenow, fLastRWSendTime);
         mi->ResetFileStatus(timenow);
      }

      // This could be an info about an already closed file
      if (mi->fKillme) {
         iter->second = 0;
      }

      ++iter;
      if (iter != fMonInfoRepo->end())
         mi = iter->second;
      else
         mi = 0;
   }

   for (iter = fMonInfoRepo->begin(); iter != fMonInfoRepo->end(); ) 
   {
      if (!iter->second)
         iter = fMonInfoRepo->erase(iter);
      else
         ++iter;
   }

   // This info is a summary valid for all the monitored files at once.
   // It makes no sense at all to send data relative to a specific file here
   // Cycle through the summary...
   TIter nxt2(&summary);
   MonitoredTFileSummary *sum;
   while ((sum = (MonitoredTFileSummary *)nxt2())) {

      if (sum->fReadThroughput >= 0) {
         valname = sum->GetName();
         valname += "_avgthrpt_rd";
         TMonaLisaValue *valreadthr = new TMonaLisaValue(valname, sum->fReadThroughput);
         valuelist->Add(valreadthr);
      }

      if ( sum->fWriteThroughput >= 0 ) {
         valname = sum->GetName();
         valname += "_avgthrpt_wr";
         TMonaLisaValue *valwritethr = new TMonaLisaValue(valname, sum->fWriteThroughput);
         valuelist->Add(valwritethr);
      }

   }


   // send it to monalisa
   success = SendParameters(valuelist);

   fLastRWSendTime = timenow;

   delete valuelist;
   return success;
}

////////////////////////////////////////////////////////////////////////////////
/// Send the parameters to MonaLisa.

Bool_t TMonaLisaWriter::SendParameters(TList *valuelist, const char *identifier)
{
   if (!fInitialized) {
      Error("SendParameters", "Monitoring is not properly initialized!");
      return kFALSE;
   }

   if (!valuelist) {
      Error("SendParameters", "No values in the value list!");
      return kFALSE;
   }

   if (identifier == 0)
      identifier = fJobId;

   TIter nextvalue(valuelist);

   TMonaLisaValue *objval;
   TMonaLisaText *objtext;
   TObject *monobj;

   Int_t apmon_nparams = valuelist->GetSize();
   char **apmon_params = 0;
   Int_t *apmon_types = 0;
   char **apmon_values = 0;
   Double_t *bufDouble = 0; // buffer for int, long, etc. that is to be sent as double

   if (apmon_nparams) {

      apmon_params = (char **) malloc(apmon_nparams * sizeof(char *));
      apmon_values = (char **) malloc(apmon_nparams * sizeof(char *));
      apmon_types = (int *) malloc(apmon_nparams * sizeof(int));
      bufDouble = new Double_t[apmon_nparams];

      Int_t looper = 0;
      while ((monobj = nextvalue())) {
         if (!strcmp(monobj->ClassName(), "TMonaLisaValue")) {
            objval = (TMonaLisaValue *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %f",
                    objval->GetName(), objval->GetValue());

            apmon_params[looper] = (char *) objval->GetName();
            apmon_types[looper] = XDR_REAL64;
            apmon_values[looper] = (char *) (objval->GetValuePtr());
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TMonaLisaText")) {
            objtext = (TMonaLisaText *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with text %s",
                    objtext->GetName(), objtext->GetText());

            apmon_params[looper] = (char *) objtext->GetName();
            apmon_types[looper] = XDR_STRING;
            apmon_values[looper] = (char *) (objtext->GetText());
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TNamed")) {
            TNamed* objNamed = (TNamed *) monobj;

            if (fVerbose)
              Info("SendParameters", "adding tag %s with text %s",
                   objNamed->GetName(), objNamed->GetTitle());

            apmon_params[looper] = (char *) objNamed->GetName();
            apmon_types[looper] = XDR_STRING;
            apmon_values[looper] = (char *) (objNamed->GetTitle());
            looper++;
         }
         // unfortunately ClassName() converts Double_t to double, etc.
         if (!strcmp(monobj->ClassName(), "TParameter<double>")) {
            TParameter<double>* objParam = (TParameter<double> *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %f",
                    objParam->GetName(), objParam->GetVal());

            apmon_params[looper] = (char *) objParam->GetName();
            apmon_types[looper] = XDR_REAL64;
            apmon_values[looper] = (char *) &(objParam->GetVal());
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TParameter<Long64_t>")) {
            TParameter<Long64_t>* objParam = (TParameter<Long64_t> *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %lld",
                    objParam->GetName(), objParam->GetVal());

            apmon_params[looper] = (char *) objParam->GetName();
            apmon_types[looper] = XDR_REAL64;
            bufDouble[looper] = objParam->GetVal();
            apmon_values[looper] = (char *) (bufDouble + looper);
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TParameter<long>")) {
            TParameter<long>* objParam = (TParameter<long> *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %ld",
                    objParam->GetName(), objParam->GetVal());

            apmon_params[looper] = (char *) objParam->GetName();
            apmon_types[looper] = XDR_REAL64;
            bufDouble[looper] = objParam->GetVal();
            apmon_values[looper] = (char *) (bufDouble + looper);
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TParameter<float>")) {
            TParameter<float>* objParam = (TParameter<float> *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %f",
                    objParam->GetName(), objParam->GetVal());

            apmon_params[looper] = (char *) objParam->GetName();
            apmon_types[looper] = XDR_REAL64;
            bufDouble[looper] = objParam->GetVal();
            apmon_values[looper] = (char *) (bufDouble + looper);
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TParameter<int>")) {
            TParameter<int>* objParam = (TParameter<int> *) monobj;

            if (fVerbose)
               Info("SendParameters", "adding tag %s with val %d",
                    objParam->GetName(), objParam->GetVal());

            apmon_params[looper] = (char *) objParam->GetName();
            apmon_types[looper] = XDR_REAL64;
            bufDouble[looper] = objParam->GetVal();
            apmon_values[looper] = (char *) (bufDouble + looper);
            looper++;
         }
      }

      // change number of parameters to the actual found value
      apmon_nparams = looper;

      if (fVerbose)
         Info("SendParameters", "n: %d name: %s identifier %s ...,",
              apmon_nparams, GetName(), identifier);

      ((ApMon *) fApmon)->sendParameters((char *) GetName(), (char*)identifier,
                                         apmon_nparams, apmon_params,
                                         apmon_types, apmon_values);

      free(apmon_params);
      free(apmon_values);
      free(apmon_types);
      delete[] bufDouble;
   }
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Set MonaLisa log level.

void TMonaLisaWriter::SetLogLevel(const char *loglevel)
{
   ((ApMon *) fApmon)->setLogLevel((char *) loglevel);
}

////////////////////////////////////////////////////////////////////////////////
/// Print info about MonaLisa object.

void TMonaLisaWriter::Print(Option_t *) const
{
   std::cout << "Site     (Farm) : " << fName << std::endl;
   std::cout << "JobId    (Node) : " << fJobId << std::endl;
   std::cout << "SubJobId (Node) : " << fSubJobId << std::endl;
   std::cout << "HostName        : " << fHostname << std::endl;
   std::cout << "Pid             : " << fPid << std::endl;
   std::cout << "Inititialized   : " << fInitialized << std::endl;
   std::cout << "Verbose         : " << fVerbose << std::endl;

}
