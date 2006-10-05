// @(#)root/monalisa:$Name:  $:$Id: TMonaLisaWriter.cxx,v 1.1 2006/06/26 16:17:05 rdm Exp $
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
//////////////////////////////////////////////////////////////////////////

#include "TMonaLisaWriter.h"
#include "TSystem.h"
#include "TGrid.h"
#include "TFile.h"
#include "TUrl.h"
#include "TStopwatch.h"
#include "Riostream.h"


ClassImp(TMonaLisaWriter)

//______________________________________________________________________________
TMonaLisaWriter::TMonaLisaWriter(const char *monid, const char *monsubid,
                                 const char *montag, const char *monserver)
{
   // Create MonaLisa write object.

   Init(monid,monsubid,montag,monserver);
}

//______________________________________________________________________________
void TMonaLisaWriter::Init(const char *monid, const char* monsubid,
                           const char *montag, const char *monserver)
{
   // Creates a TMonaLisaWriter object to send monitoring information to a
   // MonaLisa server using the MonaLisa ApMon package (libapmoncpp.so/UDP
   // packets). The MonaLisa ApMon library for C++ can be downloaded at
   // http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,
   // current version:
   // http://monalisa.cacr.caltech.edu/download/apmon/ApMon_cpp-2.0.6.tar.gz
   //
   // The ROOT implementation is primary optimized for process/job monitoring,
   // although all other generic MonaLisa ApMon functionality can be exploited
   // through the ApMon class directly (gMonitoringWriter->GetApMon()).
   //
   // Monitoring information in MonaLisa is structured in the following tree
   // structure:
   // <farmname>
   //    |
   //    ---> <nodename1>
   //              |
   //              ---> <key1> - <value1>
   //              ---> <key2> - <value2>
   //    ---> <nodename2>
   //              |
   //              ---> <key3> - <value3>
   //              ---> <key4> - <value4>
   //
   // The parameter monid is equivalent to the MonaLisa node name, for the
   // case of process monitoring it can be just an identifier to classify
   // the type of jobs e.g. "PROOF_PROCESSING".
   // If monid is not specified, TMonaLisaWriter tries to set it in this order
   // from environement variables:
   // - PROOF_JOB_ID
   // - GRID_JOB_ID
   // - LCG_JOB_ID
   // - ALIEN_MASTERJOB_ID
   // - ALIEN_PROC_ID
   //
   // The parameter montag is equivalent to the MonaLisa farm name, for the
   // case of process monitoring it can be a process identifier e.g. a PROOF
   // session ID.
   //
   // The parameter monserver specifies the server to whom to send the
   // monitoring UDP packets. If not specified, the hostname (the port is
   // a default one) is specified in the environment variable APMON_CONFIG.
   //
   // To use TMonaLisaWriter, libMonaLisa.so has to be loaded.
   //
   // According to the fact, that the deepness of the MonaLisa naming scheme
   // is only 3 (<farm><node><value>), a special naming scheme is used for
   // process monitoring. There is a high-level method to send progress
   // information of Tree analysis (# of events, datasize).
   // To distinguish individual nodes running the processing, part of the
   // information is kept in the <value> parameter of ML.
   // <value> is named as:
   //    <site-name>:<host-name>:<pid>:<valuetag>
   // <site-name> is taken from an environment variable in the following order:
   // - PROOF_SITE
   // - GRID_SITE
   // - ALIEN_SITE
   // - default 'none'
   // <host-name> is taken from gSystem->Hostname()
   // <pid> is the process ID of the ROOT process
   //
   // Example of use for Process Monitoring:
   //   new TMonaLisaWriter("BATCH_ANALYSIS","AnalysisLoop-00001","lxplus050.cern.ch");
   // Once when you create an analysis task, execute
   //   gMonitoringWriter->SendInfoUser("myname");
   //   gMonitoringWriter->SendInfoDescription("My first Higgs analysis");
   //   gMonitoringWriter->SendInfoTime();
   //   gMonitoringWriter->SendInfoStatus("Submitted");
   //
   // On each node executing a subtask, you can set the status of this subtask:
   //   gMonitoringWriter->SendProcessingStatus("Started");
   // During the processing of your analysis you can send progress updates:
   //   gMonitoringWriter->SendProcessProgress(100,1000000); <= 100 events, 1MB processed
   //   ....
   //   gMonitoringWriter-SendProcessingStatus("Finished");
   //   delete gMonitoringWriter; gMonitoringWriter=0;
   //
   // Example of use for any Generic Monitoring information:
   //   TList *valuelist = new TList();
   //   valuelist->SetOwner(kTRUE);
   //   // append a text object
   //   TMonaLisaText *valtext = new TMonaLisaText("decaychannel","K->eeg");
   //   valuelist->Add(valtext);
   //   // append a double value
   //   TMonaLisaValue* valdouble = new TMonaLisaValue("n-gamma",5);
   //   valuelist->Add(valdouble);
   //   Bool_t success = SendParameters(valuelist);
   //   delete valuelist;

   SetName(montag);
   SetTitle(montag);

   fVerbose = kFALSE;           // no verbosity as default

   fLastSendTime = time(0);
   fLastProgressTime = time(0);
   fReportInterval = 5; // default interval is 5
   if (gSystem->Getenv("APMON_INTERVAL")) {
      fReportInterval = atoi(gSystem->Getenv("APMON_INTERVAL"));
      if (fReportInterval < 1)
         fReportInterval =1;
      Info("TMonaLisaWriter","Setting APMON Report Interval to %d seconds",fReportInterval);
   }

   char *apmon_config[1] =
      { ((monserver == 0) ? gSystem->Getenv("APMON_CONFIG") : (char *) monserver) };
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

   gMonitoringWriter = this;
}

//______________________________________________________________________________
TMonaLisaWriter::TMonaLisaWriter(const char *monid, const char *montag,
				                     const char *monserver)
{
   // Create a MonaLisa writer object.

   Init(monid,monid,montag,monserver);
}

//______________________________________________________________________________
TMonaLisaWriter::~TMonaLisaWriter()
{
   // Cleanup.

   if (gMonitoringWriter == this)
      gMonitoringWriter = 0;
}

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendInfoStatus(const char *status)
{
   // Sends a <status> text to MonaLisa following the process scheme:
   //    <site> --> <jobid> --> 'status' = <status>
   // Used to set a global status for a groupjob, e.g.
   // a master-job or the general status of PROOF processing.

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendInfoUser(const char *user)
{
   // Sends the <user> text to MonaLisa following the process scheme:
   //    <site> --> <jobid> --> 'user' = <user>

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendInfoDescription(const char *jobtag)
{
   // Sends the description <jobtag> following the processing scheme:
   //    <site> --> <jobid> --> 'jobname' = <jobtag>

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendInfoTime()
{
   // Sends the current time to MonaLisa following the processing scheme
   //    <site> --> <jobid> --> 'time' = >unixtimestamp<

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendProcessingStatus(const char *status, Bool_t restarttimer)
{
   // Send the procesing status 'status' to MonaLisa following the
   // processing scheme:
   //    <site> --> <jobid> --> 'status' = <status>
   // Used, to set the processing status of individual subtaks e.g. the
   // status of a batch (sub-)job or the status of a PROOF slave
   // participating in query <jobid>

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendProcessingProgress(Double_t nevent, Double_t nbytes, Bool_t force)
{
   // Send the procesing progress to MonaLisa.

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

   Double_t totmem = (Double_t)TStopwatch::GetTotalMemory();
   Double_t rssmem = (Double_t)TStopwatch::GetResidentMemory();
   Double_t shdmem = (Double_t)TStopwatch::GetSharedMemory();

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

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendFileReadProgress(TFile* file, Bool_t force)
{
   // Send the fileread progress to MonaLisa.

   if (!force && (time(0)-fLastSendTime) < fReportInterval) {
      // if the progress is not forced, we send maximum < 1 per second!
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

   Long64_t nbytes=file->GetBytesRead();
   // create a monitor text object
   TMonaLisaValue *valread = new TMonaLisaValue("readbytes", nbytes);
   valuelist->Add(valread);
   TString strbytes="";
   strbytes+=nbytes;
   TMonaLisaText *valstrread = new TMonaLisaText("readbytes_str", strbytes.Data());
   valuelist->Add(valstrread);
   TMonaLisaText *valhost = new TMonaLisaText("hostname",fHostname);
   valuelist->Add(valhost);
   TMonaLisaText *valsid = new TMonaLisaText("subid", fSubJobId.Data());
   valuelist->Add(valsid);
   TMonaLisaText *valdest = new TMonaLisaText("destname",file->GetEndpointUrl()->GetHost());
   valuelist->Add(valdest);
   TMonaLisaValue *valfid = new TMonaLisaValue("fileid",file->GetFileCounter());
   valuelist->Add(valfid);
   TString strfid="";
   strfid+=file->GetFileCounter();
   TMonaLisaText *valstrfid = new TMonaLisaText("fileid_str",strfid.Data());
   valuelist->Add(valstrfid);

   // send it to monalisa
   success = SendParameters(valuelist);
   fLastSendTime = time(0);
   delete valuelist;
   return success;
}

//______________________________________________________________________________
Bool_t TMonaLisaWriter::SendParameters(TList *valuelist)
{
   // Send the parameters to MonaLisa.

   if (!fInitialized) {
      Error("SendParameters", "Monitoring is not properly initialized!");
      return kFALSE;
   }

   if (!valuelist) {
      Error("SendParameters", "No values in the value list!");
      return kFALSE;
   }

   TIter nextvalue(valuelist);

   TMonaLisaValue *objval;
   TMonaLisaText *objtext;
   TObject *monobj;

   Int_t apmon_nparams = valuelist->GetSize();
   char **apmon_params;
   Int_t *apmon_types;
   char **apmon_values;

   if (apmon_nparams) {

      apmon_params = (char **) malloc(apmon_nparams * sizeof(char *));
      apmon_values = (char **) malloc(apmon_nparams * sizeof(char *));
      apmon_types = (int *) malloc(apmon_nparams * sizeof(int));

      Int_t looper = 0;
      while ((monobj = nextvalue())) {
         if (!strcmp(monobj->ClassName(), "TMonaLisaValue")) {
            objval = (TMonaLisaValue *) monobj;

            if (fVerbose)
               Info("SendParameters", "Adding Tag %s with val %f\n",
                    objval->GetName(), objval->GetValue());

            apmon_params[looper] = (char *) objval->GetName();
            apmon_types[looper] = XDR_REAL64;
            apmon_values[looper] = (char *) (objval->GetValuePtr());
            looper++;
         }
         if (!strcmp(monobj->ClassName(), "TMonaLisaText")) {
            objtext = (TMonaLisaText *) monobj;

            if (fVerbose)
               Info("SendParameters", "Adding Tag %s with Text %s\n",
                    objtext->GetName(), objtext->GetText());

            apmon_params[looper] = (char *) objtext->GetName();
            apmon_types[looper] = XDR_STRING;
            apmon_values[looper] = (char *) (objtext->GetText());
            looper++;
         }
      }

      if (fVerbose)
         Info("SendParameters", "n: %d name: %s identifier %s ...,",
              apmon_nparams, GetName(), fJobId.Data());

      ((ApMon *) fApmon)->sendParameters((char *) GetName(), (char*)fJobId.Data(),
                                         apmon_nparams, apmon_params,
                                         apmon_types, apmon_values);

      free(apmon_params);
      free(apmon_values);
      free(apmon_types);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TMonaLisaWriter::SetLogLevel(const char *loglevel)
{
   // Set MonaLisa log level.

   ((ApMon *) fApmon)->setLogLevel((char *) loglevel);
}

//______________________________________________________________________________
void TMonaLisaWriter::Print(Option_t *) const
{
   // Print info about MonaLisa object.

   cout << "Site     (Farm) : " << fName << endl;
   cout << "JobId    (Node) : " << fJobId << endl;
   cout << "SubJobId (Node) : " << fSubJobId << endl;
   cout << "HostName        : " << fHostname << endl;
   cout << "Pid             : " << fPid << endl;
   cout << "Inititialized   : " << fInitialized << endl;
   cout << "Verbose         : " << fVerbose << endl;

}
