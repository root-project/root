// @(#)root/monalisa:$Name:  $:$Id: TMonaLisa.cxx,v 1.1 2005/12/11 02:39:28 rdm Exp $
// Author: Andreas Peters   5/10/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMonaLisa                                                            //
//                                                                      //
// Class defining interface to MonaLisa Monitoring Services in ROOT.    //
// The TMonaLisa object is used to send monitoring information to a     //
// MonaLisa server using the MonaLisa ApMon package (libapmoncpp.so/UDP //
// packets). The MonaLisa ApMon library for C++ can be downloaded at    //
// http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,     //
// current version:                                                     //
// http://monalisa.cacr.caltech.edu/download/apmon/ApMon_c-2.0.6.tar.gz //
//                                                                      //
// The ROOT implementation is primary optimized for process/job         //
// monitoring, although all other generic MonaLisa ApMon functionality  //
// can be exploited through the ApMon class directly                    //
// (gMonaLisa->GetApMon()).                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMonaLisa.h"
#include "TSystem.h"
#include "TGrid.h"
#include "Riostream.h"

TMonaLisa *gMonaLisa = 0;


ClassImp(TMonaLisa)

//______________________________________________________________________________
TMonaLisa::TMonaLisa(const char *monid, const char *montag,
                     const char *monserver)
{
   // Creates a TMonaLisa object to send monitoring information to a
   // MonaLisa server using the MonaLisa ApMon package (libapmoncpp.so/UDP
   // packets). The MonaLisa ApMon library for C++ can be downloaded at
   // http://monalisa.cacr.caltech.edu/monalisa__Download__ApMon.html,
   // current version:
   // http://monalisa.cacr.caltech.edu/download/apmon/ApMon_cpp-2.0.6.tar.gz
   //
   // The ROOT implementation is primary optimized for process/job monitoring,
   // although all other generic MonaLisa ApMon functionality can be exploited
   // through the ApMon class directly (gMonaLisa->GetApMon()).
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
   // If monid is not specified, TMonaLisa tries to set it in this order
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
   // To use TMonaLisa, libMonaLisa.so has to be loaded.
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
   //   new TMonaLisa("BATCH_ANALYSIS","AnalysisLoop-00001","lxplus050.cern.ch");
   // Once when you create an analysis task, execute
   //   gMonaLisa->SendInfoUser("myname");
   //   gMonaLisa->SendInfoDescription("My first Higgs analysis");
   //   gMonaLisa->SendInfoTime();
   //   gMonaLisa->SendInfoStatus("Submitted");
   //
   // On each node executing a subtask, you can set the status of this subtask:
   //   gMonaLisa->SendProcessingStatus("Started");
   // During the processing of your analysis you can send progress updates:
   //   gMonaLisa->SendProcessProgress(100,1000000); <= 100 events, 1MB processed
   //   ....
   //   gMonaLisa-SendProcessingStatus("Finished");
   //   delete gMonaLisa; gMonaLisa=0;
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

   char *apmon_config[1] =
      { ((monserver == 0) ? getenv("APMON_CONFIG") : (char *) monserver) };
   if (apmon_config[0] == 0) {
      Error("TMonaLisa",
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
      Error("TMonaLisa", "Error initializing ApMon: %s", e.what());
      Error("TMonaLisa", "Disabling apmon.");
      fInitialized = kFALSE;
      return;
   }

   if (monid == 0) {
      if (getenv("PROOF_JOB_ID"))
         fJobId = getenv("PROOF_JOB_ID");
      else if (getenv("GRID_JOB_ID"))
         fJobId = getenv("GRID_JOB_ID");
      else if (getenv("LCG_JOB_ID"))
         fJobId = getenv("LCG_JOB_ID");
      else if (getenv("ALIEN_MASTERJOBID"))
         fJobId = getenv("ALIEN_MASTERJOBID");
      else if (getenv("ALIEN_PROC_ID"))
         fJobId = getenv("ALIEN_PROC_ID");
      else
         fJobId = "-no-job-id";
   } else {
      fJobId = (char *) monid;
   }

   fHostname = gSystem->HostName();
   fPid = gSystem->GetPid();

   if (fVerbose)
      Info("Initialized for ML Server <%s> - setting JobID <%s>\n",
           apmon_config[0], fJobId);

   fInitialized = kTRUE;

   gMonaLisa = this;
}

//______________________________________________________________________________
TMonaLisa::~TMonaLisa()
{
   // Cleanup.

}

//______________________________________________________________________________
Bool_t TMonaLisa::SendInfoStatus(const char *status)
{
   // Sends a <status> text to MonaLisa following the process scheme:
   //    <monid> --> <jobid> --> 'status' = <status>
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
Bool_t TMonaLisa::SendInfoUser(const char *user)
{
   // Sends the <user> text to MonaLisa following the process scheme:
   //    <monid> --> <jobid> --> 'user' = <user>

   if (!fInitialized) {
      Error("TMonaLisa",
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
Bool_t TMonaLisa::SendInfoDescription(const char *jobtag)
{
   // Sends the description <jobtag> following the processing scheme:
   //    <monid> --> <jobid> --> 'jobname' = <jobtag>

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
Bool_t TMonaLisa::SendInfoTime()
{
   // Sends the current time to MonaLisa following the processing scheme
   //    <monid> --> <jobid> --> 'time' = >unixtimestamp<

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
Bool_t TMonaLisa::SendProcessingStatus(const char *status)
{
   // Send the procesing status 'status' to MonaLisa following the
   // processing scheme:
   //    <monid> --> <jobid> --> '<site>:<host>:<pid>:status' = <status>
   // Used, to set the processing status of individual subtaks e.g. the
   // status of a batch (sub-)job or the status of a PROOF slave
   // participating in query <jobid>

   if (!fInitialized) {
      Error("TMonaLisa",
            "Monitoring initialization has failed - you can't send to MonaLisa!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TString mltag;
   // set the site
   if (getenv("PROOF_SITE"))
      mltag = getenv("PROOF_SITE");
   else if (getenv("GRID_SITE"))
      mltag = getenv("GRID_SITE");
   else if (getenv("ALIEN_SITE"))
      mltag = getenv("ALIEN_SITE");
   else
      mltag = "none:";

   mltag += ":";

   // set the host + pid
   mltag += fHostname;
   mltag += ":";
   mltag += fPid;
   mltag += ":status";

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaText *valtext = new TMonaLisaText(mltag.Data(), status);
   valuelist->Add(valtext);

   // send it to monalisa
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

//______________________________________________________________________________
Bool_t TMonaLisa::SendProcessingProgress(Double_t nevent, Double_t nbytes)
{
   // Send the procesing progress to MonaLisa.

   if (!fInitialized) {
      Error("SendProcessingProgress",
            "Monitoring is not properly initialized!");
      return kFALSE;
   }

   Bool_t success = kFALSE;

   TString mltag;
   // set the site
   if (getenv("PROOF_SITE"))
      mltag = getenv("PROOF_SITE");
   else if (getenv("GRID_SITE"))
      mltag = getenv("GRID_SITE");
   else if (getenv("ALIEN_SITE"))
      mltag = getenv("ALIEN_SITE");
   else
      mltag = "none:";

   mltag += ":";

   // set the host + pid
   mltag += fHostname;
   mltag += ":";
   mltag += fPid;
   mltag += ":";
   TString eventtag;
   TString bytetag;
   eventtag = mltag;
   bytetag = mltag;
   eventtag += "entries";
   bytetag += "datasize";

   TList *valuelist = new TList();
   valuelist->SetOwner(kTRUE);

   // create a monitor text object
   TMonaLisaValue *valevent = new TMonaLisaValue(eventtag.Data(), nevent);
   TMonaLisaValue *valbyte = new TMonaLisaValue(bytetag.Data(), nbytes);
   valuelist->Add(valevent);
   valuelist->Add(valbyte);

   // send it to monalisaTMonaLisaText* valtext = new TMonaLisaText
   success = SendParameters(valuelist);

   delete valuelist;
   return success;
}

//______________________________________________________________________________
Bool_t TMonaLisa::SendParameters(TList *valuelist)
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
              apmon_nparams, GetName(), fJobId);

      ((ApMon *) fApmon)->sendParameters((char *) GetName(), fJobId,
                                         apmon_nparams, apmon_params,
                                         apmon_types, apmon_values);
      free(apmon_params);
      free(apmon_values);
      free(apmon_types);
   }
   return kTRUE;
}

//______________________________________________________________________________
void TMonaLisa::SetLogLevel(const char *loglevel)
{
   // Set MonaLisa log level.

   ((ApMon *) fApmon)->setLogLevel((char *) loglevel);
}

//______________________________________________________________________________
void TMonaLisa::Print(Option_t *) const
{
   // Print info about MonaLisa object.

   cout << "MonGroup (Farm) : " << fName << endl;
   cout << "MonTag   (Node) : " << fJobId << endl;
   cout << "HostName        : " << fHostname << endl;
   cout << "Pid             : " << fPid << endl;
   cout << "Inititialized   : " << fInitialized << endl;
   cout << "Verbose         : " << fVerbose << endl;

}
