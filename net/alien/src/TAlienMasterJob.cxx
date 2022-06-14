// @(#)root/alien:$Id$
// Author: Jan Fiete Grosse-Oetringhaus   27/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienMasterJob                                                      //
//                                                                      //
// Special Grid job which contains a master job which controls          //
// underlying jobs resulting from job splitting.                        //
//                                                                      //
// Related classes are TAlienJobStatus.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienJobStatus.h"
#include "TAlienMasterJob.h"
#include "TAlienMasterJobStatus.h"
#include "TAlienJob.h"
#include "TObjString.h"
#include "gapi_job_operations.h"
#include "Riostream.h"
#include "TGridResult.h"
#include "TAlien.h"
#include "TFileMerger.h"
#include "TBrowser.h"

ClassImp(TAlienMasterJob);

////////////////////////////////////////////////////////////////////////////////
/// Browser interface.

void TAlienMasterJob::Browse(TBrowser* b)
{
   if (b) {
      b->Add(GetJobStatus());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Gets the status of the master job and all its sub jobs.
/// Returns a TAlienMasterJobStatus object, 0 on failure.

TGridJobStatus *TAlienMasterJob::GetJobStatus() const
{
   TString jobID;
   jobID = fJobID;

   GAPI_JOBARRAY* gjobarray = gapi_queryjobs("-", "%", "-", "-", jobID.Data(),
                                             "-", "-", "-", "-");

   if (!gjobarray)
      return 0;

   if (gjobarray->size() == 0) {
      delete gjobarray;
      return 0;
   }

   TAlienMasterJobStatus *status = new TAlienMasterJobStatus(fJobID);

   TAlienJob masterJob(fJobID);
   status->fMasterJob = dynamic_cast<TAlienJobStatus*>(masterJob.GetJobStatus());

   std::vector<GAPI_JOB>::const_iterator jobIter = gjobarray->begin();
   for (; jobIter != gjobarray->end(); ++jobIter) {

      GAPI_JOB gjob = *jobIter;
      TAlienJobStatus* jobStatus = new TAlienJobStatus();
      TObjString* jID = 0;

      std::map<std::string, std::string>::const_iterator iter = gjob.gapi_jobmap.begin();
      for (; iter != gjob.gapi_jobmap.end(); ++iter) {
         jobStatus->fStatus.Add(new TObjString(iter->first.c_str()), new TObjString(iter->second.c_str()));
         if (strcmp(iter->first.c_str(), "queueId") == 0)
            jID = new TObjString(iter->second.c_str());
      }

      if (jID != 0)
         status->fJobs.Add(jID, jobStatus);
      else
         delete jobStatus;
   }

   return status;
}

////////////////////////////////////////////////////////////////////////////////

void TAlienMasterJob::Print(Option_t* options) const
{
   std::cout << " ------------------------------------------------ " << std::endl;
   std::cout << " Master Job ID                   : " << fJobID << std::endl;
   std::cout << " ------------------------------------------------ " << std::endl;
   TAlienMasterJobStatus* status = (TAlienMasterJobStatus*)(GetJobStatus());
   if (!status) {
      Error("Print","Cannot get the information for this masterjob");
      return;
   }

   std::cout << " N of Subjobs                    : " << status->GetNSubJobs() << std::endl;
   std::cout << " % finished                      : " << status->PercentFinished()*100 << std::endl;
   std::cout << " ------------------------------------------------ " << std::endl;
   TIterator* iter = status->GetJobs()->MakeIterator();

   TObjString* obj = 0;
   while ((obj = (TObjString*)iter->Next()) != 0) {
      TAlienJobStatus* substatus = (TAlienJobStatus*)status->GetJobs()->GetValue(obj->GetName());
      printf(" SubJob: [%-7s] %-10s %20s@%s  RunTime: %s\n",substatus->GetKey("queueId"),substatus->GetKey("status"),substatus->GetKey("node"),substatus->GetKey("site"),substatus->GetKey("runtime"));
   }
   std::cout << " ------------------------------------------------ " << std::endl;
   iter->Reset();
   if ( strchr(options,'l') ) {
      while ((obj = (TObjString*)iter->Next()) != 0) {
         TAlienJobStatus* substatus = (TAlienJobStatus*)status->GetJobs()->GetValue(obj->GetName());
         // list sandboxes
         const char* outputdir = substatus->GetJdlKey("OutputDir");

         TString sandbox;
         if (outputdir) {
            sandbox = outputdir;
         } else {
            sandbox = TString("/proc/") + TString(substatus->GetKey("user")) + TString("/") + TString(substatus->GetKey("queueId")) + TString("/job-output");
         }

         printf(" Sandbox [%-7s] %s \n", substatus->GetKey("queueId"),sandbox.Data());
         std::cout << " ================================================ " << std::endl;

         if (!gGrid->Cd(sandbox)) {
            continue;
         }

         TGridResult* dirlist = gGrid->Ls(sandbox);
         dirlist->Sort(kTRUE);
         Int_t i =0;
         while (dirlist->GetFileName(i)) {
            printf("%-24s ",dirlist->GetFileName(i++));
            if (!(i%4)) {
               printf("\n");
            }
         }
         printf("\n");
         delete dirlist;
      }
   }
   std::cout << " ----------LITE_JOB_OPERATIONS-------------------------------------- " << std::endl;
   delete status;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TAlienMasterJob::Merge()
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TAlienMasterJob::Merge(const char* inputname,const char* mergeoutput)
{
   TFileMerger merger;

   TAlienMasterJobStatus* status = (TAlienMasterJobStatus*)(GetJobStatus());
   TIterator* iter = status->GetJobs()->MakeIterator();

   TObjString* obj = 0;
   while ((obj = (TObjString*)iter->Next()) != 0) {
      TAlienJobStatus* substatus = (TAlienJobStatus*)status->GetJobs()->GetValue(obj->GetName());
      TString sandbox;// list sandboxes
      const char* outputdir = substatus->GetJdlKey("OutputDir");
      printf(" Sandbox [%-7s] %s \n", substatus->GetKey("queueId"),sandbox.Data());
      std::cout << " ================================================ " << std::endl;
      if (outputdir) {
         sandbox = outputdir;
      } else {
         sandbox = TString("/proc/") + TString(substatus->GetKey("user")) + TString("/") + TString(substatus->GetKey("queueId")) + TString("/job-output");
      }
      merger.AddFile(TString("alien://")+sandbox+ TString("/") + TString(inputname));
   }

   if (mergeoutput) {
      merger.OutputFile(mergeoutput);
   }

   return merger.Merge();
}
