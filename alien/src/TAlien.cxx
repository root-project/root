// @(#)root/alien:$Name:  $:$Id: TAlien.cxx,v 1.20 2007/03/19 16:55:55 rdm Exp $
// Author: Andreas Peters   5/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlien                                                               //
//                                                                      //
// Class defining interface to TAlien GRID services.                    //
//                                                                      //
// To force to connect to a running API Service, use                    //
//   - TGrid::Connect("alien://<apihosturl>/","user");                  //
//                                                                      //
// If you want to use TGrid::Connect("alien://");                       //
// you have to set the environment variables                            //
// export alien_API_HOST=<host>                                         //
// export alien_API_PORT=<port>                                         //
// export alien_API_USER=<user>                                         //
//                                                                      //
//                                                                      //
// some examples:                                                       //
// -------------------------------------------------------------------- //
// connect in an initialized API environemnt (see above)                //
// > TGrid::Connect("alien://");                                        //
// -> this is not thread safe                                           //
// connect using an already established shell token:                    //
// > TGrid::Connect("alien://",0,0,"t");                                //
// -> this is thread safe                                               //
// -------------------------------------------------------------------- //
// change the working directory                                         //
// > gGrid->Cd("/alice"); //=> returns 1 for success, 0 for error       //
//                                                                      //
//                                                                      //
// -------------------------------------------------------------------- //
// get the working directory                                            //
// > printf("Working Directory is %s\n",gGrid->Pwd());                  //
//                                                                      //
//                                                                      //
// -------------------------------------------------------------------- //
// get a filename list of the working directory and print it            //
// > TGridResult* result = gGrid->Ls("/alice");                         //
// > Int_t i=0;                                                         //
// > while (result->GetFileName(i))\                                    //
// printf("File %s\n",result->GetFileName(i++));                        //
// -------------------------------------------------------------------- //
//                                                                      //
// get all file permissions                                             //
// > TGridResult* result = gGrid->Ls("/alice","-la");                   //
// > while (result->GetFileName(i))\                                    //
// printf("The permissions are %s\n",result->GetKey(i++,"permissions")  //
//                                                                      //
// => the defined keys for GetKey can be seen using result->Print();    //
// -------------------------------------------------------------------- //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TUrl.h"
#include "TAlien.h"
#include "TString.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TMap.h"
#include "TSystem.h"
#include "TAlienJDL.h"
#include "TAlienResult.h"
#include "TAlienJob.h"
#include "TAlienCollection.h"
#include "TAlienMasterJobStatus.h"
#include "TAlienJobStatusList.h"
#include "TAlienJobStatus.h"
#include "gapiUI.h"
#include "gapi_job_operations.h"

using namespace std;



ClassImp(TAlien)

//______________________________________________________________________________
TAlien::TAlien(const char *gridurl, const char *uid, const char * passwd,
               const char *options)
{
   gSystem->Unsetenv("GCLIENT_EXTRA_ARG");

   // Connect to the AliEn grid.
   TUrl *gurl = new TUrl(gridurl);

   fGridUrl = gridurl;
   fGrid    = "alien";
   if (!strlen(gurl->GetHost())) {
      if (gSystem->Getenv("alien_API_HOST"))
         fHost = gSystem->Getenv("alien_API_HOST");
      else
         fHost = "";
   } else {
      fHost = gurl->GetHost();
   }

   if (gurl->GetPort()<=0) {
      if (gSystem->Getenv("alien_API_PORT"))
         fPort = atoi(gSystem->Getenv("alien_API_PORT"));
      else
         fPort = 0;
   } else {
      fPort = gurl->GetPort();
   }

   if (!strlen(uid)) {
      if (gSystem->Getenv("alien_API_USER")) {
         fUser = gSystem->Getenv("alien_API_USER");
      }
   } else {
      fUser = uid;
   }

   fOptions = options;

   if (gDebug > 1)
      Info("TAlien", "%s => %s port: %d user: %s",gridurl,fHost.Data(),fPort,fUser.Data());

   Bool_t fstoken;
   fstoken = kFALSE;

   if (options && (options[0] == 't')) {
       fGc = GapiUI::MakeGapiUI(kTRUE);
       fstoken = kTRUE;
   } else {
       fGc = GapiUI::MakeGapiUI(kFALSE);
   }

   if (!fGc) {
      Error("TAlien", "could not connect to a alien service at:");
      Error("TAlien", "host: %s port: %d user: %s", fHost.Data(), fPort, fUser.Data());
      MakeZombie();
   } else {
      if (passwd) {
         if (!strlen(passwd)) {
            passwd = 0;
         }
      }
      if (fstoken) {
         if (!Command("motd")) {
            Error("TAlien", "we have no valid connection ... I try to connect ...");
            fGc->Connect(fHost, fPort, fUser, passwd);
         }
      } else {
         fGc->Connect(fHost, fPort, fUser, passwd);
      }

      char cwd[1024];
      const char* cwdp=0;
      cwdp=Pwd();
      fHome=TString("/");
      if (cwdp) {
         sprintf(cwd,"%s",cwdp);
         if (Cd("")) {
            cwdp=Pwd();
            if (cwdp) {
               fHome = TString(cwdp);
            }
         }
      }

      if (!fGc->Connected()) {
         Error("TAlien", "could not authenticate at:");
         Error("TAlien", "host: %s port: %d user: %s",fHost.Data(),fPort,fUser.Data());
         MakeZombie();
      } else {
         gGrid = this;
         Command("motd");
         Stdout();
      }
      // export this UI to all POSIX functions like gapi_dir_xxx gapi_job_xxx
      fGc->SetGapiUI(fGc);
   }
   delete gurl;
}

//______________________________________________________________________________
TAlien::~TAlien()
{
   // do we need to delete fGc ? (rdm)

   if (gDebug > 1)
      Info("~TAlien", "destructor called");
}

//______________________________________________________________________________
void TAlien::Shell()
{
   // Start an interactive AliEn shell.

   fGc->Shell();
}

//______________________________________________________________________________
TString TAlien::Escape(const char *input)
{
   // Escape \" by \\\".

   if (!input)
      return TString();

   TString output(input);
   output.ReplaceAll("\"", "\\\"");

   return output;
}

//______________________________________________________________________________
TGridJob *TAlien::Submit(const char *jdl)
{
   // Submit a command to AliEn. Returns 0 in case of error.

   if (!jdl)
      return 0;

   TString command("submit =< ");
   //command += Escape(jdl);
   command += jdl;

   cout << command << endl;

   TGridResult* result = Command(command,kFALSE,kOUTPUT);
   TAlienResult* alienResult = dynamic_cast<TAlienResult*>(result);
   TList* list = dynamic_cast<TList*>(alienResult);
   if (!list) {
      if (result)
         delete result;
      return 0;
   }

   alienResult->DumpResult();

   TString jobID = "0";

   TIterator* iter = list->MakeIterator();
   TObject* object = 0;
   while ((object = iter->Next()) != 0) {
      TMap* map = dynamic_cast<TMap*>(object);

      TObject* jobIDObject = map->GetValue("jobId");
      TObjString* jobIDStr = dynamic_cast<TObjString*>(jobIDObject);
      if (jobIDStr) {
         jobID = jobIDStr->GetString();
      }
   }
   delete iter;
   delete result;

   if (jobID == "0") {
      Error("Submit", "error submitting job");
      return 0;
   }

   Info("Submit", "your job was submitted with the ID = %s", jobID.Data());

   return dynamic_cast<TGridJob*>(new TAlienJob(jobID));
}

//______________________________________________________________________________
TGridJDL *TAlien::GetJDLGenerator()
{
   // Create a Alien JDL.

   return new TAlienJDL();
}

//______________________________________________________________________________
TGridJobStatusList *TAlien::Ps(const char* /*options*/, Bool_t verbose)
{
   // Get job status list.

   GAPI_JOBARRAY* gjobarray = gapi_queryjobs("-", gGrid->GetUser(), "-", "-", "-",
                                             "-", "-", "-", "-");
   if (!gjobarray)
      return 0;

   if (gjobarray->size() == 0) {
      delete gjobarray;
      return 0;
   }

   TGridJobStatusList* joblist = new TAlienJobStatusList();

   for (UInt_t i = 0; i < gjobarray->size(); i++ ) {
      TMap* statusmap = new TMap();
      GAPI_JOB gjob = gjobarray->at(i);
      std::map<std::string, std::string>::const_iterator iter = gjob.gapi_jobmap.begin();
      for (; iter != gjob.gapi_jobmap.end(); ++iter) {
         //      printf("Adding %s = %s\n", iter->first.c_str(),iter->second.c_str());
         statusmap->Add(new TObjString(iter->first.c_str()), new TObjString(iter->second.c_str()));
      }
      TGridJobStatus *status = (TGridJobStatus*)new TAlienJobStatus(statusmap);
      joblist->Add((TGridJobStatus*)status);
      if (verbose) {
      }
   }
   delete gjobarray;
   return joblist;
}

//______________________________________________________________________________
Bool_t TAlien::Kill(UInt_t jobid)
{
   // Kill a specific job.

   if (gapi_kill(jobid))
      return kFALSE;
   else
      return kTRUE;
}

//______________________________________________________________________________
UInt_t TAlien::Resubmit(UInt_t jobid)
{
   // Resubmit a specific job.

   if (gapi_resubmit(jobid))
      return kFALSE;
   else
      return kTRUE;
}

//______________________________________________________________________________
TGridResult *TAlien::Command(const char *command, bool interactive, UInt_t stream)
{
   // Execute AliEn command. Returns 0 in case or error.

   if (fGc) {
      if (fGc->Command(command)) {
         // command successful
         TAlienResult* gresult = new TAlienResult();

         for (Int_t column = 0 ; column < (fGc->GetStreamColumns(stream)); column++) {
            TMap *gmap = new TMap();
            gmap->SetOwner(kTRUE);
            for (Int_t row=0; row < fGc->GetStreamRows(stream,column); row++) {
               gmap->Add((TObject*)(new TObjString(fGc->GetStreamFieldKey(stream,column,row))),
                         (TObject*)(new TObjString(fGc->GetStreamFieldValue(stream,column,row))));
            }
            gresult->Add(gmap);
         }

         if (interactive) {
            // write also stdout/stderr to the screen
            fGc->DebugDumpStreams();
         }
         return gresult;
      }
   }
   return 0;
}

//______________________________________________________________________________
TGridResult *TAlien::LocateSites()
{
   return Command("locatesites");
}

//______________________________________________________________________________
void TAlien::Stdout()
{
   if (fGc) {
      fGc->PrintCommandStdout();
   }
}

//______________________________________________________________________________
void TAlien::Stderr()
{
   if (fGc) {
      fGc->PrintCommandStderr();
   }
}

//______________________________________________________________________________
TGridResult *TAlien::Query(const char *path, const char *pattern,
                           const char *conditions, const char *options)
{
   TString cmdline = TString("find -z ") + TString(options) + TString(" ") + TString(path) + TString(" ")  + TString(pattern) + TString(" ") + TString(conditions);
   return Command(cmdline);
}

//______________________________________________________________________________
TGridResult *TAlien::OpenDataset(const char *lfn, const char *options)
{
   TString cmdline = TString("getdataset") + TString(" ") + TString(options) + TString(" ") + TString(lfn);
   return Command(cmdline, kTRUE);
}

//______________________________________________________________________________
TMap *TAlien::GetColumn(UInt_t stream, UInt_t column)
{
   TMap *gmap = new TMap();
   for (Int_t row = 0; row < fGc->GetStreamRows(stream,column); row++) {
      gmap->Add((TObject*)(new TObjString(fGc->GetStreamFieldKey(stream,column,row))),
                (TObject*)(new TObjString(fGc->GetStreamFieldValue(stream,column,row))));
   }
   return gmap;
}

//______________________________________________________________________________
const char *TAlien::GetStreamFieldValue(UInt_t stream, UInt_t column, UInt_t row)
{
   return fGc->GetStreamFieldValue(stream,column,row);
}

//______________________________________________________________________________
const char *TAlien::GetStreamFieldKey(UInt_t stream, UInt_t column, UInt_t row)
{
   return fGc->GetStreamFieldKey(stream,column,row);
}

//______________________________________________________________________________
UInt_t TAlien::GetNColumns(UInt_t stream)
{
   return fGc->GetStreamColumns(stream);
}

//--- Catalogue Interface

//______________________________________________________________________________
TGridResult *TAlien::Ls(const char* ldn, Option_t* options, Bool_t verbose)
{
   TString cmdline = TString("ls") + TString(" ") + TString(options) + TString(" ") + TString(ldn);

   return Command(cmdline, verbose);
}

//______________________________________________________________________________
Bool_t TAlien::Cd(const char* ldn, Bool_t verbose)
{
   TString cmdline = TString("cd") + TString(" ") + TString(ldn);

   Command(cmdline, kFALSE);

   if (verbose) {
      Stdout();
      Stderr();
   }

   const char* result = (GetStreamFieldValue(kOUTPUT,0,0));
   if (result) {
      if (strlen(result) > 0) {
         if (atoi(result) == 1) {
            return kTRUE;
         }
      }
   }

   Error("Cd","Cannot change to directory %s\n",ldn);
   if (!verbose) Stdout();
   return kFALSE;
}

//______________________________________________________________________________
const char* TAlien::Pwd(Bool_t verbose)
{
   TString cmdline = TString("pwd");

   TGridResult* result = Command(cmdline, kFALSE, kENVIR);

   if (verbose) {
      Stdout();
      Stderr();
   }

   if (result) {
      TMap* resultmap = ((TMap*)result->At(0));
      if (resultmap) {
         TObjString* pwd = (TObjString*)resultmap->GetValue("pwd");
         if (pwd) {
            fPwd = pwd->GetName();
            delete resultmap;
            return fPwd;
         } else {
            delete resultmap;
            return 0;
         }
      }
   }

   Error("pwd","Cannot get current working directory\n");
   if (!verbose) Stdout();
   return 0;
}

//______________________________________________________________________________
Bool_t TAlien::Mkdir(const char* ldn, Option_t* options, Bool_t verbose)
{
   TString cmdline = TString("mkdir");
   if (strlen(options)) {
      cmdline +=  (TString(" ") + TString(options));
   } else {
      cmdline += (TString(" -s ") + TString(ldn));
   }

   Command(cmdline, kFALSE);

   if (verbose) {
      Stdout();
      Stderr();
   }

   const char* result = (GetStreamFieldValue(kOUTPUT,0,0));
   if (result) {
      if (strlen(result) > 0) {
         if (atoi(result) == 1) {
            return kTRUE;
         }
      }
   }

   Error("Mkdir","Cannot create directory %s\n",ldn);
   if (!verbose) Stdout();
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TAlien::Rmdir(const char* ldn, Option_t* options, Bool_t verbose)
{
   TString cmdline = TString("rmdir");
   if (strlen(options)) {
      cmdline +=  (TString(" ") + TString(options));
   } else {
      cmdline += (TString(" -s ") + TString(ldn));
   }

   Command(cmdline, kFALSE);

   if (verbose) {
      Stdout();
      Stderr();
   }

   const char* result = (GetStreamFieldValue(kOUTPUT,0,0));
   if (result) {
      if (strlen(result) > 0) {
         if (atoi(result) == 1) {
            return kTRUE;
         }
      }
   }

   Error("Rmdir","Cannot remove directory %s\n",ldn);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlien::Register(const char* lfn, const char* turl, Long_t size, const char* se, const char* guid, Bool_t verbose)
{
   TString cmdline = TString("register") + TString(" ") + TString(lfn) + TString(" ") + TString(turl);
   if (se) {
      cmdline += TString(" ");
      cmdline += size;
      cmdline += TString(" ");
      cmdline += TString(se);
      if (guid) {
         cmdline += TString(" ");
         cmdline += TString(guid);
      }
   }

   Command(cmdline, kFALSE);

   if (verbose) {
      Stdout();
      Stderr();
   }

   const char* result = (GetStreamFieldValue(kOUTPUT,0,0));
   if (result) {
      if (strlen(result) > 0) {
         if (atoi(result) == 1) {
            return kTRUE;
         }
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TAlien::Rm(const char* lfn, Option_t* options, Bool_t verbose)
{
   TString cmdline = TString("rm") + TString(" -s ") + TString(options) + TString(" ") + TString(lfn);

   Command(cmdline, kFALSE);

   if (verbose) {
      Stdout();
      Stderr();
   }

   const char* result = (GetStreamFieldValue(kOUTPUT,0,0));
   if (result) {
      if (strlen(result) > 0) {
         if (atoi(result) == 1) {
            return kTRUE;
         }
      }
   }

   Error("Rm","Cannot remove %s\n",lfn);
   return kFALSE;
}

//______________________________________________________________________________
TGridCollection *TAlien::OpenCollection(const char *collectionfile,
                                        UInt_t maxentries)
{
   // Factory function for a TAlienCollection based on an XML file.

   return (TGridCollection*)TAlienCollection::Open(collectionfile, maxentries);
}

//______________________________________________________________________________
TGridCollection *TAlien::OpenCollectionQuery(TGridResult *queryresult,
                                             Bool_t nogrouping)
{
   // Factory function fo a TAlienCollection based on a gGrid Query.

   return (TGridCollection*)TAlienCollection::OpenQuery(queryresult, nogrouping);
}
