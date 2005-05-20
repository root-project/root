// @(#)root/alien:$Name:  $:$Id: TAlien.h,v 1.8 2003/11/13 17:01:15 rdm Exp $
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
// To start a local API Grid service, use                               //
//   - TGrid::Connect("alien://localhost");                             //
//   - TGrid::Connect("alien://");                                      //
//                                                                      //
// To force to connect to a running API Service, use                    //
//   - TGrid::Connect("alien://<apihosturl>/?direct");                  //
//                                                                      //
// To get a remote API Service from the API factory service, use        //
//   - TGrid::Connect("alien://<apifactoryurl>");                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include "TUrl.h"
#include "TAlien.h"
#include "TString.h"
#include "TObjString.h"
#include "TObjArray.h"
#include "TMap.h"
#include "TAlienJDL.h"
#include "TAlienResult.h"
#include "TAlienJob.h"

#include "gliteUI.h"

using namespace std;


ClassImp(TAlien)

//______________________________________________________________________________
TAlien::TAlien(const char *gridurl, const char *uid, const char * /*passwd*/,
               const char *options)
{
   // Connect to the AliEn grid.

   TUrl *gurl = new TUrl(gridurl);

   fGridUrl = gridurl;
   fGrid    = "alien";
   fHost    = gurl->GetHost();
   fPort    = gurl->GetPort();
   fUser    = uid;
   fOptions = options;

   if (gDebug > 1)
      Info("TAlien", "%s => %s port: %d user: %s",gridurl,fHost.Data(),fPort,fUser.Data());

   fGc = GliteUI::MakeGliteUI(kFALSE);
   fGc->Connect(fHost, fPort, fUser);
   if (!fGc) {
      Error("TAlien", "could not connect to a alien service at:");
      Error("TAlien", "host: %s port: %d user: %s", fHost.Data(), fPort, fUser.Data());
      MakeZombie();
   } else {
      if (!fGc->Connected()) {
         Error("TAlien", "could not authenticate at:");
         Error("TAlien", "host: %s port: %d user: %s",fHost.Data(),fPort,fUser.Data());
         MakeZombie();
      } else {
         gGrid = this;
         Command("motd");
         Stdout();
      }
   }
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
   command += Escape(jdl);

   cout << command << endl;

   TGridResult* result = Command(command);
   TAlienResult* alienResult = dynamic_cast<TAlienResult*>(result);
   TList* list = dynamic_cast<TList*>(alienResult);
   if (!list) {
      if (result)
         delete result;
      return 0;
   }

   alienResult->DumpResult();

   GridJobID_t jobID = 0;

   TIterator* iter = list->MakeIterator();
   TObject* object = 0;
   while ((object = iter->Next()) != 0) {
      TMap* map = dynamic_cast<TMap*>(object);

      TObject* jobIDObject = map->GetValue("jobId");
      TObjString* jobIDStr = dynamic_cast<TObjString*>(jobIDObject);
      if (jobIDStr) {
         jobID = atoi(jobIDStr->GetString());
      }
   }
   delete iter;
   delete result;

   if (jobID == 0) {
      Error("Submit", "error submitting job");
      return 0;
   }

   Info("Submit", "your job was submitted with the ID = %d", jobID);

   return dynamic_cast<TGridJob*>(new TAlienJob(jobID));
}

//______________________________________________________________________________
TGridJDL *TAlien::GetJDLGenerator()
{
   return new TAlienJDL();
}

//______________________________________________________________________________
TGridResult *TAlien::Command(const char *command, bool interactive)
{
   // Execute AliEn command. Returns 0 in case or error.

   if (fGc) {
      if (fGc->Command(command)) {
         Int_t stream = kOUTPUT;
         // command successful
         TAlienResult* gresult = new TAlienResult();

         for (Int_t column = 0 ; column < (fGc->GetStreamColumns(stream)); column++) {
            TMap *gmap = new TMap();
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
   TString cmdline = TString("find -r ") + TString(options) + TString(" ") + TString(path) + TString(" ")  + TString(pattern) + TString(" ") + TString(conditions);
   return Command(cmdline.Data());
}

//______________________________________________________________________________
TGridResult *TAlien::OpenDataset(const char *lfn, const char *options)
{
   TString cmdline = TString("getdataset") + TString(" ") + TString(options) + TString(" ") + TString(lfn);
   return Command(cmdline.Data(),kTRUE);
}
