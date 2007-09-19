// @(#)root/main:$Id$
// Author: G Ganis 10/5/2007

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Launching program for remote ROOT sessions                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include "TInterpreter.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TPluginManager.h"
#include "TSystem.h"
#include "TString.h"

static Int_t MakeCleanupScript(Int_t loglevel);
static FILE *RedirectOutput(TString &logfile, const char *loc);

static const char *gAppName = "roots";

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // The main program: start a TApplication which connects back to the client.

   // Prepare the application
   if (argc < 4) {
      fprintf(stderr, "%s: insufficient input:"
                      " client URL must to be provided\n", gAppName);
      gSystem->Exit(1);
   }

   // Parse the debug level
   int loglevel = -1;
   TString argdbg(argv[3]);
   if (argdbg.BeginsWith("-d=")) {
      argdbg.ReplaceAll("-d=","");
      loglevel = argdbg.Atoi();
   }
   if (loglevel > 0) {
      fprintf(stderr,"%s: Starting remote session on %s\n", gAppName, gSystem->HostName());
      if (loglevel > 1) {
         fprintf(stderr,"%s:    argc: %d\n", gAppName, argc);
         for (Int_t i = 0; i < argc; i++)
            fprintf(stderr,"%s:    argv[%d]: %s\n", gAppName, i, argv[i]);
      }
   }

   // Cleanup script
   if (MakeCleanupScript(loglevel) != 0)
      fprintf(stderr,"%s: Error: failed to create cleanup script\n", gAppName);

   // Redirect the output
   TString logfile;
   FILE *fLog = RedirectOutput(logfile, ((loglevel > 1) ? gAppName : 0));
   if (fLog) {
      if (loglevel > 0)
         fprintf(stderr,"%s: output redirected to %s\n", gAppName, logfile.Data());
   } else {
      fprintf(stderr,"%s: problems redirecting output\n", gAppName);
      gSystem->Exit(1);
   }

   // Url to contact back
   TString url = argv[1];

   // Like in batch mode
   gROOT->SetBatch();

   // Enable autoloading
   gInterpreter->EnableAutoLoading();

   // Instantiate the TApplication object to be run
   TPluginHandler *h = 0;
   TApplication *theApp = 0;
   if ((h = gROOT->GetPluginManager()->FindHandler("TApplication","server"))) {
      if (h->LoadPlugin() == 0) {
         theApp = (TApplication *) h->ExecPlugin(4, &argc, argv, fLog, logfile.Data());
      } else {
         fprintf(stderr, "%s: failed to load plugin for TApplicationServer\n", gAppName);
      }
   } else {
      fprintf(stderr, "%s: failed to find plugin for TApplicationServer\n", gAppName);
   }

   // Run it
   if (theApp) {
      theApp->Run();
   } else {
      fprintf(stderr, "%s: failed to instantiate TApplicationServer\n", gAppName);
      gSystem->Exit(1);
   }

   // Done
   gSystem->Exit(0);
}

//______________________________________________________________________________
FILE *RedirectOutput(TString &logfile, const char *loc)
{
   // Redirect stdout to 'logfile'. This log file will be flushed to the
   // client or master after each command.
   // On success return a pointer to the open log file. Return 0 on failure.

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: enter\n", loc);

   // Log file under $TEMP
   logfile = Form("%s/roots-%d-%d.log", gSystem->TempDirectory(),
                                        gSystem->GetUid(), gSystem->GetPid());
   const char *lfn = logfile.Data();
   if (loc)
      fprintf(stderr,"%s: Path to log file: %s\n", loc, lfn);

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: reopen %s\n", loc, lfn);
   FILE *flog = freopen(lfn, "w", stdout);
   if (!flog) {
      fprintf(stderr,"%s: RedirectOutput: could not freopen stdout\n", loc);
      return 0;
   }

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: dup2 ...\n", loc);
   if ((dup2(fileno(stdout), fileno(stderr))) < 0) {
      fprintf(stderr,"%s: RedirectOutput: could not redirect stderr\n", loc);
      return 0;
   }

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: read open ...\n", loc);
   FILE *fLog = fopen(lfn, "r");
   if (!fLog) {
      fprintf(stderr,"%s: RedirectOutput: could not open logfile %s\n", loc, lfn);
      return 0;
   }

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: done!\n", loc);
   // We are done
   return fLog;
}

//______________________________________________________________________________
Int_t MakeCleanupScript(Int_t loglevel)
{
   // Create a script that can be executed to cleanup this process in case of
   // problems. Return 0 on success, -1 in case of any problem.

   // The file path
   TString cleanup = Form("%s/roots-%d-%d.cleanup", gSystem->TempDirectory(),
                                                    gSystem->GetUid(), gSystem->GetPid());
   // Open the file
   FILE *fc = fopen(cleanup.Data(), "w");
   if (fc) {
      fprintf(fc,"#!/bin/sh\n");
      fprintf(fc,"\n");
      fprintf(fc,"# Cleanup script for roots process %d\n", gSystem->GetPid());
      fprintf(fc,"# Usage:\n");
      fprintf(fc,"#   ssh %s@%s %s\n", gSystem->Getenv("USER"), gSystem->HostName(), cleanup.Data());
      fprintf(fc,"#\n");
      fprintf(fc,"kill -9 %d", gSystem->GetPid());
      // Close file
      fclose(fc);
      if (chmod(cleanup.Data(), S_IRUSR | S_IWUSR | S_IXUSR) != 0) {
         fprintf(stderr,"%s: Error: cannot make script %s executable\n", gAppName, cleanup.Data());
         unlink(cleanup.Data());
         return -1;
      } else {
         if (loglevel > 1)
            fprintf(stderr,"%s: Path to cleanup script %s\n", gAppName, cleanup.Data());
      }
   } else {
      fprintf(stderr,"%s: Error: file %s could not be created\n", gAppName, cleanup.Data());
      return -1;
   }

   // Done
   return 0;
}
