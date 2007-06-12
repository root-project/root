// @(#)root/main:$Name:  $:$Id: pmain.cxx,v 1.14 2007/06/08 09:17:25 rdm Exp $
// Author: Fons Rademakers   15/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// PMain                                                                //
//                                                                      //
// Main program used to create PROOF server application.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <errno.h>

#ifdef WIN32
#include <io.h>
#endif
#include <stdio.h>
#include <errno.h>

#ifdef R__HAVE_CONFIG
#include "RConfigure.h"
#endif
#ifdef R__AFS
#include "TAFS.h"
#endif
#include "TApplication.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

// Special type for the hook to the TXProofServ constructor, needed to avoid
// using the plugin manager
typedef TApplication *(*TProofServ_t)(Int_t *argc, char **argv, FILE *flog);
#ifdef R__AFS
// Special type for the hook to the TAFS constructor, needed to avoid
// using the plugin manager
typedef TAFS *(*TAFS_t)(const char *, const char *, Int_t);
// Instance of the AFS token class
static TAFS *gAFS = 0;
#endif

//______________________________________________________________________________
static FILE *RedirectOutput(const char *logfile, const char *loc)
{
   // Redirect stdout to 'logfile'. This log file will be flushed to the
   // client or master after each command.
   // On success return a pointer to the open log file. Return 0 on failure.

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: enter: %s\n", loc, logfile);

   if (!logfile || strlen(logfile) <= 0) {
      fprintf(stderr,"%s: RedirectOutput: logfile path undefined\n", loc);
      return 0;
   }

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: reopen %s\n", loc, logfile);
   FILE *flog = freopen(logfile, "w", stdout);
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
   FILE *fLog = fopen(logfile, "r");
   if (!fLog) {
      fprintf(stderr,"%s: RedirectOutput: could not open logfile %s\n", loc, logfile);
      return 0;
   }

   if (loc)
      fprintf(stderr,"%s: RedirectOutput: done!\n", loc);
   // We are done
   return fLog;
}

#ifdef R__AFS
//______________________________________________________________________________
static Int_t InitAFS(const char *fileafs, const char *loc)
{
   // Init AFS token using credentials at fileafs

   TString getter("GetTAFS");
   char *p = 0;
   TString afslib = "libAFSAuth";
   if ((p = gSystem->DynamicPathName(afslib, kTRUE))) {
      delete[] p;
      if (gSystem->Load(afslib) == -1) {
         if (loc)
            fprintf(stderr,"%s: can't load %s\n", loc, afslib.Data());
         return -1;
      }
   } else {
      if (loc)
         fprintf(stderr,"%s: can't locate %s\n", loc, afslib.Data());
      return -1;
   }

   // Locate constructor
   Func_t f = gSystem->DynFindSymbol(afslib, getter);
   if (f) {
      gAFS = (*((TAFS_t)f))(fileafs, 0, -1);
      if (!gAFS) {
         if (loc)
            fprintf(stderr,"%s: could not initialize a valid TAFS\n", loc);
         return -1;
      }
   } else {
      if (loc)
         fprintf(stderr,"%s: can't find %s\n", loc, getter.Data());
      return -1;
   }

   // Done
   return 0;
}
#endif

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // PROOF server main program.

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif
   int loglevel = (argc >= 6) ? strtol(argv[5], 0, 10) : -1;
   if (loglevel < 0 && getenv("ROOTPROOFLOGLEVEL"))
      loglevel = atoi(getenv("ROOTPROOFLOGLEVEL"));
   if (loglevel > 0)
      fprintf(stderr,"%s: starting %s\n", argv[1], argv[0]);

   // Redirect the output
   FILE *fLog = 0;
   char *loc = 0;
   char *logfile = (char *)getenv("ROOTPROOFLOGFILE");
   if (logfile && !getenv("ROOTPROOFDONOTREDIR")) {
      loc = (loglevel > 0) ? argv[1] : 0;
      if (loglevel > 0)
         fprintf(stderr,"%s: redirecting output to %s\n", argv[1], logfile);
      if (!(fLog = RedirectOutput(logfile, loc))) {
         fprintf(stderr,"%s: problems redirecting output to file %s\n", argv[1], logfile);
         exit(1);
      }
   }
   if (loglevel > 0)
      fprintf(stderr,"%s: output redirected to: %s\n",
             argv[1], (logfile ? logfile : "+++not redirected+++"));

#ifdef R__AFS
   // Init AFS, if required
   if (getenv("ROOTPROOFAFSCREDS")) {
      if (InitAFS(getenv("ROOTPROOFAFSCREDS"), loc) != 0) {
          fprintf(stderr,"%s: unable to initialize the AFS token\n", argv[1]);
      } else {
         if (loglevel > 0)
            fprintf(stderr,"%s: AFS token initialized\n", argv[1]);
      }
   }
#endif

   gROOT->SetBatch();
   TApplication *theApp = 0;

   // Enable autoloading
   gInterpreter->EnableAutoLoading();

   TString getter("GetTProofServ");
   TString prooflib = "libProof";
   if (argc > 2) {
      // XPD: additionally load the appropriate library
      prooflib = "libProofx";
      getter = "GetTXProofServ";
   }
   char *p = 0;
   if ((p = gSystem->DynamicPathName(prooflib, kTRUE))) {
      delete[] p;
      if (gSystem->Load(prooflib) == -1) {
         fprintf(stderr,"%s: can't load %s\n", argv[1], prooflib.Data());
         exit(1);
      }
   } else {
      fprintf(stderr,"%s: can't locate %s\n", argv[1], prooflib.Data());
      exit(1);
   }

   // Locate constructor
   Func_t f = gSystem->DynFindSymbol(prooflib, getter);
   if (f) {
      theApp = (TApplication *) (*((TProofServ_t)f))(&argc, argv, fLog);
   } else {
      fprintf(stderr,"%s: can't find %s\n", argv[1], getter.Data());
      exit(1);
   }

   // Ready to run
   if (loglevel > 0)
      fprintf(stderr,"%s: running the TProofServ application\n", argv[1]);

   theApp->Run();

#ifdef R__AFS
   // Cleanup
   if (gAFS)
      delete gAFS;
#endif

   // We can exit now
   gSystem->Exit(0);
}
