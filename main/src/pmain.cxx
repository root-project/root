// @(#)root/main:$Name:  $:$Id: pmain.cxx,v 1.10 2006/07/26 14:28:58 rdm Exp $
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

#ifdef HAVE_CONFIG
#include "config.h"
#endif
#include "TApplication.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

// Special type for the hook to the TXProofServ constructor, needed to avoid
// using the plugin manager
typedef TApplication *(*TProofServ_t)(Int_t *argc, char **argv, FILE *flog);

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

//______________________________________________________________________________
int main(int argc, char **argv)
{
   // PROOF server main program.

#ifdef R__DEBUG
   int debug = 1;
   while (debug)
      ;
#endif

   int loglevel = -1;
   if (getenv("ROOTPROOFLOGLEVEL"))
      loglevel = atoi(getenv("ROOTPROOFLOGLEVEL"));
   if (loglevel > 0)
      fprintf(stderr,"%s: starting %s\n", argv[1], argv[0]);

   // Redirect the output
   FILE *fLog = 0;
   char *logfile = 0;
   const char *sessdir = getenv("ROOTPROOFSESSDIR");
   if (sessdir && !getenv("ROOTPROOFDONOTREDIR")) {
      logfile = new char[strlen(sessdir) + 5];
      sprintf(logfile, "%s.log", sessdir);
      char *loc = (loglevel > 0) ? argv[1] : 0;
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

   gROOT->SetBatch();
   TApplication *theApp = 0;

   // Enable autoloading
   gInterpreter->EnableAutoLoading();

   TString getter("GetTProofServ");
#ifdef ROOTLIBDIR
   TString prooflib = TString(ROOTLIBDIR) + "/libProof";
#else
   TString prooflib = TString(gRootDir) + "/lib/libProof";
#endif
   if (argc > 2) {
      // XPD: additionally load the appropriate library
      prooflib.ReplaceAll("/libProof", "/libProofx");
      getter.ReplaceAll("GetTProofServ", "GetTXProofServ");
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

   // We can exit now
   gSystem->Exit(0);
}
