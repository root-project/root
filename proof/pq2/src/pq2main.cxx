// @(#)root/proof:$Id$
// Author: G. Ganis, Mar 2010

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// ************************************************************************* //
// *                                                                       * //
// *                           p q 2 m a i n                               * //
// *                                                                       * //
// * This file implements the steering main for PD2                        * //
// * The tests can be run as a standalone program or with the interpreter. * //
// *                                                                       * //
// ************************************************************************* //

#include <stdio.h>
#include <stdlib.h>

#include "TMacro.h"
#include "TString.h"
#include "TSystem.h"
#include "TUUID.h"

#include "pq2actions.h"
#include "pq2ping.h"

// Local prototype and global variables
void showFile(const char *fn, int show, int keep);
static Int_t gkeep = 0;

// Global variables used by other PQ2 components
TString flog;
TString ferr;
TString fres;
Int_t gverbose = 0;

//_____________________________batch only_____________________
int main(int argc, char **argv)
{

   // Request for help?
   if (argc > 1 && !strcmp(argv[1],"-h")) {
      printf(" \n");
      printf(" PQ2 functionality\n");
      printf(" \n");
      printf(" Usage:\n");
      printf(" \n");
      printf(" $ ./pq2 [-h] [-v] [-k] <action> [-d datasetname|datasetfile] [-s server] -u serviceurl\n");
      printf(" \n");
      printf(" Arguments:\n");
      printf("   -h            prints this menu\n");
      printf("   -v            verbose mode\n");
      printf("   -k            keep temporary files\n");
      printf("   <action>      ls, ls-files, ls-files-server, info-server, put, rm, verify\n");
      printf("   datasetname   Name of the dataset; the wild card '*' is accepted: in \n");
      printf("                 such a case the full path - as shown by pq2-ls - must \n");
      printf("                 be given in quotes, e.g. \"/default/ganis/h1-set5*\"\n");
      printf("                 (applies to: ls-files, ls-files-server, rm, verify\n");
      printf("   datasetfile   Path to the file with the list of files in the dataset or \n");
      printf("                 directory with the files containing the file lists of the \n");
      printf("                 datasets to be registered; in the first case wildcards '*' \n");
      printf("                 can be specified in the file name, i.e. \"<dir>/fil*\" is ok \n");
      printf("                 but \"<dir>/*/file\" is not. In all cases the name of the \n");
      printf("                 dataset is the name of the file finally used\n");
      printf("                 (applies to: put)\n");
      printf("   server        Name of the server for which the information is wanted; can be in \n");
      printf("                 URL form \n");
      printf("                 (applies to: ls-files-server, info-server)\n");
      printf("   serviceurl    entry point of the service to be used to get the information (PROOF master\n");
      printf("                 or data server) in the form '[user@]host.domain[:port]'\n");
      printf(" \n");
      gSystem->Exit(0);
   }

   // Parse options
   const char *action = 0;
   const char *url = 0;
   const char *dataset = 0;
   const char *servers = 0;
   const char *options = 0;
   const char *ignsrvs = 0;
   const char *excsrvs = 0;
   const char *metrics = 0;
   const char *fout = 0;
   const char *plot = 0;
   const char *infile = 0;
   const char *outfile = 0;
   const char *redir = 0;
   Int_t i = 1;
   while (i < argc) {
      if (!strcmp(argv[i],"-h")) {
         // Ignore if not first argument
         i++;
      } else if (!strcmp(argv[i],"-d")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -d should be followed by a string: ignoring");
            i++;
         } else {
            dataset = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-o")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -o should be followed by a string: ignoring");
            i++;
         } else {
            options = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-e") ||!strcmp(argv[i],"--exclude") ) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -e or --exclude should be followed by a string: ignoring");
            i++;
         } else {
            excsrvs = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-i") ||!strcmp(argv[i],"--ignore") ) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -i or --ignore should be followed by a string: ignoring");
            i++;
         } else {
            ignsrvs = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-s") || !strcmp(argv[i],"--servers")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -s or --servers should be followed by a string: ignoring");
            i++;
         } else {
            servers = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-m")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -m should be followed by a string: ignoring");
            i++;
         } else {
            metrics = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-f")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -f should be followed by a string: ignoring");
            i++;
         } else {
            fout = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-r")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -r should be followed by a string: ignoring");
            i++;
         } else {
            redir = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"-u")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" -u should be followed by a string: ignoring");
            i++;
         } else {
            url = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"--plot")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            plot = "plot.png";
            i++;
         } else {
            plot = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"--infile")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" --infile should be followed by a string: ignoring");
            i++;
         } else {
            infile = argv[i+1];
            i += 2;
         }
      } else if (!strcmp(argv[i],"--outfile")) {
         if (i+1 == argc || argv[i+1][0] == '-') {
            Printf(" --outfile should be followed by a string: ignoring");
            i++;
         } else {
            outfile = argv[i+1];
            i += 2;
         }
      } else if (!strncmp(argv[i],"-v",2)) {
         gverbose++;
         if (!strncmp(argv[i],"-vv", 3)) gverbose++;
         if (!strncmp(argv[i],"-vvv", 4)) gverbose++;
         i++;
      } else if (!strncmp(argv[i],"-k",2)) {
         gkeep++;
         i++;
      } else {
         action = argv[i];
         i++;
      }
   }
   if (!action) {
      Printf("Specifying an action is mandatory - exit");
      gSystem->Exit(1);
   }
   if (gverbose > 0) Printf("action: %s (url: %s)", action, url);

   // Find out the action index
   const int nact = 9;
   const char *actions[nact] = { "ls", "ls-files", "ls-files-server",
                                 "info-server", "put", "rm", "verify",
                                 "ana-dist", "cache" };
   const char *tags[nact]    = { "ls", "lsfiles", "filessrv",
                                 "infosrv", "put", "rm", "vfy", "anadist", "cache" };
   const char *tag = 0;
   Int_t iact = -1;
   for (i = 0; i < nact; i++) {
      if (action && !strcmp(action, actions[i])) {
         iact = i;
         tag = tags[i];
      }
   }
   if (iact == -1) {
      Printf("Unknown action: %d (%s)", iact, action ? action : "");
      gSystem->Exit(1);
   }

   // Unique temporary dir
   if (gverbose > 0) Printf("Tmp dir: %s", gSystem->TempDirectory());
   TString tdir(gSystem->TempDirectory());
   UserGroup_t *ug = gSystem->GetUserInfo(gSystem->GetUid());
   if (ug) {
      if (!(tdir.EndsWith(ug->fUser))) {
         if (!(tdir.EndsWith("/"))) tdir += "/";
         tdir += ug->fUser;
      }
      SafeDelete(ug);
   }
   if (gSystem->AccessPathName(tdir) && gSystem->mkdir(tdir, kTRUE) != 0) {
      Printf("Could create temp directory at: %s", tdir.Data());
      gSystem->Exit(1);
   }
   flog.Form("%s/pq2%s.log", tdir.Data(), tag);
   ferr.Form("%s/pq2%s.err", tdir.Data(), tag);
   fres.Form("%s/pq2%s.res", tdir.Data(), tag);
   if (!gSystem->AccessPathName(ferr)) gSystem->Unlink(ferr);
   if (!gSystem->AccessPathName(flog)) gSystem->Unlink(flog);
   if (!gSystem->AccessPathName(fres)) gSystem->Unlink(ferr);

   // Check URL
   bool def_proof= 0;
   if (!url) {
      // List of actions to be done via server
      TString dsmgracts = getenv("PQ2DSSRVACTS") ? getenv("PQ2DSSRVACTS")
                                                 : "ls:lsfiles:filessrv:infosrv:anadist:cache:" ;
      // Determine the server to be used
      TString atag(TString::Format("%s:", tag));
      if (dsmgracts.Contains(atag) && getenv("PQ2DSSRVURL")) {
         url = getenv("PQ2DSSRVURL");
      } else if (getenv("PROOFURL") || getenv("PQ2PROOFURL")) {
         url = getenv("PQ2PROOFURL") ? getenv("PQ2PROOFURL") : getenv("PROOFURL");
         def_proof = 1;
      } else {
         Printf("Specifying a service URL is mandatory - exit");
         gSystem->Exit(1);
      }
   }
   if (gverbose > 0) Printf("Checking URL: %s", url ? url : "--undef--");
   Int_t urlrc = checkUrl(url, flog.Data(), def_proof);
   if (urlrc < 0) {
      Printf("Specified URL does not identifies a running service: %s", url);
      gSystem->Exit(1);
   }

   Int_t rc = 0;
   try {
      if (iact == 0) {
         // ls
         do_ls(dataset, options);

      } else if (iact == 1) {
         // ls-files
         do_ls_files_server(dataset, 0);

      } else if (iact == 2) {
         // ls-files-server
         do_ls_files_server(dataset, servers);

      } else if (iact == 3) {
         // info-server
         do_info_server(servers);

      } else if (iact == 4) {
         // put
         do_put(dataset, options);

      } else if (iact == 5) {
         // rm
         do_rm(dataset);

      } else if (iact == 6) {
         // verify
         rc = do_verify(dataset, options, redir);

      } else if (iact == 7) {
         // ana-dist
         do_anadist(dataset, servers, ignsrvs, excsrvs, metrics, fout, plot, outfile, infile);

      } else if (iact == 8) {
         // cache
         bool clear = (options && !strcmp(options, "clear")) ? 1 : 0;
         do_cache(clear, dataset);

      } else {
         // Unknown
         Printf("Unknown action code: %d - Protocol error?", iact);
      }
   }
   catch (std::exception& exc) {
      Printf("Standard exception caught: we abort whatever it is ...");
      throw exc;
   }
   catch (const char *str) {
      Printf("Exception thrown: %s", str);
   }
   // handle every exception
   catch (...) {
      Printf("Handle uncaugth exception, terminating");
   }

   if (!gSystem->AccessPathName(ferr)) {
      showFile(ferr, 1, gkeep);
   } else {
      if (!gSystem->AccessPathName(flog)) showFile(flog, gverbose, gkeep);
      if (!gSystem->AccessPathName(fres)) showFile(fres, 1, gkeep);
   }
   if (gkeep > 0) {
      Printf("Temporary files kept: ");
      if (!gSystem->AccessPathName(ferr)) Printf(" -> %s", ferr.Data());
      if (!gSystem->AccessPathName(flog)) Printf(" -> %s", flog.Data());
      if (!gSystem->AccessPathName(fres)) Printf(" -> %s", fres.Data());
   }

   gSystem->Exit(rc);
}

//_______________________________________________________________________________________
void showFile(const char *fn, int show, int keep)
{
   // Display the content of file 'fn'

   if (fn && strlen(fn)) {
      FileStat_t st;
      if (gSystem->GetPathInfo(fn, st) != 0 || !R_ISREG(st.fMode)) {
         Printf("File '%s' cannot be stated or is not regular: ignoring", fn);
         return;
      }
      if (show > 0) { TMacro m(fn); m.Print(); }
      if (keep == 0) gSystem->Unlink(fn);
   }
}
