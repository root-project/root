// @(#)root/net:$Name:  $:$Id: TCastorFile.cxx,v 1.11 2006/04/18 05:02:08 brun Exp $
// Author: Fons Rademakers + Jean-Damien Durand 17/09/2003 + Ben Couturier 31/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TCastorFile                                                          //
//                                                                      //
// A TCastorFile is like a normal TNetFile except that it obtains the   //
// remote node (disk server) via the CASTOR API, once the disk server   //
// and the local file path are determined, the file will be accessed    //
// via the rootd daemon. File names have to be specified like:          //
//    castor:/castor/cern.ch/user/r/rdm/bla.root.                       //
//                                                                      //
// This class has been modified to be able to use a V1 or V2 stager.    //
// It uses the V1 stager API by default, and it can connect to a new    //
// CASTOR stager when defining the shell variable                       //
//    RFIO_USE_CASTOR_V2=YES                                            //
// or adding RFIO USE_CASTOR_V2 YES in /etc/castor.conf.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "NetErrors.h"
#include "TCastorFile.h"
#include "TError.h"

#include <stdlib.h>
#include <errno.h>

#ifdef _WIN32
#include <WinDef.h>
#include <WinSock2.h>
#endif

#ifdef R__CASTOR2
#include <stager_api.h>       // For the new CASTOR 2 Stager
#endif
#define RFIO_KERNEL           // Get access to extra symbols in the headers
#include <stage_api.h>        // Dial with CASTOR stager
#include <rfio_api.h>         // Util routines from rfio
#include <Cns_api.h>          // Dial with CASTOR Name Server

#define RFIO_USE_CASTOR_V2 "RFIO_USE_CASTOR_V2"
extern "C" { char *getconfent(char *, char *, int); }


//______________________________________________________________________________
static int UseCastor2API()
{
   // Function that checks whether we should use the old or new stager API.

   char *p;

   if (((p = getenv(RFIO_USE_CASTOR_V2)) == 0) &&
       ((p = getconfent("RFIO","USE_CASTOR_V2",0)) == 0)) {
      // Variable not set: compat mode
      return 0;
   }
   if ((strcmp(p,"YES") == 0) || (strcmp(p,"yes") == 0) || (atoi(p) == 1)) {
#ifdef R__CASTOR2
      // Variable set to yes or 1 : new mode
      return 1;
#else
      // Variable set to yes or 1 but old CASTOR 1: compat mode + warning
      static int once = 0;
      if (!once) {
         ::Warning("UseCastor2API", "asked to use CASTOR 2, but linked with CASTOR 1");
         once = 1;
      }
      return 0;
#endif
   }
   // Variable set but not to 1 : compat mode
   return 0;
}


ClassImp(TCastorFile)

//______________________________________________________________________________
TCastorFile::TCastorFile(const char *url, Option_t *option, const char *ftitle,
                         Int_t compress, Int_t netopt)
   : TNetFile(url, ftitle, compress, kFALSE)
{
   // Create a TCastorFile. A TCastorFile is like a normal TNetFile except
   // that it obtains the remote node (disk server) via the CASTOR API, once
   // the disk server and the local file path are determined, the file will
   // be accessed via the rootd daemon. File names have to be specified like:
   //    castor:/castor/cern.ch/user/r/rdm/bla.root.
   // The other arguments are the same as for TNetFile and TFile.

   fIsCastor  = kFALSE;
   fWrittenTo = kFALSE;

   // file is always created by stage_out_hsm() and therefore
   // exists when opened by rootd
   TString opt = option;
   opt.ToUpper();
   if (opt == "NEW" || opt == "CREATE")
      opt = "RECREATE";

   Create(url, opt, netopt);
}

//______________________________________________________________________________
void TCastorFile::FindServerAndPath()
{
   // Find the CASTOR disk server and internal file path.
   // The result of this search can be checked via fIsCastor.

   if (!UseCastor2API()) {

      struct stgcat_entry *stcp_output = 0;

      if (rfio_HsmIf_IsHsmFile(fUrl.GetFile()) == RFIO_HSM_CNS) {
         // This is a CASTOR file
         int flags = O_RDONLY;
         struct Cns_filestat st;
         int rc;
         char stageoutbuf[1025];
         char stageerrbuf[1025];

         // Check with internal stage limits - preventing overflow afterwards
         if (strlen(fUrl.GetFile()) > STAGE_MAX_HSMLENGTH) {
            serrno = ENAMETOOLONG;
            Error("FindServerAndPath", "can't open %s, error %d (%s)", fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         // Prepare the flags
         if (fOption == "CREATE" || fOption == "RECREATE" || fOption == "UPDATE")
            flags |= O_RDWR;
         if (fOption == "CREATE" || fOption == "RECREATE")
            flags |= O_CREAT | O_TRUNC;

         // Check if an existing file is going to be updated
         memset(&st, 0, sizeof(st));
         rc = Cns_stat(fUrl.GetFile(), &st);

         // Make sure that filesize is 0 if file doesn't exist
         // or that we will create (stage_out) if O_TRUNC.
         if (rc == -1 || ((flags & O_TRUNC) != 0)) st.filesize = 0;

         // Makes sure stage api does not write automatically to stdout/stderr
         if (stage_setoutbuf(stageoutbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_setoutbuf, error %d (%s)", fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }
         if (stage_seterrbuf(stageerrbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_seterrbuf, error %d (%s)", fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         struct stgcat_entry stcp_input;
         int nstcp_output;

         memset(&stcp_input, 0, sizeof(struct stgcat_entry));
         strcpy(stcp_input.u1.h.xfile, fUrl.GetFile());
         if (flags == O_RDONLY || st.filesize > 0) {
            // Do a recall
            if (stage_in_hsm((u_signed64) 0,          // Ebusy is possible...
                             (int) flags,             // open flags
                             (char *) 0,              // hostname
                             (char *) 0,              // pooluser
                             (int) 1,                 // nstcp_input
                             (struct stgcat_entry *) &stcp_input, // stcp_input
                             (int *) &nstcp_output,   // nstcp_output
                             (struct stgcat_entry **) &stcp_output, // stcp_output
                             (int) 0,                   // nstpp_input
                             (struct stgpath_entry *) 0 // stpp_input
                ) != 0) {
               Error("FindServerAndPath", "can't open %s, stage_in_hsm error %d (%s)",
                     fUrl.GetFile(), serrno, sstrerror(serrno));
               return;
            }
         } else {
            // Do a creation
            if (stage_out_hsm((u_signed64) 0,          // Ebusy is possible...
                              (int) flags,             // open flags
                              (mode_t) 0666,           // open mode (c.f. also umask)
                              // Note: This is S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH, c.f. fopen(2)
                              (char *) 0,              // hostname
                              (char *) 0,              // pooluser
                              (int) 1,                 // nstcp_input
                              (struct stgcat_entry *) &stcp_input, // stcp_input
                              (int *) &nstcp_output,   // nstcp_output
                              (struct stgcat_entry **) &stcp_output, // stcp_output
                              (int) 0,                       // nstpp_input
                              (struct stgpath_entry *) 0     // stpp_input
                ) != 0) {
               Error("FindServerAndPath", "can't open %s, stage_out_hsm error %d (%s)",
                     fUrl.GetFile(), serrno, sstrerror(serrno));
               return;
            }
         }
         if ((nstcp_output != 1) || (stcp_output == 0) ||
             (*(stcp_output->ipath) == '\0')) {
            // Impossible
            serrno = SEINTERNAL;
            if (stcp_output != 0) free(stcp_output);
            Error("FindServerAndPath", "can't open %s, error %d (%s)",
                  fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         // Parse orig string to get disk server host
         char *filename;
         char *realhost = 0;
         rfio_parse(stcp_output->ipath, &realhost, &filename);
         if (realhost == 0) {
            serrno = SEINTERNAL;
            Error("FindServerAndPath", "can't open %s, get disk server hostname from %s error %d (%s)",
                  fUrl.GetFile(), stcp_output->ipath, errno, sstrerror(serrno));
            free(stcp_output);
            return;
         }
         // Save real host and internal path
         fDiskServer = realhost;
         if (filename[0] != '/') {
            // Make file 'local' to the host
            fInternalPath  = "/";
            fInternalPath += filename;
         } else
            fInternalPath = filename;

         if (st.filesize == 0) {
            // Will force notification to stage when the file is closed
            fWrittenTo = kTRUE;
         }
      }

      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod; for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      // (it can be changed by a proper directive in $HOME/.rootauthrc)
      TString r;
      TString fqdn;
      TInetAddress addr = gSystem->GetHostByName(fDiskServer);
      if (addr.IsValid()) {
         fqdn = addr.GetHostName();
         if (fqdn == "UnNamedHost")
            fqdn = addr.GetHostAddress();
         if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
            r = "rootug://";
         else
            r = "root://";
      } else
         r = "root://";

      // Update fUrl with new path
      r += fDiskServer + "/";
      r += fInternalPath;
      TUrl rurl(r);
      fUrl = rurl;

      // Now ipath is not null and contains the real internal path on the disk
      // server 'host', e.g. it is fDiskServer:fInternalPath
      fInternalPath = stcp_output->ipath;
      free(stcp_output);

   } else {

#ifdef R__CASTOR2
      // We use the new stager API
      int flags = O_RDONLY;
      int rc;
      struct stage_io_fileresp *response = 0;
      char *requestId = 0, *url = 0;
      char stageerrbuf[1025];

      // Prepare the flags
      if (fOption == "CREATE" || fOption == "RECREATE" || fOption == "UPDATE")
         flags |= O_RDWR;
      if (fOption == "CREATE" || fOption == "RECREATE")
         flags |= O_CREAT | O_TRUNC;

      stage_seterrbuf(stageerrbuf, 1024);

      rc = stage_open(0,
                      MOVER_PROTOCOL_ROOT,
                      (fUrl.GetFile()),
                      flags,
                      (mode_t) 0666,
                      0,
                      &response,
                      &requestId,
                      0);

      if (rc != 0) {
         Error("FindServerAndPath", "stage_open failed: %s (%s)",
               sstrerror(serrno), stageerrbuf);
         if (response) free(response);
         if (requestId) free(requestId);
         return;
      }

      if (response == 0) {
         Error("FindServerAndPath", "response was null for %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         if (requestId) free(requestId);
         return;
      }

      if (response->errorCode != 0) {
         serrno = response->errorCode;
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      url = stage_geturl(response);

      if (url == 0) {
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fUrl.GetFile(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      TUrl rurl(url);
      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod; for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      // (it can be changed by a proper directive in $HOME/.rootauthrc)
      TString p;
      TString fqdn = rurl.GetHostFQDN();
      if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
         p = "rootug";
      else
         p = "root";

      // Update protocol and fUrl
      rurl.SetProtocol(p);
      fUrl = rurl;

      if (response) free(response);
      if (url) free(url);
      if (requestId) free(requestId);
#endif

   }

   fIsCastor = kTRUE;
}

//______________________________________________________________________________
Int_t TCastorFile::SysClose(Int_t fd)
{
   // Close currently open file.

   Int_t r = TNetFile::SysClose(fd);

   if (!UseCastor2API()) {
      if (fIsCastor && fWrittenTo) {
         // CASTOR file was created or modified
         rfio_HsmIf_reqtoput((char *)fInternalPath.Data());
         fWrittenTo = kFALSE;
      }
   }

   return r;
}

//______________________________________________________________________________
Bool_t TCastorFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range to remote file via rootd daemon.
   // Returns kTRUE in case of error.

   if (TNetFile::WriteBuffer(buf, len))
      return kTRUE;

   if (!UseCastor2API()) {
      if (fIsCastor && !fWrittenTo && len > 0) {
         stage_hsm_t hsmfile;

         // Change status of file in stage catalog from STAGED to STAGEOUT
         memset(&hsmfile, 0, sizeof(hsmfile));
         strcpy(hsmfile.upath, fInternalPath);
         if (stage_updc_filchg(0, &hsmfile) < 0) {
            Error("WriteBuffer", "error calling stage_updc_filchg");
            return kTRUE;
         }
         fWrittenTo = kTRUE;
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
void TCastorFile::ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                                Int_t tcpwindowsize, Bool_t forceOpen,
                                Bool_t forceRead)
{
   // Connect to remote rootd server on CASTOR disk server.

   FindServerAndPath();

   // Continue only if successful
   if (fIsCastor) {
      TNetFile::ConnectServer(stat, kind, netopt, tcpwindowsize, forceOpen, forceRead);
   } else {
      // Failure: fill these to signal it to TNetFile
      *stat = kErrFileOpen;
      *kind = kROOTD_ERR;
   }
}
