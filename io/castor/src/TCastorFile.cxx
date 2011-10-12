// @(#)root/castor:$Id$
// Author: Fons Rademakers + Jean-Damien Durand 17/09/2003 + Ben Couturier 31/05/2005
// + Giulia Taurelli 26/04/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
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
// If Castor 2.1 is used the file names can also be specified           //
// in the following ways:                                               //
//                                                                      //
//  castor://stager_host:stager_port/?path=/castor/cern.ch/user/        //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
//  castor://stager_host/?path=/castor/cern.ch/user/                    //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
//  castor:///castor?path=/castor/cern.ch/user/                         //
//    r/rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION   //
//                                                                      //
// path is mandatory as parameter but all the other ones are optional.  //
//                                                                      //
// Use "&rootAuth=<auth_prot_code>" in the option field to force the    //
// specified authentication protocol when contacting the server, e.g.   //
//                                                                      //
//  castor:///castor?path=/castor/cern.ch/user/r/rdm/bla.root           //
//    &svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION&rootAuth=3      //
//                                                                      //
// will try first the globus/GSI protocol; available protocols are      //
//  0: passwd, 1: srp, 2: krb5, 3: globus, 4: ssh, 5 uidgid             //
// The defaul is taken from the env ROOTCASTORAUTH.                     //
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
#include <Cglobals.h>
#include <rfio_constants.h>

#define RFIO_USE_CASTOR_V2 "RFIO_USE_CASTOR_V2"
#define RFIO_HSM_BASETYPE  0x0
#define RFIO_HSM_CNS       RFIO_HSM_BASETYPE+1

extern "C" { int rfio_HsmIf_reqtoput (char *); }
extern "C" { int DLL_DECL rfio_parse(char *, char **, char **); }
extern "C" { int rfio_HsmIf_IsHsmFile (const char *); }
extern "C" { char *getconfent(char *, char *, int); }

#ifdef R__CASTOR2
extern int tStageHostKey;
extern int tStagePortKey;
extern int tSvcClassKey;
extern int tCastorVersionKey;
extern "C" { int DLL_DECL use_castor2_api(); }

//______________________________________________________________________________
static int UseCastor2API()
{
   // Function that checks whether we should use the old or new stager API.

   int version = use_castor2_api();
   return version;
}

#else

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
      // Variable set to yes or 1 but old CASTOR 1: compat mode + warning
      static int once = 0;
      if (!once) {
         ::Warning("UseCastor2API", "asked to use CASTOR 2, but linked with CASTOR 1");
         once = 1;
      }
      return 0;
   }
   // Variable set but not to 1 : compat mode
   return 0;
}

#endif

//______________________________________________________________________________
static const char *GetAuthProto(TString &url)
{
   // Determine the authentication protocol to be tried first from the url
   // string or from defaults. The auth option, if any, is removed from 'url'.

   const Int_t rootNumSec = 6;
   const char *protoSec[rootNumSec] = {"rootup", "roots", "rootk",
                                       "rootg", "rooth", "rootug" };
   TString p = url;
   Int_t ii = p.Index("&rootAuth=");
   if (ii != kNPOS) {
      Int_t jj = p.Index("&", ii+1);
      if (jj != kNPOS)
         p.Remove(jj);
      p.Remove(0,ii);
      url.ReplaceAll(p, "");
      p.ReplaceAll("&rootAuth=", "");
   }
   if (p.Length() <= 0)
      // Use defaults
      p = getenv("ROOTCASTORAUTH");

   Int_t sec = -1;
   if (p.Length() > 0 && p.IsDigit()) {
      sec = p.Atoi();
      if (sec < 0 || sec > (rootNumSec - 1))
         sec = -1;
   }

   // Done
   return ((sec > -1 && sec < rootNumSec) ? protoSec[sec] : "root");
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

   // Extract the authentication info, if any; removing it from the options
   TString u(url);
   fAuthProto = GetAuthProto(u);
   fUrl.SetUrl(u);

   if (gDebug > 0)
      Info("TCastorFile","fAuthProto = %s, u: %s", fAuthProto.Data(), u.Data());

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

   // just call rfio_parse and no extra parsing is added here to that

   TString castorturl;
   char *host=0;
   char *name=0;

   // to be able to use the turl starting with  castor:
   if (!strcmp(fUrl.GetProtocol(),"castor"))
      castorturl = Form("%s://%s", "rfio", fUrl.GetFile());
   else
      castorturl = Form("%s://%s", fUrl.GetProtocol(), fUrl.GetFile());
   if (strlen(fUrl.GetOptions()) > 0)
      castorturl += Form("?%s", fUrl.GetOptions());

   // the complete turl in fname
   TString fname = castorturl; // for compatibility with rfio_parse interface
   if (::rfio_parse((char *)fname.Data(), &host, &name)>=0) {
      castorturl = Form("%s",(!name || !strstr(name,"/castor"))?fname.Data():name);
      fname = castorturl.Data();

   } else {
      Error("FindServerAndPath", "error parsing %s", fUrl.GetUrl());
      return;
   }

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
         if (rc == -1 || ((flags & O_TRUNC) != 0))
            st.filesize = 0;

         // Makes sure stage api does not write automatically to stdout/stderr
         if (stage_setoutbuf(stageoutbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_setoutbuf, error %d (%s)",
                  fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }
         if (stage_seterrbuf(stageerrbuf, 1024) != 0) {
            Error("FindServerAndPath", "can't open %s, stage_seterrbuf, error %d (%s)",
                  fUrl.GetFile(), serrno, sstrerror(serrno));
            return;
         }

         struct stgcat_entry stcp_input;
         int nstcp_output;

         memset(&stcp_input, 0, sizeof(struct stgcat_entry));
         strlcpy(stcp_input.u1.h.xfile, fUrl.GetFile(), sizeof(stcp_input.u1.h.xfile));
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
                             (int) 0,                 // nstpp_input
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
         } else {
            fInternalPath = filename;
         }

         if (st.filesize == 0) {
            // Will force notification to stage when the file is closed
            fWrittenTo = kTRUE;
         }
      }

      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod, unless there is a specific
      // request (from options or envs); for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      TString r = fAuthProto;
      if (fAuthProto == "root") {
         TString fqdn;
         TInetAddress addr = gSystem->GetHostByName(fDiskServer);
         if (addr.IsValid()) {
            fqdn = addr.GetHostName();
            if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
               r = "rootug://";
            else
               r = "root://";
         } else
            r = "root://";
      } else {
         // Fix the format
         r += "://";
      }

      // Update fUrl with new path
      r += fDiskServer + "/";
      r += fInternalPath;
      TUrl rurl(r);
      fUrl = rurl;

      if (gDebug > 0)
         Info("FindServerAndPath"," fDiskServer: %s, r: %s", fDiskServer.Data(), r.Data());

      // Now ipath is not null and contains the real internal path on the disk
      // server 'host', e.g. it is fDiskServer:fInternalPath
      fInternalPath = stcp_output==0?0:stcp_output->ipath;
      if (stcp_output)
         free(stcp_output);
   } else {

#ifdef R__CASTOR2

      // We use the new stager API
      // I use fname which has the Turl already parsed correctly

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

      int* auxVal;
      char ** auxPoint;
      struct stage_options opts;
      opts.stage_host=0;
      opts.stage_port=0;
      opts.service_class=0;
      opts.stage_version=0;

      void *ptrPoint = &auxPoint;
      void *ptrVal = &auxVal;
      int ret=Cglobals_get(& tStageHostKey, (void**)ptrPoint,sizeof(void*));
      if(ret==0){
         opts.stage_host=*auxPoint;
      }
      ret=Cglobals_get(& tStagePortKey, (void**)ptrVal,sizeof(int));
      if(ret==0){
         opts.stage_port=*auxVal;
      }
      opts.stage_version=2;
      ret=Cglobals_get(& tSvcClassKey, (void**)ptrPoint,sizeof(void*));
      if (ret==0){
         opts.service_class=*auxPoint;
      }

      // in stage_open I use the fname which is the result of the rfio_parsing
      rc = stage_open(0,
                      MOVER_PROTOCOL_ROOT,
                      (char *)fname.Data(),
                      flags,
                      (mode_t) 0666,
                      0,
                      &response,
                      &requestId,
                      &opts); // global values used as options

      if (rc != 0) {
         Error("FindServerAndPath", "stage_open failed: %s (%s)",
               sstrerror(serrno), stageerrbuf);
         if (response) free(response);
         if (requestId) free(requestId);
         return;
      }

      if (response == 0) {
         Error("FindServerAndPath", "response was null for %s (Request %s) %d/%s",
               fname.Data(), requestId,
               serrno, sstrerror(serrno));
         if (requestId) free(requestId);
         return;
      }

      if (response->errorCode != 0) {
         serrno = response->errorCode;
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fname.Data(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      url = stage_geturl(response);

      if (url == 0) {
         Error("FindServerAndPath", "error getting file %s (Request %s) %d/%s",
               fname.Data(), requestId,
               serrno, sstrerror(serrno));
         free(response);
         if (requestId) free(requestId);
         return;
      }

      TUrl rurl(url);
      // Set the protocol prefix for TNetFile.
      // For the cern.ch domain we set the default authentication
      // method to UidGid, i.e. as for rfiod, unless there is a specific
      // request (from options or envs); for this we need
      // the full FQDN or address in "nnn.mmm.iii.jjj" form
      TString p = fAuthProto;
      if (fAuthProto == "root") {
         TString fqdn = rurl.GetHostFQDN();
         if (fqdn.EndsWith(".cern.ch") || fqdn.BeginsWith("137.138."))
            fAuthProto = "rootug";
      }

      // Update protocol and fUrl
      rurl.SetProtocol(fAuthProto);
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
#ifndef R__CASTOR2
         // CASTOR file was created or modified
         rfio_HsmIf_reqtoput((char *)fInternalPath.Data());
#endif
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
         hsmfile.upath = StrDup(fInternalPath);
         if (stage_updc_filchg(0, &hsmfile) < 0) {
            Error("WriteBuffer", "error calling stage_updc_filchg");
            delete [] hsmfile.upath;
            return kTRUE;
         }
         delete [] hsmfile.upath;
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
