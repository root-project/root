// @(#)root/net:$Name:  $:$Id: TCastorFile.cxx,v 1.2 2003/09/23 15:26:55 rdm Exp $
// Author: Fons Rademakers + Jean-Damien Durand  17/09/2003

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
//////////////////////////////////////////////////////////////////////////

#include "TCastorFile.h"

#include <stdlib.h>
#include <errno.h>

#ifdef _WIN32
#include <WinDef.h>
#include <WinSock2.h>
#endif

#define RFIO_KERNEL           // Get access to extra symbols in the headers
#include <stage_api.h>        // Dial with CASTOR stager
#include <rfio_api.h>         // Util routines from rfio
#include <Cns_api.h>          // Dial with CASTOR Name Server


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

   Create(url, option, netopt);
}

//______________________________________________________________________________
void TCastorFile::FindServerAndPath()
{
   // Find the CASTOR disk server and internal file path.

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

   // Update fUrl with new path
   TString r = "root://";
   r += fDiskServer + "/";
   r += fInternalPath;
   TUrl rurl(r);
   fUrl = rurl;

   // Now ipath is not null and contains the real internal path on the disk
   // server 'host', e.g. it is fDiskServer:fInternalPath
   fInternalPath = stcp_output->ipath;
   free(stcp_output);

   fIsCastor = kTRUE;
}

//______________________________________________________________________________
Int_t TCastorFile::SysClose(Int_t fd)
{
   // Close currently open file.

   Int_t r = TNetFile::SysClose(fd);

   if (fIsCastor && fWrittenTo) {
      // CASTOR file was created or modified
      rfio_HsmIf_reqtoput((char *)fInternalPath.Data());
      fWrittenTo = kFALSE;
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

   return kTRUE;
}

//______________________________________________________________________________
void TCastorFile::ConnectServer(Int_t *stat, EMessageTypes *kind, Int_t netopt,
                                Int_t tcpwindowsize, Bool_t forceOpen,
                                Bool_t forceRead)
{
   // Connect to remote rootd server on CASTOR disk server.

   FindServerAndPath();

   TNetFile::ConnectServer(stat, kind, netopt, tcpwindowsize, forceOpen, forceRead);
}
