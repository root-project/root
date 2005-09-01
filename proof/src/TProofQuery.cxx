// @(#)root/proof:$Name:  $:$Id: TProofQuery.cxx,v 1.2 2005/08/30 10:37:56 rdm Exp $
// Author: G Ganis Aug 2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofQuery                                                          //
//                                                                      //
// A class describing PROOF queries.                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef WIN32
   #include <io.h>
   typedef long off_t;
#else
   #include <unistd.h>
#endif

#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>

#if (defined(__FreeBSD__) && (__FreeBSD__ < 4)) || defined(__OpenBSD__) || \
    (defined(__APPLE__) && (!defined(MAC_OS_X_VERSION_10_3) || \
     (MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_10_3)))
#include <sys/file.h>
#define lockf(fd, op, sz)   flock((fd), (op))
#ifndef F_LOCK
#define F_LOCK             (LOCK_EX | LOCK_NB)
#endif
#ifndef F_ULOCK
#define F_ULOCK             LOCK_UN
#endif
#endif

#include "TError.h"
#include "TProofQuery.h"
#include "TSystem.h"
#include "TDSet.h"


ClassImp(TProofQuery)

//______________________________________________________________________________
TProofQuery::TProofQuery(Int_t seqnum, Int_t startlog,
                         Long64_t entries, Long64_t first, TDSet *dset,
                         const char *selec, const char *par,
                         const char *file)
{
   // Main constructor.

   fSeqNum = seqnum;
   fStatus = kRunning;
   fStart.Set();
   fEnd.Set(fStart.Convert()-1);
   fStartLog = startlog;
   fEndLog = startlog;
   fEntries = entries;
   fFirst = first;
   fNFiles = (dset) ? dset->GetListOfElements()->GetSize() : 0;
   fTotalEntries = 0;
   fSelecName = (selec && (strlen(selec) > 0)) ? selec : "-";
   fParList = (par && (strlen(par) > 0)) ? par : "-";
   fResultFile = "-";
   if (file)
      BackupToFile(file);
}

//______________________________________________________________________________
Int_t TProofQuery::BackupToFile(const char *file)
{
   // Backup the content of this entry to file. Used on the master to record
   // the status and relevant info about queries.

   Int_t rc = 0;

   if (!file)
      return -1;

   // Open the file in write mode, truncating if it exists already
   Int_t fid = -1;
   if (gSystem->AccessPathName(file))
      fid = open(file, O_CREAT|O_WRONLY, 0644);
   else
      fid = open(file, O_TRUNC|O_WRONLY);
   if (fid == -1) {
      SysError("BackupToFile", "cannot open backup file %s", file);
      return -1;
   }

   // lock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fid, F_LOCK, (off_t) 1) == -1) {
      SysError("BackupToFile", "error locking %s", file);
      close(fid);
      return -1;
   }
#endif

   // Prepare the output buffer
   TString buf = Form("%d %d %d %d %d %d %lld %lld %d %lld %s %s %s",
                      fSeqNum, fStatus,
                      fStart.Convert(), fEnd.Convert(),
                      fStartLog, fEndLog, fEntries, fFirst, fNFiles,
                      fTotalEntries, fSelecName.Data(), fParList.Data(),
                      fResultFile.Data());
   // Write out
   Int_t r = buf.Length();
   const char *p = buf.Data();
   while (r) {
      Int_t w = write(fid, p, r);
      if (w < 0) {
         SysError("BackupToFile", "error writing to %s", file);
         rc = -1;
         break;
      }
      r -= w;
      p += w;
   }

   // Unlock the file
#if !defined(R__WIN32) && !defined(R__WINGCC)
   if (lockf(fid, F_ULOCK, (off_t)1) == -1) {
      SysError("BackupToFile", "error unlocking %s", file);
      close(fid);
      return -1;
   }
#endif

   // Close file
   close(fid);

   // We are done
   return rc;
}

//______________________________________________________________________________
Int_t TProofQuery::FetchFromFile(const char *file)
{
   // Backup the content of this entry to file. Used on the master to record
   // the status and relevant info about queries.

   Int_t rc = 0;

   if (!file || gSystem->AccessPathName(file,kReadPermission)) {
      SysError("FetchFromFile", "file not exists or inaccessible %s",
               (file ? file : ""));
      return -1;
   }

   // Open the file in read mode
   Int_t fid = open(file, O_RDONLY);
   if (fid == -1) {
      SysError("FetchFromFile", "cannot open file %s", file);
      return -1;
   }

   // Read buffer line
   char buf[2048];
   Int_t nr = 0;
   char *p = buf;
   do {
      while ((nr = read(fid, p, sizeof(buf))) < 0 && TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();
      p = (nr > 0) ? (p + nr) : p;
   } while (nr > 0);

   if (strlen(buf) > 0) {

      // Get rid of trailing '\n'
      if (buf[strlen(buf)-1] == '\n')
         buf[strlen(buf)-1] = 0;

      // Parse buffer
      Int_t np = 0, stat = 0;
      UInt_t start = 0, end = 0;
      char selnam[2048] = {0}, parlist[2048] = {0}, resfile[2048] = {0};

      if ((np = sscanf(buf,"%d %d %d %d %d %d %lld %lld %d %lld %s %s %s",
                       &fSeqNum, &stat, &start, &end,
                       &fStartLog, &fEndLog, &fEntries, &fFirst,
                       &fNFiles, &fTotalEntries,
                       selnam, parlist, resfile)) < 11) {
         Warning("FetchFromFile", "parsed only %d elements (instead of 11)", np);
         Warning("FetchFromFile", "buffer maybe corrupted: %s", buf);
      }

      // Fill members
      fStatus = (EQueryStatus) stat;
      fStart.Set(start);
      fEnd.Set(end);
      fSelecName = selnam;
      fParList = parlist;
      fResultFile = resfile;
   }

   // Close file
   close(fid);

   // We are done
   return rc;
}

//______________________________________________________________________________
void TProofQuery::SetDone(EQueryStatus status,
                          Int_t endlog, const char *tmpfile, const char *file)
{
   // Set query in done state (completed, stopped, aborted).

   status = (status < kAborted || status > kArchived) ? kAborted : status;

   fEnd.Set();
   fEndLog = (endlog > fStartLog) ? endlog : fStartLog;
   fStatus = status;
   if (fStatus >= kStopped) {
      if (tmpfile && (strlen(tmpfile) > 0))
         fResultFile = tmpfile;
      if (file)
         BackupToFile(file);
   }
}

//______________________________________________________________________________
void TProofQuery::SetArchived(const char *resfile, const char *file)
{
   // Set (or update) query in archived state.

   if (IsDone()) {
      fStatus = kArchived;
      if (resfile && (strlen(resfile) > 0))
         fResultFile = resfile;
      if (file)
         BackupToFile(file);
   }
}

//______________________________________________________________________________
void TProofQuery::Print(Option_t *opt) const
{
   // Print query content. Use opt = "F" for a full listing.

   // Attention, must match EQueryStatus
   const char *qst[] = {"aborted", "running", "stopped", "completed", "archived"};

   // Status label
   Int_t st = (fStatus > 0 && fStatus <= kArchived) ? fStatus : 0;

   // Range label
   Long64_t last = (fEntries > -1) ? fFirst+fEntries : -1;

   // Option
   Bool_t full = (!strncasecmp(opt,"F",1));

   // Print separator if full dump
   if (full) Printf("+++");

   TString range;
   if (!full)
      range = (last > -1) ? Form("(event range: %lld - %lld)", fFirst, last) : "";

   // Print header
   Printf("+++ #:%3d  status:    %9s    selector: %s %s",
          fSeqNum, qst[st], fSelecName.Data(), range.Data());

   // We are done, if not full dump
   if (!full) return;

   // Number of events processed, rate
   Int_t elapsed = fEnd.Convert() - fStart.Convert();
   Float_t rate = 0.0;
   if (fEntries > -1 && elapsed > 0)
      rate = (Float_t)fEntries / elapsed ;
   Printf("+++        processed: %d events (rate: %.1f evts/sec)", fEntries, rate);

   // Package information
   Printf("+++        packages:  %s", fParList.Data());

   // Time information
   Printf("+++        started:   %s (elapsed time: %d sec)",
          fStart.AsString(), (fEnd.Convert()-fStart.Convert()));

   // Result information
   TString res = fResultFile;
   if (fStatus != kArchived) {
      Int_t dq = res.Index("queries");
      if (dq > -1) {
         res.Remove(0,res.Index("queries"));
         res.Insert(0,"<PROOF_SandBox>/");
      }
      if (res.BeginsWith("-")) {
         res = (fStatus == kAborted) ? "not available" : "sent to client";
      }
   }
   Printf("+++        results:   %s", res.Data());
}
