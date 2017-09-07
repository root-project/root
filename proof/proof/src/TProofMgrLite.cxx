// @(#)root/proofx:$Id$
// Author: Gerardo Ganis Apr 2008

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TProofMgrLite
\ingroup proofkernel

Basic TProofMgr functionality implementation in the case of Lite session.

*/

#include <errno.h>
#ifdef WIN32
#include <io.h>
#endif

#include "TProofMgrLite.h"

#include "Riostream.h"
#include "TEnv.h"
#include "TError.h"
#include "TObjString.h"
#include "TProofLite.h"
#include "TProofLog.h"
#include "TROOT.h"
#include "TRegexp.h"
#include "TSortedList.h"

ClassImp(TProofMgrLite);

////////////////////////////////////////////////////////////////////////////////
/// Create a PROOF manager for the Lite environment.

TProofMgrLite::TProofMgrLite(const char *url, Int_t dbg, const char *alias)
          : TProofMgr(url, dbg, alias)
{
   // Set the correct servert type
   fServType = kProofLite;
}

////////////////////////////////////////////////////////////////////////////////
/// Create a new session

TProof *TProofMgrLite::CreateSession(const char *cfg,
                                     const char *, Int_t loglevel)
{
   TString c(fUrl.GetOptions());
   if (!c.Contains("workers=") && cfg && strstr(cfg, "workers=")) c = cfg;
   Int_t nwrk = TProofLite::GetNumberOfWorkers(c);
   if (nwrk == 0) return (TProof *)0;

   // Check if we have already a running session
   if (gProof && gProof->IsLite()) {
      if (gProof->IsValid()) {
         if (nwrk > 0 && gProof->GetParallel() != nwrk) {
            delete gProof;
            gProof = 0;
         } else {
            // We have already a running session
            return gProof;
         }
      } else {
         // Remove existing instance
         delete gProof;
         gProof = 0;
      }
   }

   // Create the instance
   TString u("lite");
   if (strlen(fUrl.GetOptions()) > 0) u.Form("lite/?%s", fUrl.GetOptions());
   TProof *p = new TProofLite(u, cfg, 0, loglevel, 0, this);

   if (p && p->IsValid()) {

      // Save record about this session
      Int_t ns = 1;
      if (fSessions) {
         // To avoid ambiguities in case of removal of some elements
         if (fSessions->Last())
            ns = ((TProofDesc *)(fSessions->Last()))->GetLocalId() + 1;
      } else {
         // Create the list
         fSessions = new TList;
      }

      // Create the description class
      Int_t st = (p->IsIdle()) ? TProofDesc::kIdle : TProofDesc::kRunning ;
      TProofDesc *d =
         new TProofDesc(p->GetName(), p->GetTitle(), p->GetUrl(),
                               ns, p->GetSessionID(), st, p);
      fSessions->Add(d);

   } else {
      // Session creation failed
      Error("CreateSession", "creating PROOF session");
      SafeDelete(p);
   }

   // We are done
   return p;
}

////////////////////////////////////////////////////////////////////////////////
/// Get logs or log tails from last session associated with this manager
/// instance.
/// The arguments allow to specify a session different from the last one:
///      isess   specifies a position relative to the last one, i.e. 1
///              for the next to last session; the absolute value is taken
///              so -1 and 1 are equivalent.
///      stag    specifies the unique tag of the wanted session
/// The special value stag = "NR" allows to just initialize the TProofLog
/// object w/o retrieving the files; this may be useful when the number
/// of workers is large and only a subset of logs is required.
/// If 'stag' is specified 'isess' is ignored (unless stag = "NR").
/// If 'pattern' is specified only the lines containing it are retrieved
/// (remote grep functionality); to filter out a pattern 'pat' use
/// pattern = "-v pat".
/// Returns a TProofLog object (to be deleted by the caller) on success,
/// 0 if something wrong happened.

TProofLog *TProofMgrLite::GetSessionLogs(Int_t isess, const char *stag,
                                         const char *pattern, Bool_t)
{
   TProofLog *pl = 0;

   // The absolute value of isess counts
   isess = (isess < 0) ? -isess : isess;

   // Special option in stag
   bool retrieve = 1;
   TString tag(stag);
   if (tag == "NR") {
      retrieve = 0;
      tag = "";
   }

   // The working dir
   TString sandbox(gSystem->WorkingDirectory());
   sandbox.ReplaceAll(gSystem->HomeDirectory(),"");
   sandbox.ReplaceAll("/","-");
   sandbox.Replace(0,1,"/",1);
   if (strlen(gEnv->GetValue("ProofLite.Sandbox", "")) > 0) {
      sandbox.Insert(0, gEnv->GetValue("ProofLite.Sandbox", ""));
   } else if (strlen(gEnv->GetValue("Proof.Sandbox", "")) > 0) {
      sandbox.Insert(0, gEnv->GetValue("Proof.Sandbox", ""));
   } else {
      TString sb;
      sb.Form("~/%s", kPROOF_WorkDir);
      sandbox.Insert(0, sb.Data());
   }
   gSystem->ExpandPathName(sandbox);

   TString sessiondir;
   if (tag.Length() > 0) {
      sessiondir.Form("%s/session-%s", sandbox.Data(), tag.Data());
      if (gSystem->AccessPathName(sessiondir, kReadPermission)) {
         Error("GetSessionLogs", "information for session '%s' not available", tag.Data());
         return (TProofLog *)0;
      }
   } else {
      // Get the list of available dirs
      TSortedList *olddirs = new TSortedList(kFALSE);
      void *dirp = gSystem->OpenDirectory(sandbox);
      if (dirp) {
         const char *e = 0;
         while ((e = gSystem->GetDirEntry(dirp))) {
            if (!strncmp(e, "session-", 8)) {
               TString d(e);
               Int_t i = d.Last('-');
               if (i != kNPOS) d.Remove(i);
               i = d.Last('-');
               if (i != kNPOS) d.Remove(0,i+1);
               TString path = Form("%s/%s", sandbox.Data(), e);
               olddirs->Add(new TNamed(d, path));
            }
         }
         gSystem->FreeDirectory(dirp);
      }

      // Check isess
      if (isess > olddirs->GetSize() - 1) {
         Warning("GetSessionLogs",
                 "session index out of range (%d): take oldest available session", isess);
         isess = olddirs->GetSize() - 1;
      }

      // Locate the session dir
      Int_t isx = isess;
      TNamed *n = (TNamed *) olddirs->First();
      while (isx-- > 0) {
         olddirs->Remove(n);
         delete n;
         n = (TNamed *) olddirs->First();
      }
      if (!n) {
         Error("GetSessionLogs", "cannot locate session dir for index '%d' under '%s':"
                                 " cannot continue!", isess, sandbox.Data());
         return (TProofLog *)0;
      }
      sessiondir = n->GetTitle();
      tag = gSystem->BaseName(sessiondir);
      tag.ReplaceAll("session-", "");

      // Cleanup
      olddirs->SetOwner();
      delete olddirs;
   }
   Info("GetSessionLogs", "analysing session dir %s", sessiondir.Data());

   // Create the instance now
   pl = new TProofLog(tag, "", this);

   void *dirp = gSystem->OpenDirectory(sessiondir);
   if (dirp) {
      TSortedList *logs = new TSortedList;
      const char *e = 0;
      while ((e = gSystem->GetDirEntry(dirp))) {
         TString fn(e);
         if (fn.EndsWith(".log") && fn.CountChar('-') > 0) {
            TString ord, url;
            if (fn.BeginsWith("session-")) {
               ord = "-1";
            } else if (fn.BeginsWith("worker-")) {
               ord = fn;
               ord.ReplaceAll("worker-", "");
               Int_t id = ord.First('-');
               if (id != kNPOS) {
                  ord.Remove(id);
               } else if (ord.Contains(".valgrind")) {
                  // Add to the list (special tag for valgrind outputs)
                  ord.ReplaceAll(".valgrind.log","-valgrind");
               } else {
                  // Not a good path
                  ord = "";
               }
               if (!ord.IsNull()) ord.ReplaceAll("0.", "");
            }
            if (!ord.IsNull()) {
               url = Form("%s/%s", sessiondir.Data(), e);
               // Add to the list
               logs->Add(new TNamed(ord, url));
               // Notify
               if (gDebug > 1)
                  Info("GetSessionLogs", "ord: %s, url: %s", ord.Data(), url.Data());
            }
         }
      }
      gSystem->FreeDirectory(dirp);

      TIter nxl(logs);
      TNamed *n = 0;
      while ((n = (TNamed *) nxl())) {
         TString ord = Form("0.%s", n->GetName());
         if (ord == "0.-1") ord = "0";
         // Add to the list
         pl->Add(ord, n->GetTitle());
      }

      // Cleanup
      logs->SetOwner();
      delete logs;
   }

   // Retrieve the default part
   if (pl && retrieve) {
      const char *pat = pattern ? pattern : "-v \"| SvcMsg\"";
      if (pat && strlen(pat) > 0)
         pl->Retrieve("*", TProofLog::kGrep, 0, pat);
      else
         pl->Retrieve();
   }

   // Done
   return pl;
}

////////////////////////////////////////////////////////////////////////////////
/// Read 'len' bytes from offset 'ofs' of the local file 'fin'.
/// Returns a TObjString with the content or 0, in case of failure

TObjString *TProofMgrLite::ReadBuffer(const char *fin, Long64_t ofs, Int_t len)
{
   if (!fin || strlen(fin) <= 0) {
      Error("ReadBuffer", "undefined path!");
      return (TObjString *)0;
   }

   // Open the file
   TString fn = TUrl(fin).GetFile();
   Int_t fd = open(fn.Data(), O_RDONLY);
   if (fd < 0) {
      Error("ReadBuffer", "problems opening file %s", fn.Data());
      return (TObjString *)0;
   }

   // Total size
   off_t start = 0, end = lseek(fd, (off_t) 0, SEEK_END);

   // Set the offset
   if (ofs > 0 && ofs < end) {
      start = lseek(fd, (off_t) ofs, SEEK_SET);
   } else {
      start = lseek(fd, (off_t) 0, SEEK_SET);
   }
   if (len > (end - start + 1) || len <= 0)
      len = end - start + 1;

   TString outbuf;
   const Int_t kMAXBUF = 32768;
   char buf[kMAXBUF];
   Int_t left = len;
   Int_t wanted = (left > kMAXBUF - 1) ? kMAXBUF - 1 : left;
   do {
      while ((len = read(fd, buf, wanted)) < 0 && TSystem::GetErrno() == EINTR)
         TSystem::ResetErrno();

      if (len < 0) {
         Error("ReadBuffer", "error reading file %s", fn.Data());
         close(fd);
         return (TObjString *)0;
      } else if (len > 0) {
         if (len == wanted)
            buf[len-1] = '\n';
         buf[len] = '\0';
         outbuf += buf;
      }

      // Update counters
      left -= len;
      wanted = (left > kMAXBUF - 1) ? kMAXBUF - 1 : left;

   } while (len > 0 && left > 0);

   // Close file
   close(fd);

   // Done
   return new TObjString(outbuf.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Read lines containing 'pattern' in 'file'.
/// Returns a TObjString with the content or 0, in case of failure

TObjString *TProofMgrLite::ReadBuffer(const char *fin, const char *pattern)
{
   // If no pattern, read everything
   if (!pattern || strlen(pattern) <= 0)
      return (TObjString *)0;

   if (!fin || strlen(fin) <= 0) {
      Error("ReadBuffer", "undefined path!");
      return (TObjString *)0;
   }
   TString fn = TUrl(fin).GetFile();

   TString pat(pattern);
   // Check if "-v"
   Bool_t excl = kFALSE;
   if (pat.Contains("-v ")) {
      pat.ReplaceAll("-v ", "");
      excl = kTRUE;
   }
   pat = pat.Strip(TString::kLeading, ' ');
   pat = pat.Strip(TString::kTrailing, ' ');
   pat = pat.Strip(TString::kLeading, '\"');
   pat = pat.Strip(TString::kTrailing, '\"');

   // Use a regular expression
   TRegexp re(pat);

   // Open file with file info
   std::ifstream in;
   in.open(fn.Data());

   TString outbuf;

   // Read the input list of files and add them to the chain
   TString line;
   while(in.good()) {

      // Read next line
      line.ReadLine(in);

      // Keep only lines with pattern
      if ((excl && line.Index(re) != kNPOS) ||
          (!excl && line.Index(re) == kNPOS)) continue;

      // Remove trailing '\n', if any
      if (!line.EndsWith("\n")) line.Append('\n');

      // Add to output
      outbuf += line;
   }
   in.close();

   // Done
   return new TObjString(outbuf.Data());
}
