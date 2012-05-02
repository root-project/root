// @(#)root/proof:$Id$
// Author: G. Ganis Mar 2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQueryResultManager                                                  //
//                                                                      //
// This class manages the query-result area.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <errno.h>
#ifdef WIN32
#   include <io.h>
#endif

#include "TQueryResultManager.h"

#include "TFile.h"
#include "THashList.h"
#include "TKey.h"
#include "TProofQueryResult.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TProof.h"
#include "TProofServ.h"
#include "TRegexp.h"
#include "TSortedList.h"
#include "TSystem.h"
#include "TVirtualProofPlayer.h"

//______________________________________________________________________________
TQueryResultManager::TQueryResultManager(const char *qdir, const char *stag,
                                         const char *sdir,
                                         TProofLockPath *lck, FILE *logfile)
{
   // Constructor

   fQueryDir        = qdir;
   fSessionTag      = stag;
   fSessionDir      = sdir;
   fSeqNum          = 0;
   fDrawQueries     = 0;
   fKeptQueries     = 0;
   fQueries         = new TList;
   fPreviousQueries = 0;
   fLock            = lck;
   fLogFile         = (logfile) ? logfile : stdout;
}

//______________________________________________________________________________
TQueryResultManager::~TQueryResultManager()
{
   // Cleanup. Not really necessary since after this dtor there is no
   // live anyway.

   SafeDelete(fQueries);
   SafeDelete(fPreviousQueries);
}

//______________________________________________________________________________
void TQueryResultManager::AddLogFile(TProofQueryResult *pq)
{
   // Add part of log file concerning TQueryResult pq to its macro
   // container.

   if (!pq)
      return;

   // Make sure everything is written to file
   fflush(fLogFile);

   // Save current position
   off_t lnow = 0;
   if ((lnow = lseek(fileno(fLogFile), (off_t) 0, SEEK_CUR)) < 0) {
      Error("AddLogFile", "problems lseeking current position on log file (errno: %d)", errno);
      return;
   }

   // The range we are interested in
   Int_t start = pq->fStartLog;
   if (start > -1)
      lseek(fileno(fLogFile), (off_t) start, SEEK_SET);

   // Read the lines and add then to the internal container
   const Int_t kMAXBUF = 4096;
   char line[kMAXBUF];
   while (fgets(line, sizeof(line), fLogFile)) {
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      pq->AddLogLine((const char *)line);
   }

   // Restore initial position if partial send
   if (lnow >= 0) lseek(fileno(fLogFile), lnow, SEEK_SET);
}
//______________________________________________________________________________
Int_t TQueryResultManager::CleanupQueriesDir()
{
   // Remove all queries results referring to previous sessions

   Int_t nd = 0;

   // Cleanup previous stuff
   if (fPreviousQueries) {
      fPreviousQueries->Delete();
      SafeDelete(fPreviousQueries);
   }

   // Loop over session dirs
   TString queriesdir = fQueryDir;
   queriesdir = queriesdir.Remove(queriesdir.Index(kPROOF_QueryDir) +
                                  strlen(kPROOF_QueryDir));
   void *dirs = gSystem->OpenDirectory(queriesdir);
   if (dirs) {
      char *sess = 0;
      while ((sess = (char *) gSystem->GetDirEntry(dirs))) {

         // We are interested only in "session-..." subdirs
         if (strlen(sess) < 7 || strncmp(sess,"session",7))
            continue;

         // We do not want this session at this level
         if (strstr(sess, fSessionTag))
            continue;

         // Remove the directory
         TString qdir;
         qdir.Form("%s/%s", queriesdir.Data(), sess);
         PDB(kGlobal, 1)
            Info("RemoveQuery", "removing directory: %s", qdir.Data());
         gSystem->Exec(Form("%s %s", kRM, qdir.Data()));
         nd++;
      }
      // Close directory
      gSystem->FreeDirectory(dirs);
   } else {
      Warning("RemoveQuery", "cannot open queries directory: %s", queriesdir.Data());
   }

   // Done
   return nd;
}

//______________________________________________________________________________
void TQueryResultManager::ScanPreviousQueries(const char *dir)
{
   // Scan the queries directory for the results of previous queries.
   // The headers of the query results found are loaded in fPreviousQueries.
   // The full query result can be retrieved via TProof::Retrieve.

   // Cleanup previous stuff
   if (fPreviousQueries) {
      fPreviousQueries->Delete();
      SafeDelete(fPreviousQueries);
   }

   // Loop over session dirs
   void *dirs = gSystem->OpenDirectory(dir);
   char *sess = 0;
   while ((sess = (char *) gSystem->GetDirEntry(dirs))) {

      // We are interested only in "session-..." subdirs
      if (strlen(sess) < 7 || strncmp(sess,"session",7))
         continue;

      // We do not want this session at this level
      if (strstr(sess, fSessionTag))
         continue;

      // Loop over query dirs
      void *dirq = gSystem->OpenDirectory(Form("%s/%s", dir, sess));
      char *qry = 0;
      while ((qry = (char *) gSystem->GetDirEntry(dirq))) {

         // We are interested only in "n/" subdirs
         if (qry[0] == '.')
            continue;

         // File with the query result
         TString fn = Form("%s/%s/%s/query-result.root", dir, sess, qry);
         TFile *f = TFile::Open(fn);
         if (f) {
            f->ReadKeys();
            TIter nxk(f->GetListOfKeys());
            TKey *k =  0;
            TProofQueryResult *pqr = 0;
            while ((k = (TKey *)nxk())) {
               if (!strcmp(k->GetClassName(), "TProofQueryResult")) {
                  pqr = (TProofQueryResult *) f->Get(k->GetName());
                  if (pqr) {
                     TQueryResult *qr = pqr->CloneInfo();
                     if (qr) {
                        if (!fPreviousQueries)
                           fPreviousQueries = new TList;
                        if (qr->GetStatus() > TQueryResult::kRunning) {
                           fPreviousQueries->Add(qr);
                        } else {
                           // (For the time being) remove a non completed
                           // query if not owned by anybody
                           TProofLockPath *lck = 0;
                           if (LockSession(qr->GetTitle(), &lck) == 0) {
                              RemoveQuery(qr);
                              // Unlock and remove the lock file
                              SafeDelete(lck);
                           }
                        }
                     } else {
                        Warning("ScanPreviousQueries", "unable to clone TProofQueryResult '%s:%s'",
                                                       pqr->GetName(), pqr->GetTitle());
                     }
                  }
               }
            }
            f->Close();
            delete f;
         }
      }
      gSystem->FreeDirectory(dirq);
   }
   gSystem->FreeDirectory(dirs);
}

//______________________________________________________________________________
Int_t TQueryResultManager::ApplyMaxQueries(Int_t mxq)
{
   // Scan the queries directory and remove the oldest ones (and relative dirs,
   // if empty) in such a way only 'mxq' queries are kept.
   // Return 0 on success, -1 in case of problems

   // Nothing to do if mxq is -1.
   if (mxq < 0)
      return 0;

   // We will sort the entries using the creation time
   TSortedList *sl = new TSortedList;
   sl->SetOwner();
   // List with information
   THashList *hl = new THashList;
   hl->SetOwner();

   // Keep track of the queries per session dir
   TList *dl = new TList;
   dl->SetOwner();

   // Loop over session dirs
   TString dir = fQueryDir;
   Int_t idx = dir.Index("session-");
   if (idx != kNPOS)
      dir.Remove(idx);
   void *dirs = gSystem->OpenDirectory(dir);
   char *sess = 0;
   while ((sess = (char *) gSystem->GetDirEntry(dirs))) {

      // We are interested only in "session-..." subdirs
      if (strlen(sess) < 7 || strncmp(sess,"session",7))
         continue;

      // We do not want this session at this level
      if (strstr(sess, fSessionTag))
         continue;

      // Loop over query dirs
      Int_t nq = 0;
      void *dirq = gSystem->OpenDirectory(Form("%s/%s", dir.Data(), sess));
      char *qry = 0;
      while ((qry = (char *) gSystem->GetDirEntry(dirq))) {

         // We are interested only in "n/" subdirs
         if (qry[0] == '.')
            continue;

         // File with the query result
         TString fn = Form("%s/%s/%s/query-result.root", dir.Data(), sess, qry);

         FileStat_t st;
         if (gSystem->GetPathInfo(fn, st)) {
            PDB(kGlobal, 1)
               Info("ApplyMaxQueries","file '%s' cannot be stated: remove it", fn.Data());
            gSystem->Unlink(gSystem->DirName(fn));
            continue;
         }

         // Add the entry in the sorted list
         sl->Add(new TObjString(TString::Format("%ld", st.fMtime)));
         hl->Add(new TNamed((const char*)TString::Format("%ld",st.fMtime), fn.Data()));
         nq++;
      }
      gSystem->FreeDirectory(dirq);

      if (nq > 0)
         dl->Add(new TParameter<Int_t>(TString::Format("%s/%s", dir.Data(), sess), nq));
      else
         // Remove it
         gSystem->Exec(TString::Format("%s -fr %s/%s", kRM, dir.Data(), sess));
   }
   gSystem->FreeDirectory(dirs);

   // Now we apply the quota
   TIter nxq(sl, kIterBackward);
   Int_t nqkept = 0;
   TObjString *os = 0;
   while ((os = (TObjString *)nxq())) {
      if (nqkept < mxq) {
         // Keep this and go to the next
         nqkept++;
      } else {
         // Clean this
         TNamed *nm = dynamic_cast<TNamed *>(hl->FindObject(os->GetName()));
         if (nm) {
            gSystem->Unlink(nm->GetTitle());
            // Update dir counters
            TString tdir(gSystem->DirName(nm->GetTitle()));
            tdir = gSystem->DirName(tdir.Data());
            TParameter<Int_t> *nq = dynamic_cast<TParameter<Int_t>*>(dl->FindObject(tdir));
            if (nq) {
               Int_t val = nq->GetVal();
               nq->SetVal(--val);
               if (nq->GetVal() <= 0)
                  // Remove the directory if empty
                  gSystem->Exec(Form("%s -fr %s", kRM, tdir.Data()));
            }
         }
      }
   }

   // Cleanup
   delete sl;
   delete hl;
   delete dl;

   // Done
   return 0;
}

//______________________________________________________________________________
Int_t TQueryResultManager::LockSession(const char *sessiontag, TProofLockPath **lck)
{
   // Try locking query area of session tagged sessiontag.
   // The id of the locking file is returned in fid and must be
   // unlocked via UnlockQueryFile(fid).

   // We do not need to lock our own session
   if (strstr(sessiontag, fSessionTag))
      return 0;

   if (!lck) {
      Error("LockSession","locker space undefined");
      return -1;
   }
   *lck = 0;

   // Check the format
   TString stag = sessiontag;
   TRegexp re("session-.*-.*-.*-.*");
   Int_t i1 = stag.Index(re);
   if (i1 == kNPOS) {
      Error("LockSession","bad format: %s", sessiontag);
      return -1;
   }
   stag.ReplaceAll("session-","");

   // Drop query number, if any
   Int_t i2 = stag.Index(":q");
   if (i2 != kNPOS)
      stag.Remove(i2);

   // Make sure that parent process does not exist anylonger
   TString parlog = fSessionDir;
   parlog = parlog.Remove(parlog.Index("master-")+strlen("master-"));
   parlog += stag;
   if (!gSystem->AccessPathName(parlog)) {
      PDB(kGlobal, 1)
         Info("LockSession", "parent still running: do nothing");
      return -1;
   }

   // Lock the query lock file
   if (fLock) {
      TString qlock = fLock->GetName();
      qlock.ReplaceAll(fSessionTag, stag);

      if (!gSystem->AccessPathName(qlock)) {
         *lck = new TProofLockPath(qlock);
         if (((*lck)->Lock()) < 0) {
            Error("LockSession","problems locking query lock file");
            SafeDelete(*lck);
            return -1;
         }
      }
   }

   // We are done
   return 0;
}

//______________________________________________________________________________
Int_t TQueryResultManager::CleanupSession(const char *sessiontag)
{
   // Cleanup query dir qdir.

   if (!sessiontag) {
      Error("CleanupSession","session tag undefined");
      return -1;
   }

   // Query dir
   TString qdir = fQueryDir;
   qdir.ReplaceAll(Form("session-%s", fSessionTag.Data()), sessiontag);
   Int_t idx = qdir.Index(":q");
   if (idx != kNPOS)
      qdir.Remove(idx);
   if (gSystem->AccessPathName(qdir)) {
      Info("CleanupSession","query dir %s does not exist", qdir.Data());
      return -1;
   }

   TProofLockPath *lck = 0;
   if (LockSession(sessiontag, &lck) == 0) {

      // Cleanup now
      gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

      // Unlock and remove the lock file
      if (lck) {
         gSystem->Unlink(lck->GetName());
         SafeDelete(lck);  // Unlocks, if necessary
      }

      // We are done
      return 0;
   }

   // Notify failure
   Info("CleanupSession", "could not lock session %s", sessiontag);
   return -1;
}

//______________________________________________________________________________
void TQueryResultManager::SaveQuery(TProofQueryResult *qr, const char *fout)
{
   // Save current status of query 'qr' to file name fout.
   // If fout == 0 (default) use the default name.

   if (!qr || qr->IsDraw())
      return;

   // Create dir for specific query
   TString querydir = Form("%s/%d",fQueryDir.Data(), qr->GetSeqNum());

   // Create dir, if needed
   if (gSystem->AccessPathName(querydir))
      gSystem->MakeDirectory(querydir);
   TString ofn = fout ? fout : Form("%s/query-result.root", querydir.Data());

   // Recreate file and save query in its current status
   TFile *f = TFile::Open(ofn, "RECREATE");
   if (f) {
      f->cd();
      if (!(qr->IsArchived()))
         qr->SetResultFile(ofn);
      qr->Write();
      f->Close();
      delete f;
   }
}

//______________________________________________________________________________
void TQueryResultManager::RemoveQuery(const char *queryref, TList *otherlist)
{
   // Remove everything about query queryref; if defined 'otherlist' will containe
   // the list of removed pointers (already deleted)

   PDB(kGlobal, 1)
      Info("RemoveQuery", "Enter");

   // Parse reference string
   Int_t qry = -1;
   TString qdir;
   TProofQueryResult *pqr = LocateQuery(queryref, qry, qdir);
   // Remove instance in memory
   if (pqr) {
      if (qry > -1) {
         fQueries->Remove(pqr);
         if (otherlist) otherlist->Add(pqr);
      } else
         fPreviousQueries->Remove(pqr);
      delete pqr;
      pqr = 0;
   }

   // Remove the directory
   PDB(kGlobal, 1)
      Info("RemoveQuery", "removing directory: %s", qdir.Data());
   gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

   // Done
   return;
}

//______________________________________________________________________________
void TQueryResultManager::RemoveQuery(TQueryResult *qr, Bool_t soft)
{
   // Remove everything about query qr. If soft = TRUE leave a track
   // in memory with the relevant info

   PDB(kGlobal, 1)
      Info("RemoveQuery", "Enter");

   if (!qr)
      return;

   // Remove the directory
   TString qdir = fQueryDir;
   qdir = qdir.Remove(qdir.Index(kPROOF_QueryDir)+strlen(kPROOF_QueryDir));
   qdir = Form("%s/%s/%d", qdir.Data(), qr->GetTitle(), qr->GetSeqNum());
   PDB(kGlobal, 1)
      Info("RemoveQuery", "removing directory: %s", qdir.Data());
   gSystem->Exec(Form("%s %s", kRM, qdir.Data()));

   // Remove from memory lists
   if (soft) {
      TQueryResult *qrn = qr->CloneInfo();
      Int_t idx = fQueries->IndexOf(qr);
      if (idx > -1)
         fQueries->AddAt(qrn, idx);
      else
         SafeDelete(qrn);
   }
   fQueries->Remove(qr);
   SafeDelete(qr);

   // Done
   return;
}

//______________________________________________________________________________
TProofQueryResult *TQueryResultManager::LocateQuery(TString queryref, Int_t &qry, TString &qdir)
{
   // Locate query referenced by queryref. Return pointer to instance
   // in memory, if any, or 0. Fills qdir with the query specific directory
   // and qry with the query number for queries processed by this session.

   TProofQueryResult *pqr = 0;

   // Find out if the request is a for a local query or for a
   // previously processed one
   qry = -1;
   if (queryref.IsDigit()) {
      qry = queryref.Atoi();
   } else if (queryref.Contains(fSessionTag)) {
      Int_t i1 = queryref.Index(":q");
      if (i1 != kNPOS) {
         queryref.Remove(0,i1+2);
         qry = queryref.Atoi();
      }
   }

   // Build dir name for specific query
   qdir = "";
   if (qry > -1) {

      PDB(kGlobal, 1)
         Info("LocateQuery", "local query: %d", qry);

      // Remove query from memory list
      if (fQueries) {
         TIter nxq(fQueries);
         while ((pqr = (TProofQueryResult *) nxq())) {
            if (pqr->GetSeqNum() == qry) {
               // Dir for specific query
               qdir = Form("%s/%d", fQueryDir.Data(), qry);
               break;
            }
         }
      }

   } else {
      PDB(kGlobal, 1)
         Info("LocateQuery", "previously processed query: %s", queryref.Data());

      // Remove query from memory list
      if (fPreviousQueries) {
         TIter nxq(fPreviousQueries);
         while ((pqr = (TProofQueryResult *) nxq())) {
            if (queryref.Contains(pqr->GetTitle()) &&
                queryref.Contains(pqr->GetName()))
               break;
         }
      }

      queryref.ReplaceAll(":q","/");
      qdir = fQueryDir;
      qdir = qdir.Remove(qdir.Index(kPROOF_QueryDir)+strlen(kPROOF_QueryDir));
      qdir = Form("%s/%s", qdir.Data(), queryref.Data());
   }

   // We are done
   return pqr;
}

//______________________________________________________________________________
Bool_t TQueryResultManager::FinalizeQuery(TProofQueryResult *pq,
                                          TProof *proof, TVirtualProofPlayer *player)
{
   // Final steps after Process() to complete the TQueryResult instance.

   if (!pq || !proof || !player) {
      Warning("FinalizeQuery", "bad inputs: query = %p, proof = %p, player: %p ",
              pq ? pq : 0, proof ? proof : 0, player ? player : 0);
      return kFALSE;
   }

   Int_t qn = pq->GetSeqNum();
   Long64_t np = player->GetEventsProcessed();
   TVirtualProofPlayer::EExitStatus est = player->GetExitStatus();
   TList *out = player->GetOutputList();

   Float_t cpu = proof->GetCpuTime();
   Long64_t bytes = proof->GetBytesRead();

   TQueryResult::EQueryStatus st = TQueryResult::kAborted;

   PDB(kGlobal, 2) Info("FinalizeQuery","query #%d", qn);

   PDB(kGlobal, 1)
      Info("FinalizeQuery","%.1f %lld", cpu, bytes);

   // Some notification (useful in large logs)
   Bool_t save = kTRUE;
   switch (est) {
   case TVirtualProofPlayer::kAborted:
      PDB(kGlobal, 1)
         Info("FinalizeQuery", "query %d has been ABORTED <====", qn);
      out = 0;
      save = kFALSE;
      break;
   case TVirtualProofPlayer::kStopped:
      PDB(kGlobal, 1)
         Info("FinalizeQuery",
              "query %d has been STOPPED: %lld events processed", qn, np);
      st = TQueryResult::kStopped;
      break;
   case TVirtualProofPlayer::kFinished:
      PDB(kGlobal, 1)
         Info("FinalizeQuery",
              "query %d has been completed: %lld events processed", qn, np);
      st = TQueryResult::kCompleted;
      break;
   default:
      Warning("FinalizeQuery",
              "query %d: unknown exit status (%d)", qn, player->GetExitStatus());
   }

   // Fill some variables; in the CPU time we include also the time used on the
   // master fro preparing and merging
   PDB(kGlobal, 1)
      Info("FinalizeQuery", "cpu: %.4f, saved: %.4f, master: %.4f",
                            cpu, pq->GetUsedCPU() ,GetCpuTime());
   pq->SetProcessInfo(np, cpu - pq->GetUsedCPU() + GetCpuTime());
   pq->RecordEnd(st, out);

   // Save the logs into the query result instance
   AddLogFile(pq);

   // Done
   return save;
}

//______________________________________________________________________________
void TQueryResultManager::SaveQuery(TProofQueryResult *pq, Int_t mxq)
{
   // Save current query honouring the max number of queries allowed

   // We may need some cleanup
   if (mxq > -1) {
      if (fQueries && fKeptQueries >= mxq) {
         // Find oldest completed and archived query
         TQueryResult *fcom = 0;
         TQueryResult *farc = 0;
         TIter nxq(fQueries);
         TQueryResult *qr = 0;
         while (fKeptQueries >= mxq) {
            while ((qr = (TQueryResult *) nxq())) {
               if (qr->IsArchived()) {
                  if (qr->GetOutputList() && !farc)
                     farc = qr;
               } else if (qr->GetStatus() > TQueryResult::kRunning && !fcom) {
                  fcom = qr;
               }
               if (farc && fcom)
                  break;
            }
            if (!farc && !fcom) {
               break;
            } else if (farc) {
               RemoveQuery(farc, kTRUE);
               fKeptQueries--;
               farc = 0;
            } else if (fcom) {
               RemoveQuery(fcom);
               fKeptQueries--;
               fcom = 0;
            }
         }
      }
      if (fKeptQueries < mxq) {
         SaveQuery(pq);
         fKeptQueries++;
      } else {
         TString emsg;
         emsg.Form("Too many saved queries (%d): cannot save %s:%s",
                   fKeptQueries, pq->GetTitle(), pq->GetName());
         if (gProofServ) {
            gProofServ->SendAsynMessage(emsg.Data());
         } else {
            Warning("SaveQuery", "%s", emsg.Data());
         }
      }
   } else {
      SaveQuery(pq);
      fKeptQueries++;
   }
}
