// @(#)root/proofplayer:$Id$
// Author: G. Ganis 2009

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPacketizerFile                                                      //
//                                                                      //
// This packetizer generates packets which contain a single file path   //
// to be used in process. Used for tasks generating files, like in      //
// PROOF bench.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPacketizerFile.h"

#include "Riostream.h"
#include "TDSet.h"
#include "TError.h"
#include "TEventList.h"
#include "TMap.h"
#include "TMessage.h"
#include "TMonitor.h"
#include "TNtupleD.h"
#include "TObject.h"
#include "TParameter.h"
#include "TPerfStats.h"
#include "TProofDebug.h"
#include "TProof.h"
#include "TProofPlayer.h"
#include "TProofServ.h"
#include "TSlave.h"
#include "TSocket.h"
#include "TStopwatch.h"
#include "TTimer.h"
#include "TUrl.h"
#include "TClass.h"
#include "TMath.h"
#include "TObjString.h"
#include "TFileInfo.h"
#include "TFileCollection.h"
#include "THashList.h"

//------------------------------------------------------------------------------

class TPacketizerFile::TSlaveStat : public TVirtualPacketizer::TVirtualSlaveStat {

friend class TPacketizerFile;

private:
   Long64_t  fLastProcessed; // number of processed entries of the last packet
   Double_t  fSpeed;         // estimated current average speed of the processing slave
   Double_t  fTimeInstant;   // stores the time instant when the current packet started
   TNtupleD *fCircNtp;       // Keeps circular info for speed calculations
   Long_t    fCircLvl;       // Circularity level

public:
   TSlaveStat(TSlave *sl, TList *input);
   ~TSlaveStat();

   void        GetCurrentTime();

   void        UpdatePerformance(Double_t time);
   TProofProgressStatus *AddProcessed(TProofProgressStatus *st);
};

// Iterator wrapper
class TPacketizerFile::TIterObj : public TObject {

private:
   TString   fName;          // Name of reference
   TIter    *fIter;          // Iterator

public:
   TIterObj(const char *n, TIter *iter) : fName(n), fIter(iter) { }
   virtual ~TIterObj() { if (fIter) delete fIter; }

   const char *GetName() const {return fName;}
   TIter      *GetIter() const {return fIter;}
   void        Print(Option_t* option = "") const;
};

ClassImp(TPacketizerFile)

//______________________________________________________________________________
TPacketizerFile::TPacketizerFile(TList *workers, Long64_t, TList *input,
                                 TProofProgressStatus *st)
                : TVirtualPacketizer(input, st)
{
   // Constructor

   PDB(kPacketizer,1) Info("TPacketizerFile", "enter");
   ResetBit(TObject::kInvalidObject);
   fValid = kFALSE;
   fAssigned = 0;
   fProcNotAssigned = kTRUE;

   if (!input || (input && input->GetSize() <= 0)) {
      Error("TPacketizerFile", "input file is undefined or empty!");
      SetBit(TObject::kInvalidObject);
      return;
   }

   // Check if the files not explicitely assigned have to be processed
   Int_t procnotass = 1;
   if (TProof::GetParameter(input, "PROOF_ProcessNotAssigned", procnotass) == 0) {
      if (procnotass == 0) {
         Info("TPacketizerFile", "files not assigned to workers will not be processed");
         fProcNotAssigned = kFALSE;
      }
   }

   // These are the file to be created/processed per node; the information
   if (!(fFiles = dynamic_cast<TMap *>(input->FindObject("PROOF_FilesToProcess")))) {
      Error("TPacketizerFile", "map of files to be processed/created not found");
      SetBit(TObject::kInvalidObject);
      return;
   }

   // The worker stats
   fSlaveStats = new TMap;
   fSlaveStats->SetOwner(kFALSE);

   TList nodes;
   nodes.SetOwner(kTRUE);
   TSlave *wrk;
   TIter si(workers);
   while ((wrk = (TSlave *) si.Next())) {
      fSlaveStats->Add(wrk, new TSlaveStat(wrk, input));
      TString wrkname = TUrl(wrk->GetName()).GetHostFQDN();
      Info("TPacketizerFile", "worker: %s", wrkname.Data());
      if (!nodes.FindObject(wrkname)) nodes.Add(new TObjString(wrkname));
   }

   // The list of iterators
   fIters = new TList;
   fIters->SetOwner(kTRUE);

   // There must be something in
   fTotalEntries = 0;
   fNotAssigned = new TList;
   fNotAssigned->SetName("*");
   TIter nxl(fFiles);
   TObject *key, *o = 0;
   while ((key = nxl()) != 0) {
      THashList *wrklist = dynamic_cast<THashList *>(fFiles->GetValue(key));
      if (!wrklist) {
         TFileCollection *fc = dynamic_cast<TFileCollection *>(fFiles->GetValue(key));
         if (fc) wrklist = fc->GetList();
      }
      if (wrklist) {
         TString hname = TUrl(key->GetName()).GetHostFQDN();
         if ((o = nodes.FindObject(hname))) {
            fTotalEntries += wrklist->GetSize();
            fIters->Add(new TIterObj(hname, new TIter(wrklist)));
            // Notify
            PDB(kPacketizer,2)
               Info("TPacketizerFile", "%d files of '%s' (fqdn: '%s') assigned to '%s'",
                                       wrklist->GetSize(), key->GetName(), hname.Data(), o->GetName());
         } else {
            // We add all to the not assigned list so that they will be distributed
            // according to the load
            TIter nxf(wrklist);
            while ((o = nxf()))
               fNotAssigned->Add(o);
            // Notify
            PDB(kPacketizer,2)
               Info("TPacketizerFile", "%d files of '%s' (fqdn: '%s') not assigned",
                                       wrklist->GetSize(), key->GetName(), hname.Data());
         }
      }
   }
   if (fNotAssigned && fNotAssigned->GetSize() > 0) {
      fTotalEntries += fNotAssigned->GetSize();
      fIters->Add(new TIterObj("*", new TIter(fNotAssigned)));
      Info("TPacketizerFile", "non-assigned files: %d", fNotAssigned->GetSize());
      fNotAssigned->Print();
   }
   if (fTotalEntries <= 0) {
      Error("TPacketizerFile", "no file path in the map!");
      SetBit(TObject::kInvalidObject);
      SafeDelete(fIters);
      return;
   } else {
      Info("TPacketizerFile", "processing %lld files", fTotalEntries);
      fIters->Print();
   }

   fStopwatch = new TStopwatch();
   fStopwatch->Start();
   fValid = kTRUE;
   PDB(kPacketizer,1) Info("TPacketizerFile", "return");

   // Done
   return;
}

//______________________________________________________________________________
TPacketizerFile::~TPacketizerFile()
{
   // Destructor.

   if (fNotAssigned) fNotAssigned->SetOwner(kFALSE);
   SafeDelete(fNotAssigned);
   if (fIters) fIters->SetOwner(kTRUE);
   SafeDelete(fIters);
   SafeDelete(fStopwatch);
}

//______________________________________________________________________________
Double_t TPacketizerFile::GetCurrentTime()
{
   // Get current time

   Double_t retValue = fStopwatch->RealTime();
   fStopwatch->Continue();
   return retValue;
}

//______________________________________________________________________________
Float_t TPacketizerFile::GetCurrentRate(Bool_t &all)
{
   // Get Estimation of the current rate; just summing the current rates of
   // the active workers

   all = kTRUE;
   // Loop over the workers
   Float_t currate = 0.;
   if (fSlaveStats && fSlaveStats->GetSize() > 0) {
      TIter nxw(fSlaveStats);
      TObject *key;
      while ((key = nxw()) != 0) {
         TSlaveStat *wrkstat = (TSlaveStat *) fSlaveStats->GetValue(key);
         if (wrkstat && wrkstat->GetProgressStatus() && wrkstat->GetEntriesProcessed() > 0) {
            // Sum-up the current rates
            currate += wrkstat->GetProgressStatus()->GetCurrentRate();
         } else {
            all = kFALSE;
         }
      }
   }
   // Done
   return currate;
}

//______________________________________________________________________________
TDSetElement *TPacketizerFile::GetNextPacket(TSlave *wrk, TMessage *r)
{
   // Get next packet

   TDSetElement *elem = 0;
   if (!fValid)  return elem;

   // Find slave
   TSlaveStat *wrkstat = (TSlaveStat *) fSlaveStats->GetValue(wrk);
   if (!wrkstat) {
      Error("GetNextPacket", "could not find stat object for worker '%s'!", wrk->GetName());
      return elem;
   }

   PDB(kPacketizer,2)
      Info("GetNextPacket","worker-%s: fAssigned %lld / %lld", wrk->GetOrdinal(), fAssigned, fTotalEntries);

   // Update stats & free old element
   Double_t latency = 0., proctime = 0., proccpu = 0.;
   Long64_t bytesRead = -1;
   Long64_t totalEntries = -1; // used only to read an old message type
   Long64_t totev = 0;
   Long64_t numev = -1;

   TProofProgressStatus *status = 0;
   if (wrk->GetProtocol() > 18) {
      (*r) >> latency;
      (*r) >> status;

      // Calculate the progress made in the last packet
      TProofProgressStatus *progress = 0;
      if (status) {
         // upadte the worker status
         numev = status->GetEntries() - wrkstat->GetEntriesProcessed();
         progress = wrkstat->AddProcessed(status);
         if (progress) {
            // (*fProgressStatus) += *progress;
            proctime = progress->GetProcTime();
            proccpu  = progress->GetCPUTime();
            totev  = status->GetEntries(); // for backward compatibility
            bytesRead  = progress->GetBytesRead();
            delete progress;
         }
         delete status;
      } else
          Error("GetNextPacket", "no status came in the kPROOF_GETPACKET message");
   } else {

      (*r) >> latency >> proctime >> proccpu;

      // only read new info if available
      if (r->BufferSize() > r->Length()) (*r) >> bytesRead;
      if (r->BufferSize() > r->Length()) (*r) >> totalEntries;
      if (r->BufferSize() > r->Length()) (*r) >> totev;

      numev = totev - wrkstat->GetEntriesProcessed();
      wrkstat->GetProgressStatus()->IncEntries(numev);
   }

   fProgressStatus->IncEntries(numev);

   PDB(kPacketizer,2)
      Info("GetNextPacket","worker-%s (%s): %lld %7.3lf %7.3lf %7.3lf %lld",
                           wrk->GetOrdinal(), wrk->GetName(),
                           numev, latency, proctime, proccpu, bytesRead);

   if (gPerfStats != 0) {
      gPerfStats->PacketEvent(wrk->GetOrdinal(), wrk->GetName(), "", numev,
                              latency, proctime, proccpu, bytesRead);
   }

   if (fAssigned == fTotalEntries) {
      // Send last timer message
      HandleTimer(0);
      return 0;
   }

   if (fStop) {
      // Send last timer message
      HandleTimer(0);
      return 0;
   }

   PDB(kPacketizer,2)
      Info("GetNextPacket", "worker-%s (%s): getting next files ... ", wrk->GetOrdinal(),
                            wrk->GetName());

   // Get next file now
   TObject *nextfile = 0;

   // Find iterator associated to the worker
   TString wrkname = TUrl(wrk->GetName()).GetHostFQDN();
   TIterObj *io = dynamic_cast<TIterObj *>(fIters->FindObject(wrkname));
   if (io) {
      // Get next file to process in the list of the worker
      if (io->GetIter())
         nextfile = io->GetIter()->Next();
   }

   // If not found or all files already processed, check if a generic iterator
   // has still some files to process
   if (!nextfile && fProcNotAssigned) {
      if ((io = dynamic_cast<TIterObj *>(fIters->FindObject("*")))) {
         // Get next file to process in the list of the worker
         if (io->GetIter())
            nextfile = io->GetIter()->Next();
      }
   }

   // Return if nothing to process
   if (!nextfile) return elem;

   // The file name: we support TObjString or TFileInfo
   TString filename;
   TObjString *os = 0;
   if ((os = dynamic_cast<TObjString *>(nextfile))) {
      filename = os->GetName();
   } else {
      TFileInfo *fi = 0;
      if ((fi = dynamic_cast<TFileInfo *>(nextfile)))
         filename = fi->GetCurrentUrl()->GetUrl();
   }
   // Nothing to process
   if (filename.IsNull()) {
      Warning("GetNextPacket", "found unsupported object of type '%s' in list: it must"
                               " be 'TObjString' or 'TFileInfo'", nextfile->GetName());
      return elem;
   }
   // Prepare the packet
   PDB(kPacketizer,2)
      Info("GetNextPacket", "worker-%s: assigning: '%s' (remaining %lld files)",
                            wrk->GetOrdinal(), filename.Data(), (fTotalEntries - fAssigned));
   elem = new TDSetElement(filename, "", "", 0, 1);
   elem->SetBit(TDSetElement::kEmpty);

   // Update the total counter
   fAssigned += 1;

   return elem;
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
TPacketizerFile::TSlaveStat::TSlaveStat(TSlave *slave, TList *input)
                            : fLastProcessed(0),
                              fSpeed(0), fTimeInstant(0), fCircLvl(5)
{
   // Main constructor

   // Initialize the circularity ntple for speed calculations
   fCircNtp = new TNtupleD("Speed Circ Ntp", "Circular process info","tm:ev");
   TProof::GetParameter(input, "PROOF_TPacketizerFileCircularity", fCircLvl);
   fCircLvl = (fCircLvl > 0) ? fCircLvl : 5;
   fCircNtp->SetCircular(fCircLvl);
   fSlave = slave;
   fStatus = new TProofProgressStatus();
}

//______________________________________________________________________________
TPacketizerFile::TSlaveStat::~TSlaveStat()
{
   // Destructor

   SafeDelete(fCircNtp);
}

//______________________________________________________________________________
void TPacketizerFile::TSlaveStat::UpdatePerformance(Double_t time)
{
   // Update the circular ntple

   Double_t ttot = time;
   Double_t *ar = fCircNtp->GetArgs();
   Int_t ne = fCircNtp->GetEntries();
   if (ne <= 0) {
      // First call: just fill one ref entry and return
      fCircNtp->Fill(0., 0);
      fSpeed = 0.;
      return;
   }
   // Fill the entry
   fCircNtp->GetEntry(ne-1);
   ttot = ar[0] + time;
   fCircNtp->Fill(ttot, GetEntriesProcessed());

   // Calculate the speed
   fCircNtp->GetEntry(0);
   Double_t dtime = (ttot > ar[0]) ? ttot - ar[0] : ne+1 ;
   Long64_t nevts = GetEntriesProcessed() - (Long64_t)ar[1];
   fSpeed = nevts / dtime;
   PDB(kPacketizer,2)
      Info("UpdatePerformance", "time:%f, dtime:%f, nevts:%lld, speed: %f",
                                time, dtime, nevts, fSpeed);

}

//______________________________________________________________________________
TProofProgressStatus *TPacketizerFile::TSlaveStat::AddProcessed(TProofProgressStatus *st)
{
   // Update the status info to the 'st'.
   // return the difference (*st - *fStatus)

   if (st) {
      // The entriesis not correct in 'st'
      Long64_t lastEntries = st->GetEntries() - fStatus->GetEntries();
      // The last proc time should not be added
      fStatus->SetLastProcTime(0.);
      // Get the diff
      TProofProgressStatus *diff = new TProofProgressStatus(*st - *fStatus);
      *fStatus += *diff;
      // Set the correct value
      fStatus->SetLastEntries(lastEntries);
      return diff;
   } else {
      Error("AddProcessed", "status arg undefined");
      return 0;
   }
}

//______________________________________________________________________________
void TPacketizerFile::TIterObj::Print(Option_t *) const
{
   // Printf info

   Printf("Iterator '%s' controls %d units", GetName(),
          ((GetIter() && GetIter()->GetCollection()) ? GetIter()->GetCollection()->GetSize()
                                                     : -1));
}
