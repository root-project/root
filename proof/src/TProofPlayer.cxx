// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.34 2004/05/18 11:32:49 rdm Exp $
// Author: Maarten Ballintijn   07/01/02

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofPlayer                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProofPlayer.h"

#include "THashList.h"
#include "TEventIter.h"
#include "TVirtualPacketizer.h"
#include "TPacketizer.h"
#include "TPacketizer2.h"
#include "TSelector.h"
#include "TSocket.h"
#include "TProofServ.h"
#include "TProof.h"
#include "TSlave.h"
#include "TROOT.h"
#include "TError.h"
#include "MessageTypes.h"
#include "TMessage.h"
#include "TDSetProxy.h"
#include "TString.h"
#include "TSystem.h"
#include "TFile.h"
#include "TProofDebug.h"
#include "TTimer.h"
#include "TMap.h"
#include "TEnv.h"
#include "TProofStats.h"

#include "Api.h"


class TAutoBinVal : public TObjString {
private:
   Double_t fXmin, fXmax, fYmin, fYmax, fZmin, fZmax;

public:
   TAutoBinVal(const char *name, Double_t xmin, Double_t xmax, Double_t ymin,
               Double_t ymax, Double_t zmin, Double_t zmax) : TObjString(name)
   {
      fXmin = xmin; fXmax = xmax;
      fYmin = ymin; fYmax = ymax;
      fZmin = zmin; fZmax = zmax;
   }
   void GetAll(Double_t& xmin, Double_t& xmax, Double_t& ymin,
               Double_t& ymax, Double_t& zmin, Double_t& zmax)
   {
      xmin = fXmin; xmax = fXmax;
      ymin = fYmin; ymax = fYmax;
      zmin = fZmin; zmax = fZmax;
   }

};


//------------------------------------------------------------------------------


ClassImp(TProofPlayer)

//______________________________________________________________________________
TProofPlayer::TProofPlayer()
{
   // Default ctor.

   fAutoBins      = 0;
   fInput         = new TList;
   fOutput        = 0;
   fSelector      = 0;
   fSelectorClass = 0;
   fFeedbackTimer = 0;
   fEvIter        = 0;
}

//______________________________________________________________________________
TProofPlayer::~TProofPlayer()
{
   delete fInput;
   if (fSelectorClass && fSelectorClass->IsLoaded()) delete fSelector;
   delete fFeedbackTimer;
   delete fEvIter;
}

//______________________________________________________________________________
void TProofPlayer::StopProcess(Bool_t abort)
{
   if (fEvIter != 0) fEvIter->StopProcess(abort);
}

//______________________________________________________________________________
void TProofPlayer::AddInput(TObject *inp)
{
   fInput->Add(inp);
}

//______________________________________________________________________________
void TProofPlayer::ClearInput()
{
   fInput->Clear();
}

//______________________________________________________________________________
TObject *TProofPlayer::GetOutput(const char *name) const
{
   if (fOutput != 0) {
      return fOutput->FindObject(name);
   } else {
      return 0;
   }
}

//______________________________________________________________________________
TList *TProofPlayer::GetOutputList() const
{
   return fOutput;
}

//______________________________________________________________________________
void TProofPlayer::StoreOutput(TList *)
{
   MayNotUse("StoreOutput");
}

//______________________________________________________________________________
void TProofPlayer::StoreFeedback(TSlave *, TList *)
{
   MayNotUse("StoreFeedback");
}

//______________________________________________________________________________
void TProofPlayer::Progress(Long64_t total, Long64_t processed)
{
   PDB(kGlobal,1)
      Info("Progress","%2f (%lld/%lld)", 100.*processed/total, processed, total);

   Long_t parm[2];
   parm[0] = (Long_t) (&total);
   parm[1] = (Long_t) (&processed);
   Emit("Progress(Long64_t,Long64_t)", parm);

   gProof->Progress(total, processed);
}

//______________________________________________________________________________
void TProofPlayer::Feedback(TList *objs)
{
   PDB(kGlobal,1) Info("Feedback","%d Objects", objs->GetSize());
   PDB(kFeedback,1) {
      Info("Feedback","%d Objects", objs->GetSize());
      objs->ls();
   }

   Emit("Feedback(TList *objs)", (Long_t) objs);

   gProof->Feedback(objs);
}

//______________________________________________________________________________
Int_t TProofPlayer::Process(TDSet *dset, const char *selector_file,
                            Option_t *option, Long64_t nentries, Long64_t first,
                            TEventList * /*evl*/)
{
   PDB(kGlobal,1) Info("Process","Enter");

   fOutput = 0;
   if (fSelectorClass && fSelectorClass->IsLoaded()) delete fSelector;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      fSelectorClass = 0;
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }
   fSelectorClass = fSelector->IsA();

   Int_t version = fSelector->Version();

   TCleanup clean(this);
   SetupFeedback();

   fSelector->SetOption(option);
   fSelector->SetInputList(fInput);

   dset->Reset();

   fEvIter = TEventIter::Create(dset, fSelector, first, nentries);


   if (version == 0) {
      PDB(kLoop,1) Info("Process","Call Begin(0)");
      fSelector->Begin(0);
   } else {
      if (gProof != 0 && !gProof->IsMaster()) {
         // on client (for local run)
         PDB(kLoop,1) Info("Process","Call Begin(0)");
         fSelector->Begin(0);
      }
      PDB(kLoop,1) Info("Process","Call SlaveBegin(0)");
      fSelector->SlaveBegin(0);  // Init is called explicitly from GetNextEvent()
   }

   PDB(kLoop,1) Info("Process","Looping over Process()");

   // Loop over range
   Long64_t entry;
   while ((entry = fEvIter->GetNextEvent()) >= 0) {


      if(version == 0) {
         PDB(kLoop,3)Info("Process","Call ProcessCut(%lld)", entry);
         if(fSelector->ProcessCut(entry)) {
            PDB(kLoop,3)Info("Process","Call ProcessFill(%lld)", entry);
            fSelector->ProcessFill(entry);
         }
      } else {
         PDB(kLoop,3)Info("Process","Call Process(%lld)", entry);
         fSelector->Process(entry);
      }

      gSystem->DispatchOneEvent(kTRUE);
      if (gROOT->IsInterrupted()) break;
   }

   StopFeedback();

   delete fEvIter; fEvIter = 0;

   // Finalize

   if (version == 0) {
      PDB(kLoop,1) Info("Process","Call Terminate()");
      fSelector->Terminate();
   } else {
      PDB(kLoop,1) Info("Process","Call SlaveTerminate()");
      fSelector->SlaveTerminate();
      if (gProof != 0 && !gProof->IsMaster()) {
         PDB(kLoop,1) Info("Process","Call Terminate()");
         fSelector->Terminate();
      }
   }

   fOutput = fSelector->GetOutputList();

   return 0;
}

//______________________________________________________________________________
void TProofPlayer::UpdateAutoBin(const char *name,
                                 Double_t& xmin, Double_t& xmax,
                                 Double_t& ymin, Double_t& ymax,
                                 Double_t& zmin, Double_t& zmax)
{
   if ( fAutoBins == 0 ) {
      fAutoBins = new THashList;
   }

   TAutoBinVal *val = (TAutoBinVal*) fAutoBins->FindObject(name);

   if ( val == 0 ) {
      val = new TAutoBinVal(name,xmin,xmax,ymin,ymax,zmin,zmax);
      fAutoBins->Add(val);
   } else {
      val->GetAll(xmin,xmax,ymin,ymax,zmin,zmax);
   }
}

//______________________________________________________________________________
TDSetElement *TProofPlayer::GetNextPacket(TSlave *, TMessage *)
{
   MayNotUse("GetNextPacket");
   return 0;
}

//______________________________________________________________________________
void TProofPlayer::SetupFeedback()
{
   MayNotUse("SetupFeedback");
}

//______________________________________________________________________________
void TProofPlayer::StopFeedback()
{
   MayNotUse("StopFeedback");
}

//______________________________________________________________________________
Int_t TProofPlayer::DrawSelect(TDSet *set, const char *varexp,
                               const char *selection, Option_t *option,
                               Long64_t nentries, Long64_t firstentry)
{
   TNamed *varexpobj = new TNamed("varexp", varexp);
   TNamed *selectionobj = new TNamed("selection", selection);

   fInput->Clear();  // good idea? what about a feedbacklist, but old query
                     // could have left objs? clear at end? no, may want to
                     // rerun, separate player?

   fInput->Add(varexpobj);
   fInput->Add(selectionobj);

   Int_t r = Process(set, "TProofDraw", option, nentries, firstentry);

   fInput->Remove(varexpobj);
   fInput->Remove(selectionobj);
   delete varexpobj;
   delete selectionobj;

   return r;
}



//------------------------------------------------------------------------------

ClassImp(TProofPlayerLocal)


//------------------------------------------------------------------------------

ClassImp(TProofPlayerRemote)


//______________________________________________________________________________
TProofPlayerRemote::TProofPlayerRemote(TProof *proof)
{
   fProof         = proof;
   fOutputLists   = 0;
   fPacketizer    = 0;
   fFeedbackLists = 0;
   fProofStats    = 0;
}

//______________________________________________________________________________
TProofPlayerRemote::~TProofPlayerRemote()
{
   delete fOutput;      // owns the output list
   delete fOutputLists;

   if (fFeedbackLists != 0) {
      TIter next(fFeedbackLists);
      while (TMap *m = (TMap*) next()) {
         m->DeleteValues();
      }
   }
   delete fFeedbackLists;
   delete fPacketizer;
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::Process(TDSet *dset, const char *selector_file,
                                  Option_t *option, Long64_t nentries,
                                  Long64_t first, TEventList * /*evl*/)
{
   // Process specified TDSet on PROOF.
   // Returns -1 in case error, 0 otherwise.

   PDB(kGlobal,1) Info("Process","Enter");

   delete fOutput;
   fOutput = new TList;

   if(fProof->IsMaster()){
      Bool_t doHist = (fInput->FindObject("PROOF_StatsHist") != 0);
      Bool_t doTrace = (fInput->FindObject("PROOF_StatsTrace") != 0);

      if (doHist || doTrace) {
         Int_t nslaves = fProof->GetListOfSlaves()->GetSize();
         fProofStats = new TProofStats(nslaves, fOutput, doHist, doTrace);
         fProofStats->SimpleEvent(TProofEvent::kStart);
      }
   } else {
      if (gEnv->GetValue("Proof.StatsHist", 0)) {
         fInput->Add(new TNamed("PROOF_StatsHist",""));
      }
      if (gEnv->GetValue("Proof.StatsTrace", 0)) {
         fInput->Add(new TNamed("PROOF_StatsTrace",""));
      }
   }

   // If the filename does not contain "." assume class is compiled in
   if ( strchr(selector_file,'.') != 0 ) {
      TString filename = selector_file;
      TString aclicMode;
      TString arguments;
      TString io;
      filename = gSystem->SplitAclicMode(filename, aclicMode, arguments, io);

      PDB(kSelector,1) Info("Process", "Sendfile: %s", filename.Data() );
      if ( fProof->SendFile(filename) == -1 ) return -1;

      // NOTE: should we allow more extension?
      if ( filename.EndsWith(".C") ) {
         filename.Replace(filename.Length()-1,1,"h");
         if (!gSystem->AccessPathName(filename,kReadPermission)) {
            PDB(kSelector,1) Info("Process", "SendFile: %s", filename.Data() );
            if ( fProof->SendFile(filename) == -1 ) return -1;
         }
      }
   }

   TMessage mesg(kPROOF_PROCESS);
   TString fn(gSystem->BaseName(selector_file));

   TDSet *set = dset;
   if (fProof->IsMaster()) {

      PDB(kPacketizer,1) Info("Process","Create Proxy TDSet");
      set = new TDSetProxy( dset->GetType(), dset->GetObjName(),
                            dset->GetDirectory() );

      delete fPacketizer;
      fPacketizer = new TPacketizer(dset, fProof->GetListOfActiveSlaves(),
                                     first, nentries);

      if ( !fPacketizer->IsValid() ) {
         return -1;
      }
   } else {
      if (fSelectorClass && fSelectorClass->IsLoaded()) delete fSelector;
      fSelectorClass = 0;
      fSelector = TSelector::GetSelector(selector_file);
      if (fSelector == 0) return -1;
      fSelectorClass = fSelector->IsA();
      fSelector->SetInputList(fInput);
      fSelector->Begin(0);
   }

   TCleanup clean(this);
   SetupFeedback();

   TString opt = option;
   mesg << set << fn << fInput << opt << nentries << first; // no evl yet

   PDB(kGlobal,1) Info("Process","Calling Broadcast");
   fProof->Broadcast(mesg);

   PDB(kGlobal,1) Info("Process","Calling Collect");
   fProof->Collect();

   StopFeedback();

   PDB(kGlobal,1) Info("Process","Calling Merge Output");
   MergeOutput();


   if (fProof->IsMaster()) {
      if (fProofStats != 0) {
         fProofStats->SimpleEvent(TProofEvent::kStop);
         delete fProofStats;
         fProofStats=0;
      }
   } else {
      TIter next(fOutput);
      TList *output = fSelector->GetOutputList();
      while(TObject* obj = next()) {
         output->Add(obj);
      }
      fSelector->Terminate();
   }

   return 0;
}


//______________________________________________________________________________
void TProofPlayerRemote::MergeOutput()
{
   PDB(kOutput,1) Info("MergeOutput","Enter");

   if ( fOutputLists == 0 ) {
      PDB(kOutput,1) Info("MergeOutput","Leave (no output)");
      return;
   }

   TIter next(fOutputLists);

   TList *list;
   while ( (list = (TList *) next()) ) {
      Long_t offset = 0;

      TObject *obj = list->First();
      list->Remove(obj);
      fOutput->Add(obj);

      if ( list->IsEmpty() ) continue;

      // direct CINT, also possible via TInterpreter?
      G__ClassInfo ci(obj->ClassName());
      G__CallFunc cf;

      if ( ci.IsValid() )
         cf.SetFuncProto( &ci, "Merge", "TCollection*", &offset);

      if ( cf.IsValid() ) {
         cf.SetArg((Long_t)list);
         cf.Exec(obj);
      } else {
         // No Merge interface, return individual objects
         while ( (obj = list->First()) ) {
            fOutput->Add(obj);
            list->Remove(obj);
         }
      }
   }

   delete fOutputLists; fOutputLists = 0;
   PDB(kOutput,1) Info("MergeOutput","Leave (%d object(s))", fOutput->GetSize());
}

//______________________________________________________________________________
void TProofPlayerRemote::StopProcess(Bool_t abort)
{
   if (fPacketizer != 0) fPacketizer->StopProcess(abort);
}

//______________________________________________________________________________
void TProofPlayerRemote::StoreOutput(TList *out)
{
   PDB(kOutput,1) Info("StoreOutput","Enter");

   if ( out == 0 ) {
      PDB(kOutput,1) Info("StoreOutput","Leave (empty)");
      return;
   }

   TIter next(out);
   out->SetOwner(kFALSE);  // take ownership of the contents

   if (fOutputLists == 0) {
      PDB(kOutput,2) Info("StoreOutput","Create fOutputLists");
      fOutputLists = new TList;
      fOutputLists->SetOwner();
   }

   TObject *obj;
   while( (obj = next()) ) {
      PDB(kOutput,2) Info("StoreOutput","Find '%s'", obj->GetName() );

      TList *list = (TList *) fOutputLists->FindObject( obj->GetName() );
      if ( list == 0 ) {
         PDB(kOutput,2) Info("StoreOutput","List not Found (creating)", obj->GetName() );
         list = new TList;
         list->SetName( obj->GetName() );
         list->SetOwner();
         fOutputLists->Add( list );
      }
      list->Add( obj );
   }

   delete out;
   PDB(kOutput,1) Info("StoreOutput","Leave");
}

//______________________________________________________________________________
TList *TProofPlayerRemote::MergeFeedback()
{
   PDB(kFeedback,1) Info("MergeFeedback","Enter");

   if ( fFeedbackLists == 0 ) {
      PDB(kFeedback,1) Info("MergeFeedback","Leave (no output)");
      return 0;
   }

   TList *fb = new TList;   // collection of feedback object

   TIter next(fFeedbackLists);

   TMap *map;
   while ( (map = (TMap*) next()) ) {
      Long_t offset = 0;

      // turn map into list ...

      TList *list = new TList;
      TIter keys(map);

      while ( TObject *key = keys() ) {
         list->Add(map->GetValue(key));
      }

      // clone first object, remove from list

      TObject *obj = list->First();
      list->Remove(obj);
      obj = obj->Clone();
      fb->Add(obj);

      if ( list->IsEmpty() ) {
         delete list;
         continue;
      }

      // merge list with clone
      // direct CINT, also possible via TInterpreter?
      G__ClassInfo ci(obj->ClassName());
      G__CallFunc cf;

      if ( ci.IsValid() )
         cf.SetFuncProto( &ci, "Merge", "TCollection*", &offset);

      if ( cf.IsValid() ) {
         cf.SetArg((Long_t)list);
         cf.Exec(obj);
      } else {
         // No Merge interface, return copy of individual objects
         while ( (obj = list->First()) ) {
            fb->Add(obj->Clone());
            list->Remove(obj);
         }
      }

      delete list;
   }

   PDB(kFeedback,1) Info("MergeFeedback","Leave (%d object(s))", fb->GetSize());

   return fb;
}

//______________________________________________________________________________
void TProofPlayerRemote::StoreFeedback(TSlave *slave, TList *out)
{
   PDB(kFeedback,1) Info("StoreFeedback","Enter");

   if ( out == 0 ) {
      PDB(kFeedback,1) Info("StoreFeedback","Leave (empty)");
      return;
   }

   if ( !gProof->IsMaster() ) {
      // in client
      Feedback(out);
      delete out;
      return;
   }

   if (fFeedbackLists == 0) {
      PDB(kFeedback,2) Info("StoreFeedback","Create fFeedbackLists");
      fFeedbackLists = new TList;
      fFeedbackLists->SetOwner();
   }

   TIter next(out);
   out->SetOwner(kFALSE);  // take ownership of the contents

   TObject *obj;
   while( (obj = next()) ) {
      PDB(kFeedback,2) Info("StoreFeedback","Find '%s'", obj->GetName() );

      TMap *map = (TMap*) fFeedbackLists->FindObject(obj->GetName());
      if ( map == 0 ) {
         PDB(kFeedback,2) Info("StoreFeedback","Map not Found (creating)", obj->GetName() );
         map = new TMap;
         map->SetName( obj->GetName() );
         // TODO: needed? allowed? map->SetOwner();
         fFeedbackLists->Add(map);
      }

      map->Remove(slave);
      map->Add(slave, obj);
   }

   delete out;
   PDB(kFeedback,1) Info("StoreFeedback","Leave");
}

//______________________________________________________________________________
void TProofPlayerRemote::SetupFeedback()
{
   if (!gProof->IsMaster()) return; // Client does not need timer

   TList *fb = (TList*) fInput->FindObject("FeedbackList");

   PDB(kFeedback,1) Info("SetupFeedback","\"FeedbackList\" %sfound",
      fb == 0 ? "NOT ":"");

   if (fb == 0) return;

   // OK, feedback was requested, setup the timer

   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(500,kFALSE);
}

//______________________________________________________________________________
void TProofPlayerRemote::StopFeedback()
{
   if (fFeedbackTimer == 0) return;

   PDB(kFeedback,1) Info("StopFeedback","Stop Timer");

   delete fFeedbackTimer; fFeedbackTimer = 0;
}

//______________________________________________________________________________
Bool_t TProofPlayerRemote::HandleTimer(TTimer *)
{
   PDB(kFeedback,2) Info("HandleTimer","Entry");

   if ( fFeedbackTimer == 0 ) return kFALSE; // timer already switched off

   if ( fFeedbackLists == 0 ) return kFALSE;

   TList *fb = MergeFeedback();

   PDB(kFeedback,2) Info("HandleTimer","Sending %d objects", fb->GetSize());

   TMessage m(kPROOF_FEEDBACK);
   m << fb;

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   delete fb;
   return kFALSE; // ignored?
}

//______________________________________________________________________________
TDSetElement *TProofPlayerRemote::GetNextPacket(TSlave *slave, TMessage *r)
{
   TDSetElement *e = fPacketizer->GetNextPacket( slave, r );

   if ( e != 0 ) {
      PDB(kPacketizer,2)
         Info("GetNextPacket","To slave-%d (%s): '%s' '%s' '%s' %lld %lld",
              slave->GetOrdinal(), slave->GetName(), e->GetFileName(),
              e->GetDirectory(), e->GetObjName(), e->GetFirst(), e->GetNum());
   } else {
      PDB(kPacketizer,2) Info("GetNextPacket","Done");
   }

   return e;
}


//------------------------------------------------------------------------------


ClassImp(TProofPlayerSlave)


//______________________________________________________________________________
TProofPlayerSlave::TProofPlayerSlave()
{
  fSocket = 0;
  fFeedback = 0;
}

//______________________________________________________________________________
TProofPlayerSlave::TProofPlayerSlave(TSocket *socket)
{
      fSocket = socket;
      fFeedback = 0;
}

//______________________________________________________________________________
void TProofPlayerSlave::SetupFeedback()
{
   //
   TList *fb = (TList*) fInput->FindObject("FeedbackList");

   PDB(kFeedback,1) Info("SetupFeedback","\"FeedbackList\" %sfound",
      fb == 0 ? "NOT ":"");

   if (fb == 0) return;

   // OK, feedback was requested, setup the timer

   fFeedbackTimer = new TTimer;
   fFeedbackTimer->SetObject(this);
   fFeedbackTimer->Start(500,kFALSE);

   fFeedback = fb;

}

//______________________________________________________________________________
void TProofPlayerSlave::StopFeedback()
{
   if (fFeedbackTimer == 0) return;

   PDB(kFeedback,1) Info("StopFeedback","Stop Timer");

   fFeedbackTimer->Stop();
   delete fFeedbackTimer;
   fFeedbackTimer = 0;
}

//______________________________________________________________________________
Bool_t TProofPlayerSlave::HandleTimer(TTimer *)
{
   PDB(kFeedback,2) Info("HandleTimer","Entry");

   if ( fFeedback == 0 ) return kFALSE;

   TList *fb = new TList;
   fb->SetOwner(kFALSE);

   if (fOutput == 0) {
      fOutput = fSelector->GetOutputList();
   }

   if (fOutput) {
      TIter next(fFeedback);
      while( TObjString *name = (TObjString*) next() ) {
         // TODO: find object in memory ... maybe allow only in fOutput ?
         TObject *o = fOutput->FindObject(name->GetName());
         if (o != 0) fb->Add(o);
      }
   }

   PDB(kFeedback,2) Info("HandleTimer","Sending %d objects", fb->GetSize());

   TMessage m(kPROOF_FEEDBACK);
   m << fb;

   // send message to client;
   gProofServ->GetSocket()->Send(m);

   delete fb;
   return kFALSE; // ignored?
}


//______________________________________________________________________________
Int_t TProofPlayerSlave::DrawSelect(TDSet * /*set*/, const char * /*varexp*/,
                               const char * /*selection*/, Option_t * /*option*/,
                               Long64_t /*nentries*/, Long64_t /*firstentry*/)
{
   MayNotUse("DrawSelect");

   return -1;
}
