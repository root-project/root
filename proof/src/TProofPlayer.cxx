// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.19 2003/03/04 17:09:41 rdm Exp $
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

   fAutoBins = 0;
   fInput    = new TList;
   fOutput   = 0;
   fSelector = 0;
   fFeedbackTimer = 0;
}

//______________________________________________________________________________
TProofPlayer::~TProofPlayer()
{
   delete fInput;
   delete fSelector;
   delete fFeedbackTimer;
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
   PDB(kGlobal,1) Info("Progress","%2f (%ld/%ld)", 100.*processed/total, processed, total);

   Long_t parm[2];
   parm[0] = total;
   parm[1] = processed;
   Emit("Progress(Long64_t,Long64_t)", parm);

   gProof->Progress(total,processed);
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

   fOutput = 0; delete fSelector;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   SetupFeedback();

   fSelector->SetOption(option);
   fSelector->SetInputList(fInput);

   dset->Reset();

   TEventIter *evIter = TEventIter::Create(dset, fSelector, first, nentries);

   PDB(kLoop,1) Info("Process","Call Begin(0)");

   fSelector->Begin( 0 );  // Init is called explicitly from GetNextEvent()

   PDB(kLoop,1) Info("Process","Looping over Process()");

   // Loop over range
   Long64_t entry;
   while ((entry = evIter->GetNextEvent()) >= 0) {

      PDB(kLoop,3)Info("Process","Call Process(%ld)", entry);

      Bool_t stop = fSelector->Process(entry);
      if (stop) {}  // remove unused warning

      gSystem->DispatchOneEvent(kTRUE);

      if (gROOT->IsInterrupted()) break;
   }

   StopFeedback();

   // Finalize
   PDB(kLoop,1) Info("Process","Call Terminate");

   fSelector->Terminate();

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
   MayNotUse("DrawSelect");
   return -1;
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

   TString filename = selector_file;

   // If the filename does not contain "." assume class is compiled in
   if ( strchr(selector_file,'.') != 0 ) {
      filename = filename.Strip(TString::kTrailing,'+');

      PDB(kSelector,1) Info("Process", "Sendfile: %s", filename.Data() );
      fProof->SendFile(filename);

      if ( filename.EndsWith(".C") ) {
         filename.ReplaceAll(".C",".h");
         PDB(kSelector,1) Info("Process", "Sendfile: %s", filename.Data() );
         fProof->SendFile(filename);
      }
   }

   TMessage mesg(kPROOF_PROCESS);
   TString fn(gSystem->BaseName(selector_file));

   TDSet *set = dset;
   if ( fProof->IsMaster() ) {

      PDB(kPacketizer,1) Info("Process","Create Proxy TDSet");
      set = new TDSetProxy( dset->GetType(), dset->GetObjName(),
                        dset->GetDirectory() );

      delete fPacketizer;
//      fPacketizer = new TPacketizer(dset, fProof->GetListOfActiveSlaves(),
//                                    first, nentries);
      fPacketizer = new TPacketizer2(dset, fProof->GetListOfActiveSlaves(),
                                     first, nentries);

      if ( !fPacketizer->IsValid() ) {
         return -1;
      }
   }

   SetupFeedback();

   TString opt = option;
   mesg << set << fn << fInput << opt << nentries << first; // no evl yet

   PDB(kGlobal,1) Info("Process","Calling Broadcast");
   fProof->Broadcast(mesg);

   PDB(kGlobal,1) Info("Process","Calling Collect");
   fProof->SetPlayer(this);  // Fix SetPlayer to release current player
   fProof->Collect();

   StopFeedback();

   PDB(kGlobal,1) Info("Process","Calling Merge Output");
   MergeOutput();

   return 0;
}

//______________________________________________________________________________
Int_t TProofPlayerRemote::DrawSelect(TDSet *set, const char *varexp,
                                     const char *selection, Option_t *option,
                                     Long64_t nentries, Long64_t firstentry)
{
   Info("DrawSelect","Not implemented");
   return 0;

   // Analyze options


   // Process query


   // Display results
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
         Info("GetNextPacket","To slave-%d (%s): '%s' '%s' '%s' %d %d",
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
   fFeedback = 0;
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
