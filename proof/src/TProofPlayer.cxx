// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.6 2002/06/13 15:13:21 rdm Exp $
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
#include "TPacketizer.h"
#include "TSelector.h"
#include "TProof.h"
#include "TROOT.h"
#include "TError.h"
#include "MessageTypes.h"
#include "TMessage.h"
#include "TDSetProxy.h"
#include "TString.h"
#include "TSystem.h"
#include "TFile.h"

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

   fInput    = new TList;
   fOutput   = 0;
   fSelector = 0;
}

//______________________________________________________________________________
TProofPlayer::~TProofPlayer()
{
   delete fInput;
   delete fSelector;
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
void TProofPlayer::StoreOutput(TList *out)
{
   Fatal("StoreOutput", "this method must be overridden!");
}


//______________________________________________________________________________
Int_t TProofPlayer::Process(TDSet *dset, const char *selector_file,
                                 Long64_t nentries, Long64_t first,
                                 TEventList *evl)
{
   Info("Process","Enter");

   fOutput = 0; delete fSelector;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   fSelector->SetInputList(fInput);

   dset->Reset();

   TEventIter *evIter = TEventIter::Create(dset, fSelector, first, nentries);

   if ( gDebug > 2 )
      Info("Process","Call Begin");

   fSelector->Begin( 0 );  // Init is called explicitly from GetNextEvent()

   if ( gDebug > 2 )
      Info("Process","Loop over Process()");

   // Loop over range
   Long64_t entry;
   while ((entry = evIter->GetNextEvent()) >= 0) {

      if ( gDebug > 3 )
         Info("Process","Call Process(%ld)", entry);

      Bool_t stop = fSelector->Process(entry);
      if (stop) {}  // remove unused warning

      if (gROOT->IsInterrupted()) break;
   }

   // Finalize
   if ( gDebug > 2 )
      Info("Process","Call Terminate");
   fSelector->Terminate();

   fOutput = fSelector->GetOutputList();

   return 0;
}


//______________________________________________________________________________
void TProofPlayer::UpdateAutoBin(const char *name, Double_t& xmin, Double_t& xmax,
                Double_t& ymin, Double_t& ymax, Double_t& zmin, Double_t& zmax)
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
TDSetElement *TProofPlayer::GetNextPacket(TSlave *slave)
{
   MayNotUse("GetNextPacket");
   return 0;
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
}


//______________________________________________________________________________
TProofPlayerRemote::~TProofPlayerRemote()
{
   delete fOutput;      // owns the output list
   delete fOutputLists;
}


//______________________________________________________________________________
Int_t TProofPlayerRemote::Process(TDSet *dset, const char *selector_file,
                                  Long64_t nentries, Long64_t first,
                                  TEventList *evl)
{
   // Process specified TDSet on PROOF.
   // Returns -1 in case error, 0 otherwise.

Info("Process","---- Start ----");

   TString filename = selector_file;
   filename = filename.Strip(TString::kTrailing,'+');
Info("Process", "Sendfile: %s", filename.Data() );
   fProof->SendFile(filename);

   if ( filename.EndsWith(".C") ) {
         filename.ReplaceAll(".C",".h");
Info("Process", "Sendfile: %s", filename.Data() );
         fProof->SendFile(filename);
   }

#if 0
   ... compiling will get private commands ...
   TString libname = filename;
   Ssiz_t dot_pos = libname.Last('.');
   if ( dot_pos >= 0 )
      libname.Replace(dot_pos,1,"_");
   Ssiz_t slash_pos = libname.Last('/');  // Unix specfic !!
   // G__loadfile seems to need this
   if ( slash_pos < 0 )
      libname.Prepend("./");              // Unix specfic !!
   libname.Append(".");
   libname.Append(gSystem->GetSoExt());


   TString compileCmd("gSystem->CompileMacro(\"");
   compileCmd += filename;
   compileCmd += "\",\"f\",\"";
   compileCmd += libname;
   compileCmd += "\")";
   if ( fProof->Exec(compileCmd, TProof::kUnique) == -1 ) {
      Warning("Process","Compile failed");
      return -1;
   }

   TString unloadCmd("G__unloadfile(\"");
   unloadCmd += libname;
   unloadCmd +=  "\")";
   fProof->Exec(unloadCmd, TProof::kActive);

   TString loadCmd("G__loadfile(\"");
   loadCmd += libname;
   loadCmd +=  "\")";
   fProof->Exec(loadCmd, TProof::kActive);
#endif

   TMessage mesg(kPROOF_PROCESS);
   TString fn(selector_file);

   TDSet *set = dset;
   if ( fProof->IsMaster() ) {
Info("Process","Create Proxy DSet");
      set = new TDSetProxy( dset->GetType(), dset->GetObjName(),
                        dset->GetDirectory() );

      delete fPacketizer;
      fPacketizer = new TPacketizer(dset, fProof->GetListOfActiveSlaves(),
                                 first, nentries);
   }

   mesg << set << fn << fInput << nentries << first; // no evl yet
Info("Process","Broadcast");

   fProof->Broadcast(mesg);
Info("Process","Collect");

   fProof->SetPlayer(this);  // Fix SetPlayer to release current player

   fProof->Collect();

Info("Process","Calling Merge Output");
   MergeOutput();

   return 0;
}


//______________________________________________________________________________
void TProofPlayerRemote::MergeOutput()
{
Info("MergeOutput","Enter");
   delete fOutput;
   fOutput = new THashList;

   if ( fOutputLists == 0 ) {
      Info("MergeOutput","Leave (empty)");
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
Info("MergeOutput","Leave");
}


//______________________________________________________________________________
void TProofPlayerRemote::StoreOutput(TList *out)
{
Info("StoreOutput","Enter");

   if ( out == 0 ) {
Info("StoreOutput","Leave (empty)");
      return;
   }

   TIter next(out);

   if (fOutputLists == 0) {
Info("StoreOutput","Create fOutputLists");
      fOutputLists = new TList;
      fOutputLists->SetOwner();
   }

   TObject *obj;
   while( (obj = next()) ) {
Info("StoreOutput","Find '%s'", obj->GetName() );
fOutputLists->Print();
      TList *list = (TList *) fOutputLists->FindObject( obj->GetName() );
      if ( list == 0 ) {
Info("StoreOutput","List not Found", obj->GetName() );
         list = new TList;
         list->SetName( obj->GetName() );
         list->SetOwner();
         fOutputLists->Add( list );
      }
      list->Add( obj );
   }

   out->SetOwner(kFALSE);  // Needed??
   delete out;
Info("StoreOutput","Leave");
}



//______________________________________________________________________________
TDSetElement *TProofPlayerRemote::GetNextPacket(TSlave *slave)
{
   TDSetElement *e = fPacketizer->GetNextPacket( slave );

   if ( e != 0 ) {
      Info("GetNextPacket","'%s' '%s' '%s' %d %d", e->GetFileName(),
            e->GetDirectory(), e->GetObjName(),e->GetFirst(),e->GetNum());
   } else {
      Info("GetNextPacket","Done");
   }

   return e;
}


//------------------------------------------------------------------------------


ClassImp(TProofPlayerSlave)


//______________________________________________________________________________
TProofPlayerSlave::TProofPlayerSlave()
{
}


//______________________________________________________________________________
TProofPlayerSlave::TProofPlayerSlave(TSocket *socket)
{
      fSocket = socket;
}

