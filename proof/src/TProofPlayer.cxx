// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.1 2002/01/18 14:24:09 rdm Exp $
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
#include "TSelector.h"
#include "TProof.h"
#include "TROOT.h"
#include "TError.h"
#include "MessageTypes.h"
#include "TMessage.h"
#include "TDSet.h"
#include "TString.h"
#include "TSystem.h"

#include "Api.h"



//------------------------------------------------------------------------------


ClassImp(TProofPlayer)

//______________________________________________________________________________
TProofPlayer::TProofPlayer()
{
   // Default ctor.

   fInput    = new THashList;
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
   return fOutput->FindObject(name);
}

//______________________________________________________________________________
TList *TProofPlayer::GetOutputList() const
{
   return fOutput;
}

void TProofPlayer::StoreOutput(TList *out)
{
   Fatal("TProofPlayer::StoreOutput", "this method must be overridden!");
}
//------------------------------------------------------------------------------


ClassImp(TProofPlayerLocal)

//______________________________________________________________________________
Int_t TProofPlayerLocal::Process(TDSet *dset, const char *selector_file,
                                 Int_t nentries, Int_t first,
                                 TEventList *evl)
{
   ::Info("TProofPlayerLocal::Process","Voila!");

   // create TSelector
   delete fSelector; fSelector = 0;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   // create TEventIter
   TEventIter *evIter = new TEventIterLocal(dset);

   // Init
   fSelector->SetInputList(fInput);

   evIter->Init(fSelector);

   // Loop
   while (evIter->GetNextEvent(fSelector)) {

      Bool_t stop = fSelector->Process();
      if (stop) {}  // remove unused warning

      if (gROOT->IsInterrupted()) break;

   }

   // Finalize
   fSelector->Terminate();

   fOutput = fSelector->GetOutputList();

   return 0;
}


//------------------------------------------------------------------------------

ClassImp(TProofPlayerRemote)


//______________________________________________________________________________
TProofPlayerRemote::TProofPlayerRemote(TProof *proof) : fProof(proof)
{
}


//______________________________________________________________________________
TProofPlayerRemote::~TProofPlayerRemote()
{
   delete fOutput;
}


//______________________________________________________________________________
Int_t TProofPlayerRemote::Process(TDSet *set, const char *selector_file,
                                  Int_t nentries, Int_t first,
                                  TEventList *evl)
{

::Info("TProofPlayerRemote::Process","Voila!");

   TString filename = selector_file;
   filename = filename.Strip(TString::kTrailing,'+');
   fProof->SendFile(filename);

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
      ::Warning("TProofPlayerRemote::Process","Compile failed");
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

   TMessage mesg(kPROOF_PROCESS);
   TString fn(selector_file);

   mesg << set << fn << fInput << nentries << first; // no evl yet
::Info("TProofPlayerRemote::Process","Broadcast");

   fProof->Broadcast(mesg);
::Info("TProofPlayerRemote::Process","Collect");

   fProof->SetPlayer(this);  // Fix SetPlayer to release current player
   fProof->Collect();


::Info("TProofPlayerRemote::Process","Calling Merge Output");
   MergeOutput();

   return 0;
}


//______________________________________________________________________________
void TProofPlayerRemote::MergeOutput()
{
   delete fOutput;
   fOutput = new THashList;

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
}


//______________________________________________________________________________
void TProofPlayerRemote::StoreOutput(TList *out)
{
::Info("TProofPlayerRemote::StoreOutput","Enter");
   TIter next(out);

   if (fOutputLists == 0) {
::Info("TProofPlayerRemote::StoreOutput","Create fOutputLists");
      fOutputLists = new TList;
      fOutputLists->SetOwner();
   }

   TObject *obj;
   while( (obj = next()) ) {
::Info("TProofPlayerRemote::StoreOutput","Find '%s'", obj->GetName() );
fOutputLists->Print();
      TList *list = (TList *) fOutputLists->FindObject( obj->GetName() );
      if ( list == 0 ) {
::Info("TProofPlayerRemote::StoreOutput","List not Found", obj->GetName() );
         list = new TList;
         list->SetName( obj->GetName() );
         list->SetOwner();
         fOutputLists->Add( list );
      }
      list->Add( obj );
   }

   out->SetOwner(kFALSE);  // Needed??
   delete out;
::Info("TProofPlayerRemote::StoreOutput","Done");
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


//______________________________________________________________________________
Int_t TProofPlayerSlave::Process(TDSet *dset, const char *selector_file,
                                 Int_t nentries, Int_t first,
                                 TEventList *evl)
{
   Info("TProofPlayerSlave::Process","Voila!");

   // create TSelector
   delete fSelector; fSelector = 0;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   // create TEventIter
   TEventIter *evIter = new TEventIterSlave(fSocket);

   // Init
   fSelector->SetInputList(fInput);

   evIter->Init(fSelector);

   // Loop
   while (evIter->GetNextEvent(fSelector)) {

      Bool_t stop = fSelector->Process();
      if (stop) {}  // remove unused warning

      if (gROOT->IsInterrupted()) break;

   }

   // Finalize
   fSelector->Terminate();

   fOutput = fSelector->GetOutputList();
   if ( fOutput == 0 ) {
Info("TProofPlayerSlave::Process","No selector output list??? creating one");
      fOutput = new TList;
   } else if ( !strcmp(fOutput->ClassName(),"TList") ) {
Info("TProofPlayerSlave::Process","Selector output is a %s ??? creating TList", fOutput->ClassName() );
      fOutput = new TList;
   }

   return 0;
}

