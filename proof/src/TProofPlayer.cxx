// @(#)root/proof:$Name:  $:$Id: TProofPlayer.cxx,v 1.3 2002/03/13 01:52:21 rdm Exp $
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

//______________________________________________________________________________
void TProofPlayer::StoreOutput(TList *out)
{
   Fatal("StoreOutput", "this method must be overridden!");
}


//______________________________________________________________________________
Int_t TProofPlayer::Process(TDSet *dset, const char *selector_file,
                                 Int_t nentries, Int_t first,
                                 TEventList *evl)
{
   Info("Process","Enter");

   delete fSelector;
   fSelector = TSelector::GetSelector(selector_file);

   if ( !fSelector ) {
      Error("Process", "Cannot load: %s", selector_file );
      return -1;
   }

   fSelector->SetInputList(fInput);

   // fSelector->  0 /* tree */ );   // TODO: the init logic needs to be changed
   dset->Reset();


   TEventIter *evIter = 0;
   TFile      *finp = 0;
   TString     filename;
   TDirectory *dir = 0;
   TString     path;
   TDSetElement *e;
   TString     objName;
   Bool_t      once = kTRUE;

   while ( (e = dset->Next()) ) {

      // Check Filename
      if ( finp == 0 || filename != e->GetFileName() ) {
         delete evIter; evIter = 0;
         if ( dir != finp ) { delete dir; } // dir != top-level
         dir = 0;
         delete finp; finp = 0;
         path = "";

         filename = e->GetFileName();
         finp = TFile::Open(filename);

         if ( finp->IsZombie() ) {
            Error("Process","Cannot open file: %s (%s)",
               filename.Data(), strerror(finp->GetErrno()) );
            // cleanup ?
            return -1;
         }
         Info("Process","Opening file: %s", filename.Data() );
      }

      // Check Directory
      if ( dir == 0 || path != e->GetDirectory() ) {
         TDirectory *dirsave = gDirectory;
         delete evIter; evIter = 0;
         delete dir;

         path = e->GetDirectory();
         if ( ! finp->cd(path) ) {
            Error("Process","Cannot cd to: %s",
               path.Data() );
            return -1;
         }
         Info("Process","Cd to: %s", path.Data() );
         dir = gDirectory;

         dirsave->cd();
      }


      // Check Objectname :-/
      if ( objName != e->GetObjName() ) {
         delete evIter; evIter = 0;
         objName = e->GetObjName();
      }

      // New TEventIter?
      if ( evIter == 0 ) {
         evIter = TEventIter::Create(dset, dir, fSelector);
         if ( evIter == 0 ) {
            return -1;
         }
      }

      if ( !evIter->InitRange( e->GetFirst(), e->GetNum() ) ) {
         return -1;
      }

      // Loop over range

      while (evIter->GetNextEvent()) {

         if ( once ) {
Info("Process","Call Begin");

            fSelector->Begin( /* need to change API */ 0);
            once = kFALSE;
         }

Info("Process","Call Process");
         Bool_t stop = fSelector->Process();
         if (stop) {}  // remove unused warning

         if (gROOT->IsInterrupted()) break;
      }
   }

   // Finalize
   fSelector->Terminate();

   fOutput = fSelector->GetOutputList();

   return 0;
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
   fSet           = 0;
   fElem          = 0;
}


//______________________________________________________________________________
TProofPlayerRemote::~TProofPlayerRemote()
{
   delete fOutput;      // owns the output list
   delete fOutputLists;
}


//______________________________________________________________________________
Int_t TProofPlayerRemote::Process(TDSet *dset, const char *selector_file,
                                  Int_t nentries, Int_t first,
                                  TEventList *evl)
{

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

   fSet = dset;
   TDSet *set = dset;
   if ( fProof->IsMaster() ) {
Info("Process","Create Proxy DSet");
      set = new TDSetProxy( dset->GetType(), dset->GetObjName(),
                        dset->GetDirectory() );
   }

   mesg << set << fn << fInput << nentries << first; // no evl yet

Info("Process","Broadcast");

   fProof->Broadcast(mesg);
Info("Process","Collect");

   fProof->SetPlayer(this);  // Fix SetPlayer to release current player

   fPacketizer = new TPacketizer(dset, fProof->GetListOfActiveSlaves());

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

