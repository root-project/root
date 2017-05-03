#include<Mpi/TMpiFile.h>
#include<TKey.h>
#include<TTree.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
Bool_t TMpiFileMerger::OutputMemFile(const char *outputfile, const char *mode, Int_t compressionLevel)
{
   fExplicitCompLevel = kTRUE;

   TFile *oldfile = fOutputFile;
   fOutputFile = 0; // This avoids the complaint from RecursiveRemove about the file being deleted which is here spurrious. (see RecursiveRemove).
   SafeDelete(oldfile);

   fOutputFilename = outputfile;

   // We want gDirectory untouched by anything going on here
   TDirectory::TContext ctxt;
   fOutputFile = new  TMemFile(outputfile, mode, "", compressionLevel);
   if (!(fOutputFile) || fOutputFile->IsZombie()) {
      Error("OutputMemFile", "cannot open the sync files %s", fOutputFilename.Data());
      return kFALSE;
   }
   return kTRUE;

}

//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option, const Char_t *ftitle, Int_t compress): TMemFile(name, buffer, size, option, ftitle, compress), fComm(comm)
{

}


//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle, Int_t compress): TMemFile(name, option, ftitle, compress), fComm(comm)
{
   TMessage::EnableSchemaEvolutionForAll(kTRUE);
}

//______________________________________________________________________________
void TMpiFile::CopyFrom(TDirectory *source, TMpiFile *file)
{
   TMpiFile *savdir = file;
   TDirectory *adir = savdir;
   adir->cd();
   //loop on all entries of this directory
   TKey *key;
   TIter nextkey(source->GetListOfKeys());
   while ((key = (TKey *)nextkey())) {
      const char *classname = key->GetClassName();
      TClass *cl = gROOT->GetClass(classname);
      if (!cl) continue;
      if (cl->InheritsFrom(TDirectory::Class())) {
         source->cd(key->GetName());
         TDirectory *subdir = file;
         adir->cd();
         CopyFrom(subdir, file);
         adir->cd();
      } else if (cl->InheritsFrom(TTree::Class())) {
         TTree *T = (TTree *)source->Get(key->GetName());
         adir->cd();
         TTree *newT = T->CloneTree(-1, "fast");
         newT->Write();
      } else {
         source->cd();
         TObject *obj = key->ReadObj();
         adir->cd();
         obj->Write();
         delete obj;
      }
   }
   adir->SaveSelf(kTRUE);
   savdir->cd();
}


//______________________________________________________________________________
/// The type is defined by the bit values in EPartialMergeType:
///   kRegular      : normal merge, overwritting the output file
///   kIncremental  : merge the input file with the content of the output file (if already exising) (default)
///   kAll          : merge all type of objects (default)
///   kResetable    : merge only the objects with a MergeAfterReset member function.
///   kNonResetable : merge only the objects without a MergeAfterReset member function.

void TMpiFile::Merge(Int_t root, Int_t type)
{
   Write();
   fMessage.Reset(kMESS_ANY);
   fMessage.WriteTString(GetName());
   fMessage.WriteLong64(GetEND());
   CopyTo(fMessage);

   TMpiMessage *msgs = NULL;
   if (fComm.GetRank() == root) {
      msgs = new TMpiMessage[fComm.GetSize()];
   }

   fComm.Gather(&fMessage, 1, msgs, fComm.GetSize(), root);
   if (fComm.GetRank() == root) {
      fMerger = new TMpiFileMerger(kFALSE, kFALSE);
      fMerger->SetPrintLevel(0);
      fMerger->OutputFile(GetName(), "RECREATE");
      for (auto i = 0; i < fComm.GetSize(); i++) {
         Long64_t length = 0;
         TString filename;
         msgs[i].SetReadMode();
         msgs[i].Reset(kMESS_ANY);
         msgs[i].ReadTString(filename);
         msgs[i].ReadLong64(length);
         TMemFile *memffile  = new TMemFile(filename, msgs[i].Buffer() + msgs[i].Length(), length, "UPDATE");
         msgs[i].SetBufferOffset(msgs[i].Length() + length);
         fMerger->AddAdoptFile(memffile);
         memffile = 0;
      }
      fMerger->PartialMerge(type);
      delete fMerger;
   }
}

//______________________________________________________________________________
//save all files from memory to disk from all processes
void TMpiFile::Save(Int_t type)
{
   auto file = TFile::Open(GetName(), "UPDATE", "", GetCompressionLevel());
   CopyFrom(this, (TMpiFile *)file);
   file->Close();
   delete file;

//     fComm.Barrier();
//     //TODO: I need to put a lock here do it in sequence and dont over write
//    Write();
//    fMessage.Reset(kMESS_ANY);
//    CopyTo(fMessage);
//
//    fMerger = new TMpiFileMerger(kFALSE, kFALSE);
//    fMerger->SetPrintLevel(0);
//    fMerger->OutputMemFile(GetName(), "UPDATE");
//
//    fMessage.SetReadMode();
//    fMessage.Reset(kMESS_ANY);
//    TMemFile *memffile  = new TMemFile(GetName(), fMessage.Buffer() + fMessage.Length(), GetEND(), "UPDATE");
//    fMessage.SetBufferOffset(fMessage.Length() + GetEND());
//    fMerger->AddAdoptFile(memffile);
//    memffile = 0;
//
//    fMerger->PartialMerge(type);
//    delete fMerger;
}


//______________________________________________________________________________
//method to synchronize all TMpiFile content in all process of a given TIntraCommunicator
void TMpiFile::Sync(Int_t type)
{
   Write();
   fMessage.Reset(kMESS_ANY);
   fMessage.WriteTString(GetName());
   fMessage.WriteLong64(GetEND());
   CopyTo(fMessage);

   TMpiMessage *msgs = NULL;
   if (fComm.GetRank() == 0) {
      msgs = new TMpiMessage[fComm.GetSize()];
   }

   fComm.Gather(&fMessage, 1, msgs, fComm.GetSize(), 0);
   if (fComm.GetRank() == 0) {
      fMerger = new TMpiFileMerger(kFALSE, kFALSE);
      fMerger->SetPrintLevel(0);
      fMerger->OutputMemFile(GetName(), "RECREATE");
      for (auto i = 0; i < fComm.GetSize(); i++) {
         Long64_t length = 0;
         TString filename;
         msgs[i].SetReadMode();
         msgs[i].Reset(kMESS_ANY);
         msgs[i].ReadTString(filename);
         msgs[i].ReadLong64(length);
         TMemFile *memffile  = new TMemFile(filename, msgs[i].Buffer() + msgs[i].Length(), length, "UPDATE");
         msgs[i].SetBufferOffset(msgs[i].Length() + length);
         fMerger->AddAdoptFile(memffile);
         memffile = 0;
      }
      fMerger->PartialMerge(type);
      TMemFile *mfile = dynamic_cast<TMemFile *>(fMerger->GetOutputFile());
//       mfile->ls();
//       TDirectory::TContext ctxt;
//       mfile->Write();
      fMessage.Reset(kMESS_ANY);
      fMessage.SetWriteMode();
      fMessage.WriteLong64(mfile->GetEND());
      mfile->CopyTo(fMessage);
   }

   fComm.Bcast(fMessage, 0); //sending the new data for all processes

   Long64_t length = 0;
   fMessage.SetReadMode();
   fMessage.Reset(kMESS_ANY);
   fMessage.ReadLong64(length);


   TMpiFile *mpifile  = new TMpiFile(fComm, GetName(), fMessage.Buffer() + fMessage.Length(), length, "UPDATE");
   fMessage.SetBufferOffset(fMessage.Length() + length);
   this->Delete("*;*");
   CopyFrom(mpifile, this);

//    this->ls();
   delete mpifile;
   if (fComm.GetRank() == 0) {
      delete fMerger;
   }
}


