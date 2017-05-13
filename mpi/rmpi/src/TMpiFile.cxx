#include<Mpi/TMpiFile.h>
#include<TKey.h>
#include<TTree.h>
#include<TThread.h>
#include <iostream>
#include <fstream>
#include<vector>
using namespace ROOT::Mpi;

//______________________________________________________________________________
//utility function to read all bytes from a file
static std::vector<Char_t> ReadBytes(const Char_t  *name)
{
   std::ifstream file(name, std::ios::binary | std::ios::ate);
   std::ifstream::pos_type position = file.tellg();
   std::vector<Char_t>  data(position);
   file.seekg(0, std::ios::beg);
   file.read(&data[0], position);
   return data;
}

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
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option, const Char_t *ftitle, Int_t compress): TMemFile(name, buffer, size, option, ftitle, compress), fComm(comm), fSync(kFALSE)
{
}


//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle, Int_t compress): TMemFile(name, option, ftitle, compress), fComm(comm), fSync(kFALSE)
{
}

//______________________________________________________________________________
TMpiFile::TMpiFile(const TMpiFile &file): TMemFile(file)
{
   fComm = file.fComm;
   fMerger = file.fMerger;
   fSync = file.fSync;
   fSyncType = file.fSyncType;
   fDiskOpenMode = file.fDiskOpenMode;
}

TMpiFile *TMpiFile::Open(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle, Int_t compress)
{
   TMpiFile *file = NULL;

   TString fOption = option;
   fOption.ToUpper();
   Bool_t create   = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update   = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read     = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read    = kTRUE;
      fOption = "READ";
   }

   if (create || recreate) {
      if (comm.IsMainProcess()) {
         auto tfile = TFile::Open(name, option, ftitle, compress);
         if (!tfile) {
            if (create && !gSystem->AccessPathName(name, kFileExists)) comm.Abort(ERR_FILE_EXISTS);
            else comm.Abort(ERR_FILE);
         }
         tfile->Close();
         delete tfile;
      }
      file = new TMpiFile(comm, name, option, ftitle, compress);
   } else {

      auto tfile = TFile::Open(name, option, ftitle, compress);
      if (!tfile) {
         comm.Abort(ERR_FILE);
      }
      tfile->Close();
      delete tfile;

      auto buffer = ReadBytes(name);
      file = new TMpiFile(comm, name, &buffer[0], buffer.size(), option, ftitle, compress);
   }
   if (file) file->fDiskOpenMode = fOption;
   return file;
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
      const Char_t *classname = key->GetClassName();
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
      // We want gDirectory untouched by anything going on here
      TDirectory::TContext ctxt;

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

// //______________________________________________________________________________
// void TMpiFile::SyncSave(Int_t type)
// {
//    auto file =TFile::Open(TFile::AsyncOpen(GetName(), "UPDATE", "", GetCompressionLevel()));
//    CopyFrom(this, (TMpiFile *)file);
//    file->Close();
//    delete file;
// }



// void TMpiFile::SyncReCreate(const TIntraCommunicator &comm, Int_t rank, const Char_t *name, TString option, Int_t &SyncMakeZombie)
// {
//    SyncMakeZombie = kFALSE;
//    if (comm.GetRank() == rank) {
//       if (option == "CREATE") {
//          if (gSystem->AccessPathName(name, kFileExists)) {
//             comm.Error("TMpiFile", "file %s already exists", name);
//             SyncMakeZombie = kTRUE;
//          } else {
//             if (gSystem->Unlink(name) != 0) {
//                comm.SysError("TMpiFile", "could not delete %s (errno: %d)", name, gSystem->GetErrno());
//                SyncMakeZombie = kTRUE;
//             }
//          }
//       }
//       if (option == "RECREATE") {
//          if (!gSystem->AccessPathName(name, kFileExists)) {
//             if (gSystem->Unlink(name) != 0) {
//                comm.SysError("TMpiFile", "could not delete %s (errno: %d)", name, gSystem->GetErrno());
//                SyncMakeZombie = kTRUE;
//             }
//          }
//       }
//    }
// //    fComm.Bcast(fSyncMakeZombie,rank);
// //    if(fSyncMakeZombie)
// //    {
// //       MakeZombie();
// //       gDirectory = gROOT;
// //    }
// }


//______________________________________________________________________________
void TMpiFile::Save(Int_t type)
{
   fSync = kTRUE;
   fSyncType = type;
}


//______________________________________________________________________________
void TMpiFile::Merge(Int_t type)
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
      TDirectory::TContext ctxt;
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

   delete mpifile;
   if (fComm.GetRank() == 0) {
      delete fMerger;
   }
}

//______________________________________________________________________________
void TMpiFile::Sync()
{
   Int_t dummy;
   if (fComm.GetRank() != 0) {
      fComm.Recv(dummy, fComm.GetRank() - 1, 0);
      if (fSync) {
         printf("--rank %d--\n", fComm.GetRank());
         SyncSave(fSyncType);
         fSync = kFALSE;
      }
   }
   fComm.Send(dummy, (fComm.GetRank() + 1) % fComm.GetSize(), 0);

   if (fComm.GetRank() == 0) {
      fComm.Recv(dummy, fComm.GetSize() - 1, 0);
      if (fSync) {
         printf("--rank %d--\n", fComm.GetRank());
         SyncSave(fSyncType);
         fSync = kFALSE;
      }
   }

}

//______________________________________________________________________________
void TMpiFile::SyncSave(Int_t type)
{
   Write();
   fMessage.Reset(kMESS_ANY);
   fMessage.SetWriteMode();
   CopyTo(fMessage);
   fMerger = new TMpiFileMerger(kFALSE, kFALSE);
   fMerger->Reset();
   fMerger->SetPrintLevel(0);
   fMerger->OutputFile(GetName(), "UPDATE");
   fMessage.SetReadMode();
   fMessage.Reset(kMESS_ANY);
   TDirectory::TContext ctxt;
   TMemFile *memffile  = new TMemFile(GetName(), fMessage.Buffer() + fMessage.Length(), GetEND(), "UPDATE");
   fMessage.SetBufferOffset(fMessage.Length() + GetEND());
   fMerger->AddAdoptFile(memffile);
   memffile = 0;
   fMerger->PartialMerge(type);
//    fMerger->Reset();
   delete fMerger;
//    ls();
//    printf("--rank %d--\n",fComm.GetRank());
}

