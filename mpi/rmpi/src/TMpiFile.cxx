#include <Mpi/TMpiFile.h>
#include <TKey.h>
#include <TTree.h>
#include <TThread.h>
#include <iostream>
#include <fstream>
#include <vector>
using namespace ROOT::Mpi;

//______________________________________________________________________________
// utility function to read all bytes from a file
static std::vector<Char_t> ReadBytes(const Char_t *name)
{
   std::ifstream file(name, std::ios::binary | std::ios::ate);
   std::ifstream::pos_type position = file.tellg();
   std::vector<Char_t> data(position);
   file.seekg(0, std::ios::beg);
   file.read(&data[0], position);
   return data;
}

//______________________________________________________________________________
Bool_t TMpiFileMerger::OutputMemFile(const char *outputfile, const char *mode, Int_t compressionLevel)
{
   fExplicitCompLevel = kTRUE;

   TFile *oldfile = fOutputFile;
   fOutputFile = 0; // This avoids the complaint from RecursiveRemove about the file being deleted which is here
                    // spurrious. (see RecursiveRemove).
   SafeDelete(oldfile);

   fOutputFilename = outputfile;

   // We want gDirectory untouched by anything going on here
   TDirectory::TContext ctxt;
   fOutputFile = new TMemFile(outputfile, mode, "", compressionLevel);
   if (!(fOutputFile) || fOutputFile->IsZombie()) {
      Error("OutputMemFile", "cannot open the sync files %s", fOutputFilename.Data());
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option,
                   const Char_t *ftitle, Int_t compress)
   : TMemFile(name, buffer, size, option, ftitle, compress), fComm(comm)
{
}

//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle,
                   Int_t compress)
   : TMemFile(name, option, ftitle, compress), fComm(comm)
{
}

//______________________________________________________________________________
TMpiFile::TMpiFile(const TMpiFile &file) : TMemFile(file)
{
   fComm = file.fComm;
   fMerger = file.fMerger;
   fDiskOpenMode = file.fDiskOpenMode;
}

//______________________________________________________________________________
/*!
   Method to create a TMpiFile instance, similar to TFile except that this needs a TIntraCommunicator object
   to comunicate the mutiple file along multiple processes.
   \param comm TIntraCommunicator for internal comminication
   \param name File's name
   \param option aperture mode "READ/CREATE/RECREATE/UPDATE"
   \param ftitle optional title for the file
   \param compress compression level
 */
TMpiFile *TMpiFile::Open(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle,
                         Int_t compress)
{
   TMpiFile *file = NULL;

   TString fOption = option;
   fOption.ToUpper();
   Bool_t create = (fOption == "CREATE") ? kTRUE : kFALSE;
   Bool_t recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   Bool_t update = (fOption == "UPDATE") ? kTRUE : kFALSE;
   Bool_t read = (fOption == "READ") ? kTRUE : kFALSE;
   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      fOption = "READ";
   }

   if (create || recreate) {
      if (comm.IsMainProcess()) {
         auto tfile = TFile::Open(name, option, ftitle, compress);
         if (!tfile) {
            if (create && !gSystem->AccessPathName(name, kFileExists))
               comm.Abort(ERR_FILE_EXISTS);
            else
               comm.Abort(ERR_FILE);
         }
         tfile->Close();
         delete tfile;
      }
      file = new TMpiFile(comm, name, option, ftitle, compress);
   } else {
      std::vector<Char_t> buffer;
      if (comm.IsMainProcess()) {

         auto tfile = TFile::Open(name, option, ftitle, compress);
         if (!tfile) {
            comm.Abort(ERR_FILE);
         }
         tfile->Close();
         delete tfile;
         buffer = ReadBytes(name);
      }
      comm.Bcast(buffer, comm.GetMainProcess());
      file = new TMpiFile(comm, name, &buffer[0], buffer.size(), option, ftitle, compress);
   }

   if (file) file->fDiskOpenMode = fOption;
   return file;
}

//______________________________________________________________________________
/*! Method to copy the content one file to other.
  \param src Source file.
  \param file   Destination file.
*/

void TMpiFile::CopyFrom(TDirectory *src, TMpiFile *file)
{
   TMpiFile *savdir = file;
   TDirectory *adir = savdir;
   adir->cd();
   // loop on all entries of this directory
   TKey *key;
   TIter nextkey(src->GetListOfKeys());
   while ((key = (TKey *)nextkey())) {
      const Char_t *classname = key->GetClassName();
      TClass *cl = gROOT->GetClass(classname);
      if (!cl) continue;
      if (cl->InheritsFrom(TDirectory::Class())) {
         src->cd(key->GetName());
         TDirectory *subdir = file;
         adir->cd();
         CopyFrom(subdir, file);
         adir->cd();
      } else if (cl->InheritsFrom(TTree::Class())) {
         TTree *T = (TTree *)src->Get(key->GetName());
         adir->cd();
         TTree *newT = T->CloneTree(-1, "fast");
         newT->Write();
      } else {
         src->cd();
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
/*! Method to copy the content from other file in the current file.
  \param source Another file.
*/
void TMpiFile::CopyFrom(TDirectory *source)
{
   CopyFrom(source, this);
}

//______________________________________________________________________________
/*! Method to merge of all  TMpiFiles in a  root process

  \param root root process (rank)  to merge the content of all files in all
  process
  \param save kTRUE if you want to save the merge procedure in the file now
  \param type type of merge is defined by the bit values in EPartialMergeType:
    - kRegular      : normal merge, overwritting the output file
    - kIncremental  : merge the input file with the content of the output file
  (if already exising) (default)
    - kAll          : merge all type of objects (default)
    - kResetable    : merge only the objects with a MergeAfterReset member
  function.
    - kNonResetable : merge only the objects without a MergeAfterReset member
  function.
*/
void TMpiFile::Merge(Int_t root, Bool_t save, Int_t type)
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
      if (save)
         fMerger->OutputFile(GetName(), "RECREATE");
      else
         fMerger->OutputMemFile(GetName(), "RECREATE");

      TDirectory::TContext ctxt;

      for (auto i = 0; i < fComm.GetSize(); i++) {
         Long64_t length = 0;
         TString filename;
         msgs[i].SetReadMode();
         msgs[i].Reset(kMESS_ANY);
         msgs[i].ReadTString(filename);
         msgs[i].ReadLong64(length);
         TMemFile *memffile = new TMemFile(filename, msgs[i].Buffer() + msgs[i].Length(), length, "UPDATE");
         msgs[i].SetBufferOffset(msgs[i].Length() + length);
         fMerger->AddAdoptFile(memffile);
         memffile = 0;
      }
      fMerger->PartialMerge(type);
      this->Delete("*;*");
      CopyFrom(fMerger->GetOutputFile());
      delete fMerger;
   }
}

//______________________________________________________________________________
/*!
   Method to save the file from memory to disk merging all in the given process
  (root)
  \param root root process (rank)  to merge the content of all files in all
  process
  \param type type of merge is defined by the bit values in EPartialMergeType:
    - kRegular      : normal merge, overwritting the output file
    - kIncremental  : merge the input file with the content of the output file
  (if already exising) (default)
    - kAll          : merge all type of objects (default)
    - kResetable    : merge only the objects with a MergeAfterReset member
  function.
    - kNonResetable : merge only the objects without a MergeAfterReset member
  function.

*/
void TMpiFile::Save(Int_t root, Int_t type)
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
      TDirectory::TContext ctxt;

      for (auto i = 0; i < fComm.GetSize(); i++) {
         Long64_t length = 0;
         TString filename;
         msgs[i].SetReadMode();
         msgs[i].Reset(kMESS_ANY);
         msgs[i].ReadTString(filename);
         msgs[i].ReadLong64(length);
         TMemFile *memffile = new TMemFile(filename, msgs[i].Buffer() + msgs[i].Length(), length, "UPDATE");
         msgs[i].SetBufferOffset(msgs[i].Length() + length);
         fMerger->AddAdoptFile(memffile);
         memffile = 0;
      }
      fMerger->PartialMerge(type);
      delete fMerger;
   }
}

//______________________________________________________________________________
/*!
  Save the file from memory to disk only from given rank,
  useful after call sync method to avoid repeated information and
  must be called only in one rank along all application.
  The idea is to save current file in a rank for example after call sync method
  \param rank process to save the file from memory to disk.
*/
void TMpiFile::SyncSave(Int_t rank)
{
   if (fComm.GetRank() == rank) {
      Write();
      TDirectory::TContext ctxt;
      auto tfile = TFile::Open(GetName(), fDiskOpenMode.Data(), GetTitle(), GetCompressionLevel());
      if (!tfile) {
         fComm.Abort(ERR_FILE);
      }
      CopyFrom(this, (ROOT::Mpi::TMpiFile *)tfile);
      tfile->Close();
      delete tfile;
   }
}

//______________________________________________________________________________
/*!
    Pethod to synchronize all TMpiFile content in all process of current
   TIntraCommunicator.
    All the data is synchronized merging all TMpiFiles in the given process
   (rank) and every TMpiFile is updated with
    a message using broadcast
    \param rank Process to merge the content of all files in all process
    \param type type of merge for synchronization is defined by the bit values
   in EPartialMergeType:
    - kRegular      : normal merge, overwritting the output file
    - kIncremental  : merge the input file with the content of the output file
   (if already exising) (default)
    - kAll          : merge all type of objects (default)
    - kResetable    : merge only the objects with a MergeAfterReset member
   function.
    - kNonResetable : merge only the objects without a MergeAfterReset member
   function.
*/
void TMpiFile::Sync(Int_t rank, Int_t type)
{
   Write();
   fMessage.Reset(kMESS_ANY);
   fMessage.WriteTString(GetName());
   fMessage.WriteLong64(GetEND());
   CopyTo(fMessage);

   TMpiMessage *msgs = NULL;
   if (fComm.GetRank() == rank) {
      msgs = new TMpiMessage[fComm.GetSize()];
   }

   fComm.Gather(&fMessage, 1, msgs, fComm.GetSize(), rank);
   if (fComm.GetRank() == rank) {
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
         TMemFile *memffile = new TMemFile(filename, msgs[i].Buffer() + msgs[i].Length(), length, "UPDATE");
         msgs[i].SetBufferOffset(msgs[i].Length() + length);
         fMerger->AddAdoptFile(memffile);
         memffile = 0;
      }
      fMerger->PartialMerge(type);
      TMemFile *mfile = dynamic_cast<TMemFile *>(fMerger->GetOutputFile());
      TDirectory::TContext ctxt;
      fMessage.SetWriteMode();
      fMessage.Reset(kMESS_ANY);
      fMessage.WriteLong64(mfile->GetEND());
      mfile->CopyTo(fMessage);
   }

   fComm.Bcast(fMessage, rank); // sending the new data for all processes

   Long64_t length = 0;
   fMessage.SetReadMode();
   fMessage.Reset(kMESS_ANY);
   fMessage.ReadLong64(length);

   this->Delete("*;*");

   TMemFile *mpifile = new TMemFile(GetName(), fMessage.Buffer() + fMessage.Length(), length, "UPDATE");
   fMessage.SetBufferOffset(fMessage.Length() + length);
   CopyFrom(mpifile);

   delete mpifile;
   if (fComm.GetRank() == rank) {
      delete fMerger;
   }
}
