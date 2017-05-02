#include<Mpi/TMpiFile.h>
#include<TKey.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TMpiFile::TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option, const Char_t *ftitle, Int_t compress): TMemFile(name, option, ftitle, compress), fComm(comm)
{
   TMessage::EnableSchemaEvolutionForAll(kTRUE);
}

//______________________________________________________________________________
void TMpiFile::Merge(Int_t root)
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
      fMerger = new TFileMerger(kFALSE, kFALSE);
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
      fMerger->PartialMerge(TFileMerger::kAllIncremental);
      delete fMerger;
   }
}

