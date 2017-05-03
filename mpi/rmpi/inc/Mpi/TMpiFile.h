// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TMpiFile
#define ROOT_Mpi_TMpiFile

#include<TMemFile.h>
#include<TFileMerger.h>
#include<Mpi/Globals.h>
#include<Mpi/TErrorHandler.h>
#include<Mpi/TIntraCommunicator.h>
namespace ROOT {

   namespace Mpi {
      class TCommunicator;
      class TMpiFileMerger : public TFileMerger {
      public:
         TMpiFileMerger(Bool_t isLocal = kTRUE, Bool_t histoOneGo = kTRUE): TFileMerger(isLocal, histoOneGo) {}
         virtual Bool_t OutputMemFile(const Char_t *url, const Char_t *mode = "RECREATE", Int_t compressionLevel = 1);
         ClassDef(TMpiFileMerger, 1)
      };
      class TMpiFile : public TMemFile {
         TIntraCommunicator fComm; //!

         TMpiMessage fMessage;     //!
         TMpiFileMerger *fMerger;     //!
      protected:
         TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);
         void CopyFrom(TDirectory *source, TMpiFile *file) ;
      public:
         TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);

         //merge of all  TMpiFile in a process root
         void Merge(Int_t root, Int_t type = TFileMerger::kAllIncremental);

         //save all files from memory to disk from all processes
         void Save(Int_t type = TFileMerger::kAllIncremental);

         //method to synchronize all TMpiFile content in all process of a given TIntraCommunicator
         void Sync(Int_t type = TFileMerger::kAllIncremental);


         ClassDef(TMpiFile, 1)

      };
   }
}
#endif
