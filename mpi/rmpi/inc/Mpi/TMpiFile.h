// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TMpiFile
#define ROOT_Mpi_TMpiFile

#include<TMemFile.h>
#include<TFileMerger.h>
#include<Mpi/Globals.h>
#include<Mpi/TErrorHandler.h>
#include<Mpi/TIntraCommunicator.h>
#include <TMutex.h>
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
         TMpiFileMerger *fMerger;  //!
         Bool_t fSync;             //
         Int_t fSyncType;          //
         TString fDiskOpenMode;    //
      protected:
         void CopyFrom(TDirectory *source, TMpiFile *file) ;
         void SyncSave(Int_t type);
         TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);
         TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);
      public:
         TMpiFile(const TMpiFile &file);
         ~TMpiFile() {};
         static TMpiFile *Open(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);
         //merge of all  TMpiFile in a  root process
         void Merge(Int_t root, Int_t type = TFileMerger::kAllIncremental);

         //method to synchronize all TMpiFile content in all process of a given TIntraCommunicator
         void Merge(Int_t type = TFileMerger::kAllIncremental);

         //save the file from memory to disk in the process that is called, but needs to be called with Sync
         //and the Sync call must be visible for all processes
         void Save(Int_t type = TFileMerger::kAllIncremental);


         //Causes all previous writes to be transferred to the storage device
         //This method are using the ring algorithm with blocking communication to do it
         // in a sequential mode along of all processes in the intracommunicator
         void Sync();

         ClassDef(TMpiFile, 1)

      };
   }
}
#endif
