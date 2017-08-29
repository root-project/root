// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#ifndef ROOT_Mpi_TMpiFile
#define ROOT_Mpi_TMpiFile

#include <TMemFile.h>
#include <TFileMerger.h>
#include <Mpi/Globals.h>
#include <Mpi/TErrorHandler.h>
#include <Mpi/TIntraCommunicator.h>
#include <TMutex.h>
namespace ROOT {

namespace Mpi {
class TCommunicator;
/**
* \class TMpiFileMerger
* Class that inherits from TFileMerger that helps to merge TMpiFiles in memory.
* \ingroup Mpi
*/

class TMpiFileMerger : public TFileMerger {
public:
   TMpiFileMerger(Bool_t isLocal = kTRUE, Bool_t histoOneGo = kTRUE) : TFileMerger(isLocal, histoOneGo) {}
   virtual Bool_t OutputMemFile(const Char_t *url, const Char_t *mode = "RECREATE", Int_t compressionLevel = 1);
   ClassDef(TMpiFileMerger, 1)
};
/**
* \class TMpiFile
* Class with all functionalities of TFile but that can to communicate the files using MPI with and object of
* TIntraCommunicator.
* \ingroup Mpi
*/
class TMpiFile : public TMemFile {
   TIntraCommunicator fComm; //!

   TMpiMessage fMessage;    //!
   TMpiFileMerger *fMerger; //!
   TString fDiskOpenMode;   //
protected:
   void CopyFrom(TDirectory *src, TMpiFile *file);
   void CopyFrom(TDirectory *src);

   TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Char_t *buffer, Long64_t size, Option_t *option = "",
            const Char_t *ftitle = "", Int_t compress = 1);
   TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "", const Char_t *ftitle = "",
            Int_t compress = 1);

public:
   TMpiFile(const TMpiFile &file);
   ~TMpiFile(){};
   static TMpiFile *Open(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "",
                         const Char_t *ftitle = "", Int_t compress = 1);
   void Merge(Int_t rank, Bool_t save = kFALSE, Int_t type = TFileMerger::kAllIncremental);

   void Sync(Int_t rank = 0, Int_t type = TFileMerger::kAllIncremental);

   void Save(Int_t rank = 0, Int_t type = TFileMerger::kAllIncremental);

   void SyncSave(Int_t rank = 0);

   ClassDef(TMpiFile, 1)
};
}
}
#endif
