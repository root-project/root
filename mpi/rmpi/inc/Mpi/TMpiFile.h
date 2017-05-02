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
      class TMpiFile : public TMemFile {
         TIntraCommunicator fComm; //!
         TMpiMessage fMessage;     //!
         TFileMerger *fMerger;     //!
      public:
         TMpiFile(const TIntraCommunicator &comm, const Char_t *name, Option_t *option = "", const Char_t *ftitle = "", Int_t compress = 1);

         void Merge(Int_t root);

         ClassDef(TMpiFile, 1)

      };
   }
}
#endif
