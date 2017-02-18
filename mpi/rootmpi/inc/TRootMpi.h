
#ifndef ROOT_Mpi_TRootMpi
#define ROOT_Mpi_TRootMpi

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#include<TSystem.h>
#ifndef _WIN32
#include<TUnixSystem.h>
#endif
#include<TRint.h>
#include <TString.h>
namespace ROOT {
   namespace Mpi {
      class TRootMpi {
      private:
         Int_t fArgc;                 //command line number of arguements
         Char_t **fArgv;              //command line arguments
         TString fMpirun;             //path to mpirun command
         TString fMpirunParams;       //mpirun params like -np 2 etc..
         TString fCompiler;           //compiler for mpi ex. mpic++
         TString fCompilerParams;     //Compile Flags/Linking flags for mpic++
         TString fHelpMsg;            //help message

      protected:
         Int_t ProcessArgs();
         Int_t Execute();
         Int_t Compile();
         void InitHelp();
      public:
         TRootMpi(Int_t argc, Char_t **argv);
         Int_t Launch();
      };
   }//end namespace Mpi
}//end namespace ROOT
#endif
