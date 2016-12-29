
#ifndef ROOT_Mpi_TRun
#define ROOT_Mpi_TRun

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
      class TRun {
      private:
         Int_t iArgc;               //command line number of arguements
         Char_t **cArgv;            //command line arguments
         TString sMpirun;           //path to mpirun command
         TString sMpirunParams;     //mpirun params like -np 2 etc..
         TString sCompiler;         //compiler for mpi ex. mpic++
         TString sCompilerParams;   //Compile Flags/Linking flags for mpic++
         TRint   *sInterpreter;     //root interpreter to run mpi macros
         TString sHelpMsg;          //help message

      protected:
         Bool_t ProcessArgs(TString *error = 0);
         Bool_t Execute();
         Bool_t Compile();
         void InitHelp();
      public:
         TRun(int argc, char **argv, TString mpirun = "mpirun");
         Int_t Launch();
      };
   }//end namespace Mpi
}//end namespace ROOT
#endif