#include<TRootMpi.h>
#include<Math/Error.h>
#include<stdlib.h>
#include<iostream>
#include<TStringLong.h>
#include <TApplication.h>
#include<TROOT.h>

using namespace ROOT::Mpi;
//______________________________________________________________________________
TRootMpi::TRootMpi(Int_t argc, Char_t **argv)
{
   fMpirun = ROOT_MPI_EXEC;
   fMpirunParams = " ";
   fArgc = argc;
   fArgv = argv;
   fCompiler = "mpic++";
   InitHelp();
}

//______________________________________________________________________________
Int_t TRootMpi::Launch()
{

   if (fArgc == 1) {
      std::cerr << fHelpMsg;
      return 0;
   }

   if (TString(fArgv[fArgc - 1]) == "-process_macro") {
      Int_t tmp_Argc = fArgc - 1; //less macro name
      TRint *rootmpi = new TRint("rootmpi", &tmp_Argc, fArgv, 0, 0, kTRUE);
      Int_t status;
      auto macroFile = rootmpi->Argv(fArgc - 2);
      rootmpi->ProcessFile(macroFile, &status);
      delete rootmpi;
      return status;
   } else {
      return ProcessArgs() ? 1 : 0;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TRootMpi::ProcessArgs()
{
   fCompilerParams.Clear();
   fMpirunParams.Clear();
   if ((TString(fArgv[1]) == TString("-h")) || (TString(fArgv[1]) == TString("--help"))) {
      std::cout << fHelpMsg;
      return 0;
   }
   if ((TString(fArgv[1]) == TString("--help-mpic++"))) {
      return gSystem->Exec("mpic++ --help");
   }
   if ((TString(fArgv[1]) == TString("--help-mpirun"))) {
      return gSystem->Exec("mpirun --help");
   }

   if (TString(fArgv[1]) == "-C") {
      for (int i = 2; i < fArgc; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");
         fCompilerParams += " " + arg;
      }
      fCompilerParams += " ";
      fCompilerParams += gSystem->GetIncludePath();
      fCompilerParams += " ";
      fCompilerParams += gSystem->GetLinkedLibs();
      fCompilerParams += " -std=c++11 ";
      fCompilerParams += " -lRMpi -lNet -lRIO ";
      return Compile();
   } else if (TString(fArgv[1]) == "-R") {
      for (int i = 2; i < fArgc; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");
         fMpirunParams += " " + arg;
      }
      return Execute();
   } else {
      TString sRootParams = " ";//added -l -q by default

      for (int i = 1; i < fArgc - 1; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");
         if ((arg == "-b") || (arg == "-n") || (arg == "-l") || (arg == "-q") || (arg == "-x") || (arg == "-memstat")) {
            sRootParams += " " + arg;
         } else {
            fMpirunParams += " " + arg;
         }
      }
      fMpirunParams += " ";
      fMpirunParams += fArgv[0];
      fMpirunParams += " \"";
      fMpirunParams += fArgv[fArgc - 1];//macro file is the last
      fMpirunParams += " \"";
      fMpirunParams += sRootParams;
      fMpirunParams += " -process_macro ";
      return Execute();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TRootMpi::Compile()
{
   auto cmd = fCompiler + " " + fCompilerParams;
   return gSystem->Exec(cmd.Data());
}

//______________________________________________________________________________
Int_t TRootMpi::Execute()
{
   auto cmd = fMpirun + " " + fMpirunParams;
   return gSystem->Exec(cmd.Data());
}

void TRootMpi::InitHelp()
{
   fHelpMsg = "Usage for Macro: rootmpi [mpirun options] [root/cling options] [macro file.C ]\n";
   fHelpMsg += "Usage for Binary Executable: rootmpi -R [mpirun options] [Binary Executable]\n";
   fHelpMsg += "Usage to Compile Code: rootmpi -C [mpic++ options] [Souce1.cxx Source2.cpp ..] \n";
   fHelpMsg += "Options:\n";
   fHelpMsg += "  --help-mpic++  show mpi options for compilation\n";
   fHelpMsg += "  --help-mpirun  show mpi options for execution\n";
   fHelpMsg += "Options Cint/ROOT:\n";
   fHelpMsg += " -b : run in batch mode without graphics\n";
   fHelpMsg += " -n : do not execute logon and logoff macros as specified in .rootrc\n";
   fHelpMsg += " -q : exit after processing command line macro files\n";
   fHelpMsg += " -l : do not show splash screen\n";
   fHelpMsg += " -x : exit on exception\n";
   fHelpMsg += " dir : if dir is a valid directory cd to it before executing\n";
   fHelpMsg += "-memstat : run with memory usage monitoring\n";
}

