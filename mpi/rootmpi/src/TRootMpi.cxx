#include<TRootMpi.h>
#include<Math/Error.h>
#include<stdlib.h>
#include<iostream>
#include<TStringLong.h>
#include <TApplication.h>
#include<TROOT.h>
#include<mpi.h>
using namespace ROOT::Mpi;
//______________________________________________________________________________
TRootMpi::TRootMpi(Int_t argc, Char_t **argv)
{

   fRootSys = TROOT::GetRootSys();

   //TODO: added codes to check all defines and paths for executables
   fMpirun = ROOT_MPI_EXEC;
   fMpirunParams = " ";

   fCompiler = ROOT_MPI_CXX;

   fValgrind = ROOT_MPI_VALGRIND;
   fCallValgrind = kFALSE;
   fValgrindParams =  Form("--suppressions=%s/etc/valgrind-root.supp", fRootSys);
   fValgrindParams += " -v --leak-check=full --show-leak-kinds=all --track-origins=yes --show-reachable=yes  ";

   fArgc = argc;
   fArgv = argv;

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
      Int_t status;
      auto macroFile = fArgv[fArgc - 2];
      gApplication->ProcessFile(macroFile, &status);
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
#if OPEN_MPI
   fMpirunParams = " -q ";
#else
   fMpirunParams = " ";
#endif

   if ((TString(fArgv[1]) == TString("-h")) || (TString(fArgv[1]) == TString("--help"))) {
      std::cout << fHelpMsg;
      return 0;
   }
   if ((TString(fArgv[1]) == TString("--help-mpic++"))) {
      return gSystem->Exec("mpic++ --help");
   }
   if ((TString(fArgv[1]) == TString("--help-mpirun"))) {
      TString cmd_help = fMpirun + " --help";
      return gSystem->Exec(cmd_help.Data());
   }


   if (TString(fArgv[1]) == "-C") {
      for (int i = 2; i < fArgc; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");
         fCompilerParams += " " + arg;
      }
      fCompilerParams += ROOT_MPI_CFLAGS;
      fCompilerParams += " ";
      fCompilerParams += gSystem->GetIncludePath();
      fCompilerParams += " ";
      fCompilerParams += gSystem->GetLinkedLibs();
      fCompilerParams += " -std=c++11 ";
      fCompilerParams += ROOT_MPI_LDFLAGS;
      fCompilerParams += " -lRMpi -lNet -lRIO -lCore -lThread -lHist ";
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
            if (arg == "-valgrind") fCallValgrind = kTRUE;
            else fMpirunParams += " " + arg;
         }
      }

      if (fCallValgrind) {
         fMpirunParams += " " + fValgrind + " " + fValgrindParams + Form(" --log-file=report-%s-%s.memcheck ", fArgv[fArgc - 1], "%p");
      }
      fMpirunParams += " ";
//       fMpirunParams += Form("%s/bin/%s",fRootSys,fArgv[0]);
      fMpirunParams += Form("%s/bin/root -l -q", fRootSys);
      fMpirunParams += " \"";
      fMpirunParams += fArgv[fArgc - 1];//macro file is the last
      fMpirunParams += " \"";
      fMpirunParams += sRootParams;
      return Execute();
   }
   return 0;
}

//______________________________________________________________________________
Int_t TRootMpi::Compile()
{
   auto cmd = fCompiler + " " + fCompilerParams;
   std::cout << cmd << std::endl;
   return gSystem->Exec(cmd.Data());
}

//______________________________________________________________________________
Int_t TRootMpi::Execute()
{
   auto cmd = fMpirun + " " + fMpirunParams;
   auto status = gSystem->Exec(cmd.Data());
   if (fCallValgrind) std::cout << "\nValgrind files " << Form(" report-%s-pid.memcheck ", fArgv[fArgc - 1]) << "must be generated." << std::endl;
   return status;
}

void TRootMpi::InitHelp()
{
   fHelpMsg = "Usage for Macro: rootmpi [mpirun options] [root/cling options] [macro file.C ]\n";
   fHelpMsg += "Usage for Binary Executable: rootmpi -R [mpirun options] [Binary Executable]\n";
   fHelpMsg += "Usage to Compile Code: rootmpi -C [mpic++ options] [Souce1.cxx Source2.cpp ..] \n";
   fHelpMsg += "Options:\n";
   fHelpMsg += "  --help-mpic++  show mpi options for compilation\n";
   fHelpMsg += "  --help-mpirun  show mpi options for execution\n";
   fHelpMsg += "Options Valgrind:\n";
   fHelpMsg += " -valgrind : launch mpi execution with valgrind\n";
   fHelpMsg += "Options Cling/ROOT:\n";
   fHelpMsg += " -b : run in batch mode without graphics\n";
   fHelpMsg += " -n : do not execute logon and logoff macros as specified in .rootrc\n";
   fHelpMsg += " -q : exit after processing command line macro files\n";
   fHelpMsg += " -l : do not show splash screen\n";
   fHelpMsg += " -x : exit on exception\n";
   fHelpMsg += " dir : if dir is a valid directory cd to it before executing\n";
   fHelpMsg += "-memstat : run with memory usage monitoring\n";
}

