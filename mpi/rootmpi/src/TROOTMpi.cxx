#include<TROOTMpi.h>
#include<Math/Error.h>
#include<stdlib.h>
#include<iostream>
#include<TStringLong.h>

using namespace ROOT::Mpi;
//______________________________________________________________________________
TROOTMpi::TROOTMpi(int argc, char **argv, TString mpirun)
{
   sMpirun = mpirun;
   sMpirunParams = " ";
   iArgc = argc;
   cArgv = argv;
   sCompiler = "mpic++ ";
   InitHelp();
}

//______________________________________________________________________________
Int_t TROOTMpi::Launch()
{

   if (iArgc == 1) {
      std::cerr << sHelpMsg;
      return false;
   }

   if (TString(cArgv[iArgc - 1]) == "-process_macro") {
      Int_t tmp_Argc = iArgc - 1;
      TRint *rootmpi = new TRint("rootmpi", &tmp_Argc, cArgv);
      rootmpi->ExitOnException();
      rootmpi->Run();
      Int_t status = rootmpi->ReturnFromRun();
//     rootmpi->Terminate(status);
      delete rootmpi;
      return status;
   } else {
      TString error;
      if (ProcessArgs(&error) != 0) {
         std::cerr << error << "\n";
         return false;
      }
   }
   return false;
}

//______________________________________________________________________________
Bool_t TROOTMpi::ProcessArgs(TString *error)
{
   sCompilerParams.Clear();
   sMpirunParams.Clear();
   if ((TString(cArgv[1]) == TString("-h")) || (TString(cArgv[1]) == TString("--help"))) {
      std::cout << sHelpMsg;
      return true;
   }
   if ((TString(cArgv[1]) == TString("--help-mpic++"))) {
      return gSystem->Exec("mpic++ --help");
   }
   if ((TString(cArgv[1]) == TString("--help-mpirun"))) {
      return gSystem->Exec("mpirun --help");
   }

   if (TString(cArgv[1]) == "-C") {
      for (int i = 2; i < iArgc; i++) {
         TString arg = cArgv[i];
         arg.ReplaceAll(" ", "");
         sCompilerParams += " " + arg;
      }
      sCompilerParams += " ";
      sCompilerParams += gSystem->GetIncludePath();
      sCompilerParams += " ";
      sCompilerParams += gSystem->GetLinkedLibs();
      sCompilerParams += " -std=c++11 ";
      sCompilerParams += " -lRMpi -lNet -lRIO ";
      return Compile();
   } else if (TString(cArgv[1]) == "-R") {
      for (int i = 2; i < iArgc; i++) {
         TString arg = cArgv[i];
         arg.ReplaceAll(" ", "");
         sMpirunParams += " " + arg;
      }
      return Execute();
   } else {
      TString sRootParams = " -l -q ";//added -l -q by default
      for (int i = 1; i < iArgc - 1; i++) {
         TString arg = cArgv[i];
         arg.ReplaceAll(" ", "");
         if ((arg == "-b") || (arg == "-n") || (arg == "-l") || (arg == "-q") || (arg == "-x") || (arg == "-memstat")) {
            sRootParams += " " + arg;
         } else {
            sMpirunParams += " " + arg;
         }
      }
      sMpirunParams += " ";
      sMpirunParams += cArgv[0];
      sMpirunParams += " \"";
      sMpirunParams += cArgv[iArgc - 1];
      sMpirunParams += " \"";
      sMpirunParams += sRootParams;
      sMpirunParams += " -process_macro ";
      return Execute();
   }
   return true;
}

//______________________________________________________________________________
Bool_t TROOTMpi::Compile()
{
   return gSystem->Exec(TStringLong(sCompiler + " " + sCompilerParams).Data());
}

//______________________________________________________________________________
Bool_t TROOTMpi::Execute()
{
   return gSystem->Exec(TStringLong(sMpirun + " " + sMpirunParams).Data());
}

void TROOTMpi::InitHelp()
{
   sHelpMsg = "Usage for Macro: rootmpi [mpirun options] [root/cling options] [macro file.C ]\n";
   sHelpMsg += "Usage for Binary Executable: rootmpi -R [mpirun options] [Binary Executable]\n";
   sHelpMsg += "Usage to Compile Code: rootmpi -C [mpic++ options] [Souce1.cxx Source2.cpp ..] \n";
   sHelpMsg += "Options:\n";
   sHelpMsg += "  --help-mpic++  show mpi options for compilation\n";
   sHelpMsg += "  --help-mpirun  show mpi options for execution\n";
   sHelpMsg += "Options Cint/ROOT:\n";
   sHelpMsg += " -b : run in batch mode without graphics\n";
   sHelpMsg += " -n : do not execute logon and logoff macros as specified in .rootrc\n";
   sHelpMsg += " -q : exit after processing command line macro files\n";
   sHelpMsg += " -l : do not show splash screen\n";
   sHelpMsg += " -x : exit on exception\n";
   sHelpMsg += " dir : if dir is a valid directory cd to it before executing\n";
   sHelpMsg += "-memstat : run with memory usage monitoring\n";
}

