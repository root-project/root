#include <Math/Error.h>
#include <TApplication.h>
#include <TROOT.h>
#include <TRootMpi.h>
#include <TStringLong.h>
#include <iostream>
#include <mpi.h>
#include <stdlib.h>
using namespace ROOT::Mpi;
//______________________________________________________________________________
TRootMpi::TRootMpi(Int_t argc, Char_t **argv)
{
// export TMPDIR for OpenMPI at mac is required
// https://www.open-mpi.org/faq/?category=osx#startup-errors-with-open-mpi-2.0.x
#if defined(R__MACOSX) && defined(OPEN_MPI)
   gSystem->Setenv("TMPDIR", "/tmp");
#endif

   fRootSys = TROOT::GetRootSys();

   // TODO: added codes to check all defines and paths for executables
   fMpirun = ROOT_MPI_EXEC;
   fMpirunParams = " ";

   fCompiler = ROOT_MPI_CXX;
#if ROOT_MPI_VALGRINDFOUND
   fValgrind = ROOT_MPI_VALGRIND;
#else
   fValgrind = "";
#endif
   fCallValgrind = kFALSE;
   fValgrindParams = Form("--suppressions=%s/etc/valgrind-root.supp", fRootSys);

   fArgc = argc;
   fArgv = argv;

#if PYTHONINTERP_FOUND
   fPython = PYTHON_EXECUTABLE;
#endif

   auto env = gSystem->Getenv("ROOT_MPI_VALGRIND_OPTIONS");
   if (env)
      fValgrindParams += Form(" %s ", env);
   else
      fValgrindParams += " -v --leak-check=full --show-leak-kinds=all --track-origins=yes --show-reachable=yes  ";

   env = gSystem->Getenv("ROOT_MPI_VALGRIND_TOOL");

   if (env) {
      // memcheck, cachegrind, callgrind, helgrind, drd, massif, lackey, none, exp-sgcheck, exp-bbv, exp-dhat
      fValgrindTool = env;
   } else // using by default memcheck
   {
      fValgrindTool = "memcheck";
   }

   env = gSystem->Getenv("ROOT_MPI_PYTHON");
   if (env)
      fPython = env;

   env = gSystem->Getenv("ROOT_MPI_VERBOSE");
   if (env)
      fVerbose = atoi(env);
   else
      fVerbose = false;

   fCompile = kFALSE;
   InitHelp();
}

//______________________________________________________________________________
Int_t TRootMpi::Launch()
{

   if (fArgc == 1) {
      std::cerr << fHelpMsg;
      return 1;
   }

   Int_t status;
   status = ProcessArgs() ? 1 : 0;
   if (status != 1) {
      if (fCompile) {
         status = Compile();
         if (status != 0) {
            auto cmd = fCompiler + " " + fCompilerParams;
            Error(Form("%s(...) %s[%d]", __FUNCTION__, __FILE__, __LINE__), "\nCompiling %s\nError code %d", cmd.Data(),
                  status);
         }
      } else {
         status = Execute();
         if (status != 0) {
            auto cmd = fMpirun + " " + fMpirunParams;
            Error(Form("%s(...) %s[%d]", __FUNCTION__, __FILE__, __LINE__), "\nExecuting %s\nError code %d", cmd.Data(),
                  status);
         }
      }
   }
   return status;
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
      fCompile = kTRUE;
      return 0;
   } else if (TString(fArgv[1]) == "-R") {
      for (int i = 2; i < fArgc - 1; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");

#if ROOT_MPI_VALGRINDFOUND
         if (arg == "-valgrind")
            fCallValgrind = kTRUE;
         else
            fMpirunParams += " " + arg;
#else
         fMpirunParams += " " + arg;
#endif
      }
      if (fCallValgrind) {
         fMpirunParams += Form(" %s --tool=%s ", fValgrind.Data(), fValgrindTool.Data());
         fMpirunParams += Form(" --log-file=report-%s-%s.%s ", TString(fArgv[fArgc - 1]).ReplaceAll("./", "").Data(),
                               "%p", fValgrindTool.Data());
      }
      fMpirunParams += fArgv[fArgc - 1];
      fCompile = kFALSE;
      return 0;
   } else {
      TString sRootParams = " "; // added -l -q by default

      for (int i = 1; i < fArgc - 1; i++) {
         TString arg = fArgv[i];
         arg.ReplaceAll(" ", "");
         if ((arg == "-b") || (arg == "-n") || (arg == "-l") || (arg == "-q") || (arg == "-x") || (arg == "-memstat")) {
            sRootParams += " " + arg;
         } else {
#if ROOT_MPI_VALGRINDFOUND
            if (arg == "-valgrind")
               fCallValgrind = kTRUE;
            else
               fMpirunParams += " " + arg;
#else
            fMpirunParams += " " + arg;
#endif
         }
      }

      if (fCallValgrind) {
         fMpirunParams += Form(" %s --tool=%s ", fValgrind.Data(), fValgrindTool.Data());
         fMpirunParams += Form(" --log-file=report-%s-%s.%s ", TString(fArgv[fArgc - 1]).ReplaceAll("./", "").Data(),
                               "%p", fValgrindTool.Data());
      }

      fMpirunParams += " ";
      if (TString(fArgv[fArgc - 1]).EndsWith(".py")) {
         fMpirunParams += fPython + " ";
         fMpirunParams += fArgv[fArgc - 1]; // macro file is the last
      } else {
         fMpirunParams += Form("%s/bin/root -l -x -q", fRootSys);
         fMpirunParams += " \"";
         fMpirunParams += fArgv[fArgc - 1]; // macro file is the last
         fMpirunParams += " \"";
         fMpirunParams += sRootParams;
      }

      fCompile = kFALSE;
   }
   return 0;
}

//______________________________________________________________________________
Int_t TRootMpi::Compile()
{
   auto cmd = fCompiler + " " + fCompilerParams;
   if (fVerbose)
      printf("\nCOMPILING %s\n", cmd.Data());
   return gSystem->Exec(cmd.Data());
}

//______________________________________________________________________________
Int_t TRootMpi::Execute()
{
   auto cmd = fMpirun + " " + fMpirunParams;
   if (fVerbose)
      printf("\nEXECUTING %s\n", cmd.Data());
   auto status = gSystem->Exec(cmd.Data());

   if (fCallValgrind)
      std::cout << "\nValgrind files "
                << Form("report-%s-%s.%s ", TString(fArgv[fArgc - 1]).ReplaceAll("./", "").Data(), "%p",
                        fValgrindTool.Data())
                << "must be generated." << std::endl;
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
   fHelpMsg += "Environment variables:\n";
   fHelpMsg += "ROOT_MPI_VERBOSE: boolean to enable/disable verbose execution.\n";
   fHelpMsg += "ROOT_MPI_VALGRIND_TOOL: string with valgrind tool, memcheck, cachegrind, callgrind, helgrind, drd, "
               "massif etc..\n";
   fHelpMsg += "ROOT_MPI_VALGRIND_OPTIONS: string with valgrind options for a given tool\n";
}
