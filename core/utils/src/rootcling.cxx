// Authors: Axel Naumann, Philippe Canal, Danilo Piparo

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_STAGE1_BUILD
#include "rootclingTCling.h"
#endif

#include "rootcling_impl.h"
#include "RConfigure.h"
#include "RConfig.h"

#ifdef _MSC_VER
  #define R__DLLEXPORT __declspec(dllexport)
#else
  #define R__DLLEXPORT
#endif

extern "C" {
   R__DLLEXPORT void usedToIdentifyRootClingByDlSym() {}
}

#ifdef _WIN32
#ifdef system
#undef system
#endif
#include <windows.h>
#include <Tlhelp32.h> // for MAX_MODULE_NAME32
#include <process.h>
#define PATH_MAX _MAX_PATH
#ifdef interface
// prevent error coming from clang/AST/Attrs.inc
#undef interface
#endif
#endif

#ifdef __APPLE__
#include <libgen.h> // Needed for basename
#include <mach-o/dyld.h>
#endif

#if !defined(R__WIN32)
#include <limits.h>
#include <unistd.h>
#endif


////////////////////////////////////////////////////////////////////////////////
/// Returns the executable path name, used e.g. by SetRootSys().

const char *GetExePath()
{
   static std::string exepath;
   if (exepath == "") {
#ifdef __APPLE__
      exepath = _dyld_get_image_name(0);
#endif
#if defined(__linux) || defined(__linux__)
      char linkname[PATH_MAX];  // /proc/<pid>/exe
      char buf[PATH_MAX];     // exe path name
      pid_t pid;

      // get our pid and build the name of the link in /proc
      pid = getpid();
      snprintf(linkname, PATH_MAX, "/proc/%i/exe", pid);
      int ret = readlink(linkname, buf, 1024);
      if (ret > 0 && ret < 1024) {
         buf[ret] = 0;
         exepath = buf;
      }
#endif
#ifdef _WIN32
      char *buf = new char[MAX_MODULE_NAME32 + 1];
      ::GetModuleFileName(NULL, buf, MAX_MODULE_NAME32 + 1);
      char *p = buf;
      while ((p = strchr(p, '\\')))
         * (p++) = '/';
      exepath = buf;
      delete[] buf;
#endif
   }
   return exepath.c_str();
}

////////////////////////////////////////////////////////////////////////////////

#ifdef __ICC
#pragma warning disable 69
#endif


int main(int argc, char **argv)
{
   // Force the emission of the symbol - the compiler cannot know that argv
   // is always set.
   if (!argv) {
      auto dummyVal =  (int)(long)&usedToIdentifyRootClingByDlSym;
      return dummyVal;
   }

   ROOT::Internal::RootCling::DriverConfig config{};
   config.fExePath = GetExePath();
#ifdef ROOT_STAGE1_BUILD
   config.fBuildingROOTStage1 = true;
#else
   config.fBuildingROOTStage1 = false;
   config.fTROOT__GetExtraInterpreterArgs = &TROOT__GetExtraInterpreterArgs;
   config.fTCling__GetInterpreter = &TCling__GetInterpreter;
   config.fInitializeStreamerInfoROOTFile = &InitializeStreamerInfoROOTFile;
   config.fAddStreamerInfoToROOTFile = &AddStreamerInfoToROOTFile;
   config.fAddTypedefToROOTFile = &AddTypedefToROOTFile;
   config.fAddEnumToROOTFile = &AddEnumToROOTFile;
   config.fAddAncestorPCMROOTFile = &AddAncestorPCMROOTFile;
   config.fCloseStreamerInfoROOTFile = &CloseStreamerInfoROOTFile;
#endif
   return rootcling_driver(argc, argv, config);
}
