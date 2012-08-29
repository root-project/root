// @(#)root/metautils:$Id$
// Author: Paul Russo, 2009-10-06

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// The ROOT::TMetaUtils class provides utility wrappers around          //
// cling, the LLVM-based interpreter. It's an internal set of tools     //
// used by TCling and rootcling.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMetaUtils.h"
#include "RConfigure.h"
#include <iostream>
#include <stdlib.h>

std::string ROOT::TMetaUtils::GetInterpreterExtraIncludePath(bool rootbuild)
{
   // Return the -I needed to find RuntimeUniverse.h
   if (!rootbuild) {
#ifndef ROOTETCDIR
      const char* rootsys = getenv("ROOTSYS");
      if (!rootsys) {
         std::cerr << "ROOT::TMetaUtils::GetInterpreterExtraIncludePath(): ERROR: environment variable ROOTSYS not set!" << std::endl;
         return "-Ietc";
      }
      return std::string("-I") + rootsys + "/etc";
#else
      return std::string("-I") + ROOTETCDIR;
#endif
   }
   // else
   return "-Ietc";
}

std::string ROOT::TMetaUtils::GetLLVMResourceDir(bool rootbuild)
{
   // Return the LLVM / clang resource directory
#ifdef R__EXTERN_LLVMDIR
   return R__EXTERN_LLVMDIR;
#else
   return GetInterpreterExtraIncludePath(rootbuild)
      .substr(2, std::string::npos) + "/cling";
#endif
}

std::string ROOT::TMetaUtils::GetROOTIncludeDir(bool rootbuild)
{
   if (!rootbuild) {
#ifndef ROOTINCDIR
      if (getenv("ROOTSYS")) {
         std::string incl_rootsys = getenv("ROOTSYS");
         return incl_rootsys + "/include";
      } else {
         std::cerr << "ROOT::TMetaUtils::GetROOTIncludeDir(): "
                   << "ERROR: environment variable ROOTSYS not set" << std::endl;
         return "include";
      }
#else
      return ROOTINCDIR;
#endif
   }
   // else
   return "include";
}

