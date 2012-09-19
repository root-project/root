// @(#)root/metautils:$Id$
// Author: Axel Naumann, Nov 2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMetaUtils
#define ROOT_TMetaUtils

namespace clang {
   class ASTContext;
   class QualType;
}

#include <string>

namespace ROOT {
   namespace TMetaUtils {

      // Add default template parameters.
      clang::QualType AddDefaultParameters(const clang::ASTContext& Ctx, clang::QualType instanceType);

      // Return the -I needed to find RuntimeUniverse.h
      std::string GetInterpreterExtraIncludePath(bool rootbuild);

      // Return the LLVM / clang resource directory
      std::string GetLLVMResourceDir(bool rootbuild);
      
      // Return the ROOT include directory
      std::string GetROOTIncludeDir(bool rootbuild);
   }; // class TMetaUtils

} // namespace ROOT

#endif // ROOT_TMetaUtils
