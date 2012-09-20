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

namespace clang {
   class CompilerInstance;
   class Module;
}

namespace cling {
   class LookupHelper;
}

namespace ROOT {
   namespace TMetaUtils {

      // Add default template parameters.
      clang::QualType AddDefaultParameters(const clang::ASTContext& Ctx, clang::QualType instanceType);

      // Return the ROOT include directory
      std::string GetROOTIncludeDir(bool rootbuild);

      // Return the dictionary file name for a module
      std::string GetModuleFileName(const char* moduleName);

      // Declare a virtual module.map to clang. Returns Module on success.
      clang::Module* declareModuleMap(clang::CompilerInstance* CI,
                                      const char* moduleFileName,
                                      const char* headers[]);
                                
      // Return the -I needed to find RuntimeUniverse.h
      std::string GetInterpreterExtraIncludePath(bool rootbuild);

      // Return the LLVM / clang resource directory
      std::string GetLLVMResourceDir(bool rootbuild);
      
      // Return the ROOT include directory
      std::string GetROOTIncludeDir(bool rootbuild);

      // Return (in the argument 'output') a mangled version of the C++ symbol/type (pass as 'input')
      // that can be used in C++ as a variable name.
      void GetCppName(std::string &output, const char *input);

      // Return the type name normalized for ROOT,
      // keeping only the ROOT opaque typedef (Double32_t, etc.) and
      // adding default template argument for all types except the STL collections
      // where we remove the default template argument if any.
      void GetNormalizedName(std::string &norm_name, const clang::QualType &type, const clang::ASTContext &ctxt);

      // Initialize the list of typedef to keep (i.e. make them opaque for normalization).
      void InitOpaqueTypedef(const cling::LookupHelper &lookup);

   }; // class TMetaUtils

} // namespace ROOT

#endif // ROOT_TMetaUtils
