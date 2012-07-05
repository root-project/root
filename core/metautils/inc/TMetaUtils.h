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

#include "llvm/ADT/SmallSet.h"

// All clang entities must stay opaque types
namespace clang {
   class ASTContext;
   class QualType;
   class Type;
}
namespace cling {
   class Interpreter;
}

namespace ROOT {
   namespace TMetaUtils {
      clang::QualType LookupTypeDecl(cling::Interpreter& interp,
                                     const char* tyname);

      clang::QualType GetPartiallyDesugaredType(const clang::ASTContext& ctx, 
                                                clang::QualType qType, 
                      const llvm::SmallSet<const clang::Type*, 4>& typesToSkip);
    }
}
#endif // ROOT_TMetaUtils
