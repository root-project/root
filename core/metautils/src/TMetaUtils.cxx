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
// The ROOT::TMetaUtils namespace provides legacy wrappers around       //
// cling, the LLVM-based interpreter. It's an internal set of tools     //
// used by TCling and rootcling.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMetaUtils.h"

#include "cling/Interpreter/Interpreter.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include <string>

//______________________________________________________________________________
clang::QualType ROOT::TMetaUtils::LookupTypeDecl(cling::Interpreter& interp,
                                                 const char* tyname)
{
   // Look for name's clang::QualType.
   std::string funcname("TCling_LookupTypeDecl_");
   funcname += interp.createUniqueName();
   std::string code("void ");
   code += funcname + "(" + tyname + "*);";
   const clang::Decl* decl = 0;
   interp.processLine(code, true, &decl);
   const clang::FunctionDecl* funcDecl = 0;
   while (decl
          && (!(funcDecl = clang::dyn_cast<clang::FunctionDecl>(decl))
              || !funcDecl->getIdentifier()
              || funcDecl->getName() != funcname )) {
             decl = decl->getNextDeclInContext();
          }
   if (funcDecl && funcDecl->getNumParams() == 1) {
      clang::QualType paramQualType = funcDecl->getParamDecl(0)->getType();
      const clang::Type* paramType = paramQualType.getTypePtrOrNull();
      if (paramType && paramType->isPointerType()) {
         return paramType->getAs<clang::PointerType>()->getPointeeType();
      }
   }
   return clang::QualType();
}
