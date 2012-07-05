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
#include "cling/Interpreter/Value.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Type.h"
#include "clang/AST/Decl.h"
#include "clang/AST/TemplateBase.h"

#include <string>

using namespace clang;

//______________________________________________________________________________
QualType ROOT::TMetaUtils::LookupTypeDecl(cling::Interpreter& interp,
                                                 const char* tyname)
{
   // Look for name's clang::QualType.
   std::string funcname;
   interp.createUniqueName(funcname);
   funcname = "TCling_LookupTypeDecl_" + funcname ;
   std::string code("void ");
   code += funcname + "(" + tyname + "*);";
   const Decl* decl = 0;
   interp.declare(code, &decl);
   const FunctionDecl* funcDecl = 0;
   while (decl
          && (!(funcDecl = dyn_cast<FunctionDecl>(decl))
              || !funcDecl->getIdentifier()
              || funcDecl->getName() != funcname )) {
             decl = decl->getNextDeclInContext();
          }
   if (funcDecl && funcDecl->getNumParams() == 1) {
      QualType paramQualType = funcDecl->getParamDecl(0)->getType();
      const Type* paramType = paramQualType.getTypePtrOrNull();
      if (paramType && paramType->isPointerType()) {
         return paramType->getAs<PointerType>()->getPointeeType();
      }
   }
   return QualType();
}

//______________________________________________________________________________
QualType ROOT::TMetaUtils::GetPartiallyDesugaredType(const ASTContext& ctx, 
                                                     QualType qType, 
                              const llvm::SmallSet<const Type*, 4>& typesToSkip)
{
//   -*-*-*-*-*"Desugars" a type while skipping the ones in the set*-*-*-*-*
//              ===================================================
//
//  Desugars a given type recursively until strips all sugar or until gets a 
//  sugared type, which is to be skipped.
//     ctx          : The ASTContext
//     qType        : The type to be desugared
//     typesToSkip  : The set of sugared types not to be desugared
//
//   -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

   // If there are no constains - use the standard desugaring.
   if (!typesToSkip.size())
      return qType.getDesugaredType(ctx);

   while(isa<TypedefType>(qType.getTypePtr())) {
      if (!typesToSkip.count(qType.getTypePtr())) 
         qType = qType.getSingleStepDesugaredType(ctx);
      else
         return qType;
   }

   // In case of template specializations iterate over the arguments and 
   // desugar them as well.
   if(const TemplateSpecializationType* TST 
      = dyn_cast<const TemplateSpecializationType>(qType.getTypePtr())) {

      llvm::SmallVector<TemplateArgument, 4> desArgs;
      for(TemplateSpecializationType::iterator I = TST->begin(), E = TST->end();
          I != E; ++I) {
         QualType SubTy = I->getAsType();
      
         if (SubTy.isNull())
            continue;

         // Check if the type needs more desugaring and recurse.
         if (isa<TypedefType>(SubTy) || isa<TemplateSpecializationType>(SubTy))
            desArgs.push_back(TemplateArgument(GetPartiallyDesugaredType(ctx,
                                                                         SubTy,
                                                                  typesToSkip)));
      }

      // If desugaring happened allocate new type in the AST.
      if (desArgs.size()) {
         QualType Result 
            = ctx.getTemplateSpecializationType(TST->getTemplateName(), 
                                                desArgs.data(),
                                                desArgs.size(),
                                                TST->getCanonicalTypeInternal());
         return Result;
      }
   }
   return qType;   
}
