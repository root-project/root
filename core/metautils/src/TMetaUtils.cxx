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
// ROOT::TMetaUtils provides utility wrappers around                    //
// cling, the LLVM-based interpreter. It's an internal set of tools     //
// used by TCling and rootcling.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMetaUtils.h"

#include "RConfigure.h"
#include <iostream>
#include <stdlib.h>

#include "TClassEdit.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/NestedNameSpecifier.h"

#include "cling/Utils/AST.h"

//////////////////////////////////////////////////////////////////////////
static
clang::NestedNameSpecifier* AddDefaultParametersNNS(const clang::ASTContext& Ctx, 
                                                    clang::NestedNameSpecifier* scope) {
    // Add default parameter to the scope if needed.

   const clang::Type* scope_type = scope->getAsType();
   if (scope_type) {
      // this is not a namespace, so we might need to desugar
     clang::NestedNameSpecifier* outer_scope = scope->getPrefix();
      if (outer_scope) {
         outer_scope = AddDefaultParametersNNS(Ctx, outer_scope);
      }

      clang::QualType addDefault = 
         ROOT::TMetaUtils::AddDefaultParameters(Ctx,
                                                clang::QualType(scope_type,0) );
      // NOTE: Should check whether the type has changed or not.
      return clang::NestedNameSpecifier::Create(Ctx,outer_scope,
                                                false /* template keyword wanted */,
                                                addDefault.getTypePtr());
   }
   return scope;
}


//////////////////////////////////////////////////////////////////////////
clang::QualType ROOT::TMetaUtils::AddDefaultParameters(const clang::ASTContext& Ctx, clang::QualType instanceType)
{
   // Add any unspecified template parameters to the class template instance,
   // mentioned anywhere in the type.
   //
   // Note: this does not strip any typedef but could be merged with cling::utils::Transform::GetPartiallyDesugaredType
   // if we can safely replace TClassEdit::IsStd with a test on the declaring scope
   // and if we can resolve the fact that the added parameter do not take into account possible use/dependences on Double32_t
   // and if we decide that adding the default is the right long term solution or not.
   // Whether it is or not depend on the I/O on whether the default template argument might change or not
   // and whether they (should) affect the on disk layout (for STL containers, we do know they do not).

   // In case of Int_t* we need to strip the pointer first, desugar and attach
   // the pointer once again.
   if (instanceType->isPointerType()) {
      // Get the qualifiers.
      clang::Qualifiers quals = instanceType.getQualifiers();      
      instanceType = AddDefaultParameters(Ctx, instanceType->getPointeeType());
      instanceType = Ctx.getPointerType(instanceType);
      // Add back the qualifiers.
      instanceType = Ctx.getQualifiedType(instanceType, quals);
   }

   // In case of Int_t& we need to strip the pointer first, desugar and attach
   // the pointer once again.
   if (instanceType->isReferenceType()) {
      // Get the qualifiers.
      bool isLValueRefTy = llvm::isa<clang::LValueReferenceType>(instanceType.getTypePtr());
      clang::Qualifiers quals = instanceType.getQualifiers();
      instanceType = AddDefaultParameters(Ctx, instanceType->getPointeeType());

      // Add the r- or l- value reference type back to the desugared one
      if (isLValueRefTy)
        instanceType = Ctx.getLValueReferenceType(instanceType);
      else
        instanceType = Ctx.getRValueReferenceType(instanceType);
      // Add back the qualifiers.
      instanceType = Ctx.getQualifiedType(instanceType, quals);
   }
   
   // Treat the Scope.
   clang::NestedNameSpecifier* prefix = 0;
   const clang::ElaboratedType* etype 
      = llvm::dyn_cast<clang::ElaboratedType>(instanceType.getTypePtr());
   if (etype) {
      // We have to also handle the prefix.
 
      prefix = AddDefaultParametersNNS(Ctx, etype->getQualifier());
      instanceType = clang::QualType(etype->getNamedType().getTypePtr(),instanceType.getLocalFastQualifiers());
   }

   // In case of template specializations iterate over the arguments and 
   // add unspecified default parameter.

   const clang::TemplateSpecializationType* TST 
      = llvm::dyn_cast<const clang::TemplateSpecializationType>(instanceType.getTypePtr());

   const clang::ClassTemplateSpecializationDecl* TSTdecl
      = llvm::dyn_cast<const clang::ClassTemplateSpecializationDecl>(instanceType.getTypePtr()->getAsCXXRecordDecl());

   if (TST && TSTdecl) {

      bool wantDefault = !TClassEdit::IsStdClass(TSTdecl->getName().str().c_str()) && 0 == TClassEdit::STLKind(TSTdecl->getName().str().c_str());

      bool mightHaveChanged = false;   
      llvm::SmallVector<clang::TemplateArgument, 4> desArgs;
      unsigned int Idecl = 0, Edecl = TSTdecl->getTemplateArgs().size();
      for(clang::TemplateSpecializationType::iterator 
             I = TST->begin(), E = TST->end();
          Idecl != Edecl; 
          ++I, ++Idecl) {

         if (I != E) {
            clang::QualType SubTy = I->getAsType();
         
            if (SubTy.isNull()) {
               desArgs.push_back(*I);
               continue;
            }
            
            // Check if the type needs more desugaring and recurse.
            if (llvm::isa<clang::TemplateSpecializationType>(SubTy)) {
               mightHaveChanged = true;
               desArgs.push_back(clang::TemplateArgument(AddDefaultParameters(Ctx,
                                                                              SubTy)));
            } else 
               desArgs.push_back(*I);
         } else if (wantDefault) {

            mightHaveChanged = true;
            
            clang::QualType SubTy = TSTdecl->getTemplateArgs().get(Idecl).getAsType();
         
            if (SubTy.isNull()) {
               desArgs.push_back(*I);
               continue;
            }
            
            static llvm::SmallSet<const clang::Type*, 4> typeToSkip;
            SubTy = cling::utils::Transform::GetPartiallyDesugaredType(Ctx,SubTy,typeToSkip,/*fullyQualified=*/ true);
            SubTy = AddDefaultParameters(Ctx,SubTy);
            desArgs.push_back(clang::TemplateArgument(AddDefaultParameters(Ctx,
                                                                           SubTy)));

         }
      }

      // If we added default parameter, allocate new type in the AST.
      if (mightHaveChanged) {
         instanceType = Ctx.getTemplateSpecializationType(TST->getTemplateName(), 
                                                          desArgs.data(),
                                                          desArgs.size(),
                                                          TST->getCanonicalTypeInternal());
      }
   }
   
   if (prefix) {
      instanceType = Ctx.getElaboratedType(clang::ETK_None,prefix,instanceType);
   }
   return instanceType;
}

#include <iostream>

void ROOT::TMetaUtils::GetCppName(std::string &out, const char *in)
{
   // Return (in the argument 'output') a mangled version of the C++ symbol/type (pass as 'input')
   // that can be used in C++ as a variable name.

   out.resize(strlen(in)*2);
   unsigned int i=0,j=0,c;
   while((c=in[i])) {
      if (out.capacity() < (j+3)) {
         out.resize(2*j);
      }
      switch(c) {
         case '+': strcpy(const_cast<char*>(out.data())+j,"pL"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '-': strcpy(const_cast<char*>(out.data())+j,"mI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '*': strcpy(const_cast<char*>(out.data())+j,"mU"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '/': strcpy(const_cast<char*>(out.data())+j,"dI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '&': strcpy(const_cast<char*>(out.data())+j,"aN"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '%': strcpy(const_cast<char*>(out.data())+j,"pE"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '|': strcpy(const_cast<char*>(out.data())+j,"oR"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '^': strcpy(const_cast<char*>(out.data())+j,"hA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '>': strcpy(const_cast<char*>(out.data())+j,"gR"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '<': strcpy(const_cast<char*>(out.data())+j,"lE"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '=': strcpy(const_cast<char*>(out.data())+j,"eQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '~': strcpy(const_cast<char*>(out.data())+j,"wA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '.': strcpy(const_cast<char*>(out.data())+j,"dO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '(': strcpy(const_cast<char*>(out.data())+j,"oP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ')': strcpy(const_cast<char*>(out.data())+j,"cP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '[': strcpy(const_cast<char*>(out.data())+j,"oB"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ']': strcpy(const_cast<char*>(out.data())+j,"cB"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '!': strcpy(const_cast<char*>(out.data())+j,"nO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ',': strcpy(const_cast<char*>(out.data())+j,"cO"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '$': strcpy(const_cast<char*>(out.data())+j,"dA"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ' ': strcpy(const_cast<char*>(out.data())+j,"sP"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case ':': strcpy(const_cast<char*>(out.data())+j,"cL"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '"': strcpy(const_cast<char*>(out.data())+j,"dQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '@': strcpy(const_cast<char*>(out.data())+j,"aT"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '\'': strcpy(const_cast<char*>(out.data())+j,"sQ"); j+=2; break; // Okay: we resized the underlying buffer if needed
         case '\\': strcpy(const_cast<char*>(out.data())+j,"fI"); j+=2; break; // Okay: we resized the underlying buffer if needed
         default: out[j++]=c; break;
      }
      ++i;
   }
   out.resize(j);
   return;
}

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
