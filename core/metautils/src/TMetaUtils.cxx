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
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Preprocessor.h"

#include "clang/Sema/Sema.h"

#include "cling/Utils/AST.h"
#include "cling/Interpreter/LookupHelper.h"

#include "llvm/Support/PathV2.h"

#include "cling/Interpreter/Interpreter.h"

// #define R__CINTWITHCLING_MODULES 1

// #define R__HAS_PATCH_TO_MAKE_EXPANSION_WORK_WITH_NON_CANONICAL_TYPE 1

//////////////////////////////////////////////////////////////////////////
ROOT::TMetaUtils::TNormalizedCtxt::TNormalizedCtxt(const cling::LookupHelper &lh)
{   
   // Initialize the list of typedef to keep (i.e. make them opaque for normalization).
   // This might be specific to an interpreter.

   if (fTypeToSkip.empty()) {
      clang::QualType toSkip = lh.findType("Double32_t");
      if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
      toSkip = lh.findType("Float16_t");
      if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
      toSkip = lh.findType("string");
      if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
      toSkip = lh.findType("std::string");
      if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
   }
}

//////////////////////////////////////////////////////////////////////////
static
clang::NestedNameSpecifier* AddDefaultParametersNNS(const clang::ASTContext& Ctx,
                                                    clang::NestedNameSpecifier* scope,
                                                    const cling::Interpreter &interpreter,
                                                    const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt) {
    // Add default parameter to the scope if needed.

   const clang::Type* scope_type = scope->getAsType();
   if (scope_type) {
      // this is not a namespace, so we might need to desugar
     clang::NestedNameSpecifier* outer_scope = scope->getPrefix();
      if (outer_scope) {
         outer_scope = AddDefaultParametersNNS(Ctx, outer_scope, interpreter, normCtxt);
      }

      clang::QualType addDefault = 
         ROOT::TMetaUtils::AddDefaultParameters(clang::QualType(scope_type,0), interpreter, normCtxt );
      // NOTE: Should check whether the type has changed or not.
      return clang::NestedNameSpecifier::Create(Ctx,outer_scope,
                                                false /* template keyword wanted */,
                                                addDefault.getTypePtr());
   }
   return scope;
}


//////////////////////////////////////////////////////////////////////////
clang::QualType ROOT::TMetaUtils::AddDefaultParameters(clang::QualType instanceType,
                                                       const cling::Interpreter &interpreter,
                                                       const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
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

   const clang::ASTContext& Ctx = interpreter.getCI()->getASTContext();

   // In case of Int_t* we need to strip the pointer first, desugar and attach
   // the pointer once again.
   if (instanceType->isPointerType()) {
      // Get the qualifiers.
      clang::Qualifiers quals = instanceType.getQualifiers();      
      instanceType = AddDefaultParameters(instanceType->getPointeeType(), interpreter, normCtxt);
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
      instanceType = AddDefaultParameters(instanceType->getPointeeType(), interpreter, normCtxt);

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
   clang::Qualifiers prefix_qualifiers = instanceType.getLocalQualifiers();
   const clang::ElaboratedType* etype 
      = llvm::dyn_cast<clang::ElaboratedType>(instanceType.getTypePtr());
   if (etype) {
      // We have to also handle the prefix.
 
      prefix = AddDefaultParametersNNS(Ctx, etype->getQualifier(), interpreter, normCtxt);
      instanceType = clang::QualType(etype->getNamedType().getTypePtr(),0);
   }

   // In case of template specializations iterate over the arguments and 
   // add unspecified default parameter.

   const clang::TemplateSpecializationType* TST 
      = llvm::dyn_cast<const clang::TemplateSpecializationType>(instanceType.getTypePtr());

   const clang::ClassTemplateSpecializationDecl* TSTdecl
      = llvm::dyn_cast_or_null<const clang::ClassTemplateSpecializationDecl>(instanceType.getTypePtr()->getAsCXXRecordDecl());

   if (TST && TSTdecl) {

      bool wantDefault = !TClassEdit::IsStdClass(TSTdecl->getName().str().c_str()) && 0 == TClassEdit::STLKind(TSTdecl->getName().str().c_str());

#ifdef R__HAS_PATCH_TO_MAKE_EXPANSION_WORK_WITH_NON_CANONICAL_TYPE
      clang::Sema& S = interpreter.getCI()->getSema();
#endif
      clang::TemplateDecl *Template = TSTdecl->getSpecializedTemplate()->getMostRecentDecl();
      clang::TemplateParameterList *Params = Template->getTemplateParameters();
      clang::TemplateParameterList::iterator Param = Params->begin(); // , ParamEnd = Params->end();
      //llvm::SmallVectorImpl<TemplateArgument> Converted; // Need to contains the other arguments.
      // Converted seems to be the same as our 'desArgs'

      bool mightHaveChanged = false;   
      llvm::SmallVector<clang::TemplateArgument, 4> desArgs;
      unsigned int Idecl = 0, Edecl = TSTdecl->getTemplateArgs().size();
      for(clang::TemplateSpecializationType::iterator 
             I = TST->begin(), E = TST->end();
          Idecl != Edecl; 
          I!=E ? ++I : 0, ++Idecl, ++Param) {

         if (I != E) {
            clang::QualType SubTy = I->getAsType();
         
            if (SubTy.isNull()) {
               desArgs.push_back(*I);
               continue;
            }
            
            // Check if the type needs more desugaring and recurse.
            if (llvm::isa<clang::TemplateSpecializationType>(SubTy)
                || llvm::isa<clang::ElaboratedType>(SubTy) ) {
               mightHaveChanged = true;
               desArgs.push_back(clang::TemplateArgument(AddDefaultParameters(SubTy,
                                                                              interpreter,
                                                                              normCtxt)));
            } else {
               desArgs.push_back(*I);
            }
            // Converted.push_back(TemplateArgument(ArgTypeForTemplate));
         } else if (wantDefault) {

            mightHaveChanged = true;
            
            clang::QualType SubTy = TSTdecl->getTemplateArgs().get(Idecl).getAsType();

            if (SubTy.isNull()) {
               desArgs.push_back(TSTdecl->getTemplateArgs().get(Idecl));
               continue;
            }
            
#ifdef R__HAS_PATCH_TO_MAKE_EXPANSION_WORK_WITH_NON_CANONICAL_TYPE
            clang::SourceLocation TemplateLoc = Template->getSourceRange ().getBegin(); //NOTE: not sure that this is the 'right' location.
            clang::SourceLocation RAngleLoc = TSTdecl->getSourceRange().getBegin(); // NOTE: most likely wrong, I think this is expecting the location of right angle
 
            clang::TemplateTypeParmDecl *TTP = llvm::dyn_cast<clang::TemplateTypeParmDecl>(*Param);
            clang::TemplateArgumentLoc ArgType = S.SubstDefaultTemplateArgumentIfAvailable(
                                                             Template,
                                                             TemplateLoc,
                                                             RAngleLoc,
                                                             TTP,
                                                             desArgs);
            clang::QualType BetterSubTy = ArgType.getArgument().getAsType();

            SubTy = cling::utils::Transform::GetPartiallyDesugaredType(Ctx,BetterSubTy,normCtxt.GetTypeToSkip(),/*fullyQualified=*/ true);
#else
            SubTy = cling::utils::Transform::GetPartiallyDesugaredType(Ctx,SubTy,normCtxt.GetTypeToSkip(),/*fullyQualified=*/ true);
#endif
            SubTy = AddDefaultParameters(SubTy,interpreter,normCtxt);
            desArgs.push_back(clang::TemplateArgument(SubTy));
         } else {
            // We are past the end of the list of specified arguements and we
            // do not want to add the default, no need to continue.
            break;
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
      instanceType = Ctx.getQualifiedType(instanceType,prefix_qualifiers);
   }
   return instanceType;
}


//////////////////////////////////////////////////////////////////////////
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


//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
void ROOT::TMetaUtils::GetNormalizedName(std::string &norm_name, const clang::QualType &type, const cling::Interpreter &interpreter, const TNormalizedCtxt &normCtxt) 
{
   // Return the type name normalized for ROOT,
   // keeping only the ROOT opaque typedef (Double32_t, etc.) and
   // adding default template argument for all types except the STL collections
   // where we remove the default template argument if any.
   //
   // This routine might actually belongs in the interpreter because
   // cache the clang::Type might be intepreter specific.

   clang::ASTContext &ctxt = interpreter.getCI()->getASTContext();

   clang::QualType normalizedType = cling::utils::Transform::GetPartiallyDesugaredType(ctxt, type, normCtxt.GetTypeToSkip(), true /* fully qualify */); 
   // Readd missing default template parameter.
   normalizedType = ROOT::TMetaUtils::AddDefaultParameters(normalizedType, interpreter, normCtxt);
   
   std::string normalizedNameStep1;
   normalizedType.getAsStringInternal(normalizedNameStep1,ctxt.getPrintingPolicy());
   
   // Still remove the std:: and default template argument and insert the Long64_t
   TClassEdit::TSplitType splitname(normalizedNameStep1.c_str(),(TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd | TClassEdit::kDropStlDefault));
   splitname.ShortType(norm_name,TClassEdit::kDropStd | TClassEdit::kDropStlDefault );

   // And replace basic_string<char>.  NOTE: we should probably do this at the same time as the GetPartiallyDesugaredType ... but were do we stop ?
   static const char* basic_string_s = "basic_string<char>";
   static const unsigned int basic_string_len = strlen(basic_string_s);
   int pos = 0;
   while( (pos = norm_name.find( basic_string_s,pos) ) >=0 ) {
      norm_name.replace(pos,basic_string_len, "string");
   }
}

//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
std::string ROOT::TMetaUtils::GetModuleFileName(const char* moduleName)
{
   // Return the dictionary file name for a module
   std::string dictFileName(moduleName);
   dictFileName += "_rdict.pcm";
   return dictFileName;
}

//////////////////////////////////////////////////////////////////////////
clang::Module* ROOT::TMetaUtils::declareModuleMap(clang::CompilerInstance* CI,
                                                  const char* moduleFileName,
                                                  const char* headers[])
{
   // Declare a virtual module.map to clang. Returns Module on success.
   clang::Preprocessor& PP = CI->getPreprocessor();
   clang::ModuleMap& ModuleMap = PP.getHeaderSearchInfo().getModuleMap();

   llvm::StringRef moduleName = llvm::sys::path::filename(moduleFileName);
   moduleName = llvm::sys::path::stem(moduleName);

   std::pair<clang::Module*, bool> modCreation;

   modCreation
      = ModuleMap.findOrCreateModule(moduleName.str().c_str(),
                                     0 /*ActiveModule*/,
                                     false /*Framework*/, false /*Explicit*/);
   if (!modCreation.second) {
      std::cerr << "TMetaUtils::declareModuleMap: "
         "Duplicate definition of dictionary module "
                << moduleFileName << std::endl;
            /*"\nOriginal module was found in %s.", - if only we could...*/
      // Go on, add new headers nonetheless.
   }

   clang::HeaderSearch& HdrSearch = PP.getHeaderSearchInfo();
   for (const char** hdr = headers; *hdr; ++hdr) {
      const clang::DirectoryLookup* CurDir;
      const clang::FileEntry* hdrFileEntry
         =  HdrSearch.LookupFile(*hdr, false /*isAngled*/, 0 /*FromDir*/,
                                 CurDir, 0 /*CurFileEnt*/, 0 /*SearchPath*/,
                                 0 /*RelativePath*/, 0 /*SuggestedModule*/);
      if (!hdrFileEntry) {
         std::cerr << "TMetaUtils::declareModuleMap: "
            "Cannot find header file " << *hdr
                   << " included in dictionary module "
                   << moduleName.data()
                   << " in include search path!";
         hdrFileEntry = PP.getFileManager().getFile(*hdr, /*OpenFile=*/false,
                                                    /*CacheFailure=*/false);
      } else {
         // Tell HeaderSearch that the header's directory has a module.map
         llvm::StringRef srHdrDir(hdrFileEntry->getName());
         srHdrDir = llvm::sys::path::parent_path(srHdrDir);
         const clang::DirectoryEntry* Dir
            = PP.getFileManager().getDirectory(srHdrDir);
         if (Dir) {
#ifdef R__CINTWITHCLING_MODULES
            HdrSearch.setDirectoryHasModuleMap(Dir);
#endif
         }
      }

#ifdef R__CINTWITHCLING_MODULES
      ModuleMap.addHeader(modCreation.first, hdrFileEntry);
#endif
   } // for headers
   return modCreation.first;
}

llvm::StringRef ROOT::TMetaUtils::GetComment(const clang::Decl &decl, clang::SourceLocation *loc)
{
   clang::SourceManager& sourceManager = decl.getASTContext().getSourceManager();
   clang::SourceLocation sourceLocation;
   // Guess where the comment start.
   // if (const clang::TagDecl *TD = llvm::dyn_cast<clang::TagDecl>(&decl)) {
   //    if (TD->isThisDeclarationADefinition())
   //       sourceLocation = TD->getBodyRBrace();
   // }
   //else 
   if (const clang::FunctionDecl *FD = llvm::dyn_cast<clang::FunctionDecl>(&decl)) {
      if (FD->isThisDeclarationADefinition()) {
         // We have to consider the argument list, because the end of decl is end of its name
         // Maybe this will be better when we have { in the arg list (eg. lambdas)
         if (FD->getNumParams())
            sourceLocation = FD->getParamDecl(FD->getNumParams() - 1)->getLocEnd();
         else
            sourceLocation = FD->getLocEnd();

         // Skip the last )
         sourceLocation = sourceLocation.getLocWithOffset(1);

         //sourceLocation = FD->getBodyLBrace();
      }
      else {
         //SkipUntil(commentStart, ';');
      }
   }

   if (sourceLocation.isInvalid())
      sourceLocation = decl.getLocEnd();

   // If the location is a macro get the expansion location.
   sourceLocation = sourceManager.getExpansionRange(sourceLocation).second;

   bool invalid;
   const char *commentStart = sourceManager.getCharacterData(sourceLocation, &invalid);
   if (invalid)
      return "";

   // The decl end of FieldDecl is the end of the type, sometimes excluding the declared name
   if (llvm::isa<clang::FieldDecl>(&decl)) {
      // Find the semicolon.
      while(*commentStart != ';')
         ++commentStart; 
   }

   // Find the end of declaration:
   // When there is definition Comments must be between ) { when there is definition. 
   // Eg. void f() //comment 
   //     {}
   if (!decl.hasBody())
      while (*commentStart !=';' && *commentStart != '\n' && *commentStart != '\r')
         ++commentStart;

   // Eat up the last char of the declaration if wasn't newline or comment terminator
   if (*commentStart != '\n' && *commentStart != '\r' && *commentStart != '{')
      ++commentStart;

   // Now skip the spaces and beginning of comments.
   while ( (isspace(*commentStart) || *commentStart == '/') 
           && *commentStart != '\n' && *commentStart != '\r') {
      ++commentStart;
   }

   const char* commentEnd = commentStart;
   while (*commentEnd != '\n' && *commentEnd != '\r' && *commentEnd != '{') {
      ++commentEnd;
   }

   if (loc) {
      // Find the true beginning of a comment.
      unsigned offset = commentStart - sourceManager.getCharacterData(sourceLocation);
      *loc = sourceLocation.getLocWithOffset(offset - 1);
   }

   return llvm::StringRef(commentStart, commentEnd - commentStart);
}
