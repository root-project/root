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
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Attr.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/ModuleMap.h"
#include "clang/Lex/Preprocessor.h"

#include "clang/Sema/Sema.h"

#include "cling/Utils/AST.h"
#include "cling/Interpreter/LookupHelper.h"

#include "llvm/Support/PathV2.h"

#include "cling/Interpreter/Interpreter.h"

// #define R__HAS_PATCH_TO_MAKE_EXPANSION_WORK_WITH_NON_CANONICAL_TYPE 1

//////////////////////////////////////////////////////////////////////////
ROOT::TMetaUtils::TNormalizedCtxt::TNormalizedCtxt(const cling::LookupHelper &lh)
{   
   // Initialize the list of typedef to keep (i.e. make them opaque for normalization)
   // and the list of typedef whose semantic is different from their underlying type
   // (Double32_t and Float16_t).
   // This might be specific to an interpreter.

   clang::QualType toSkip = lh.findType("Double32_t");
   if (!toSkip.isNull()) {
      fTypeToSkip.insert(toSkip.getTypePtr());
      fTypeWithAlternative.insert(toSkip.getTypePtr());
   }
   toSkip = lh.findType("Float16_t");
   if (!toSkip.isNull()) {
      fTypeToSkip.insert(toSkip.getTypePtr());
      fTypeWithAlternative.insert(toSkip.getTypePtr());
   }
   toSkip = lh.findType("Long64_t");
   if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
   toSkip = lh.findType("ULong64_t");
   if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
   toSkip = lh.findType("string");
   if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
   toSkip = lh.findType("std::string");
   if (!toSkip.isNull()) fTypeToSkip.insert(toSkip.getTypePtr());
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

   // In case of name* we need to strip the pointer first, add the default and attach
   // the pointer once again.
   if (llvm::isa<clang::PointerType>(instanceType.getTypePtr())) {
      // Get the qualifiers.
      clang::Qualifiers quals = instanceType.getQualifiers();      
      instanceType = AddDefaultParameters(instanceType->getPointeeType(), interpreter, normCtxt);
      instanceType = Ctx.getPointerType(instanceType);
      // Add back the qualifiers.
      instanceType = Ctx.getQualifiedType(instanceType, quals);
   }

   // In case of Int_t& we need to strip the pointer first, desugar and attach
   // the pointer once again.
   if (llvm::isa<clang::ReferenceType>(instanceType.getTypePtr())) {
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
            if (I->getKind() != clang::TemplateArgument::Type) {
               desArgs.push_back(*I);
               continue;
            }
            
            clang::QualType SubTy = I->getAsType();
         
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

            const clang::TemplateArgument& templateArg
               = TSTdecl->getTemplateArgs().get(Idecl);
            if (templateArg.getKind() != clang::TemplateArgument::Type) {
               desArgs.push_back(templateArg);
               continue;
            }
            clang::QualType SubTy = templateArg.getAsType();

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
static bool R__IsInt(const clang::Type *type) 
{
   const clang::BuiltinType * builtin = llvm::dyn_cast<clang::BuiltinType>(type->getCanonicalTypeInternal().getTypePtr());
   if (builtin) {
      return builtin->isInteger(); // builtin->getKind() == clang::BuiltinType::Int;
   } else {
      return false;
   }
}

//////////////////////////////////////////////////////////////////////////
static bool R__IsInt(const clang::FieldDecl *field) 
{
   return R__IsInt(field->getType().getTypePtr());
}

//////////////////////////////////////////////////////////////////////////
static const clang::FieldDecl *R__GetDataMemberFromAll(const clang::CXXRecordDecl &cl, const char *what)
{
   // Return a data member name 'what' in the class described by 'cl' if any.
   
   for(clang::RecordDecl::field_iterator field_iter = cl.field_begin(), end = cl.field_end();
       field_iter != end;
       ++field_iter)
   {
      if (field_iter->getNameAsString() == what) {
         return *field_iter;
      }
   }
   return 0;
   
}

//////////////////////////////////////////////////////////////////////////
static bool CXXRecordDecl__FindOrdinaryMember(const clang::CXXBaseSpecifier *Specifier,
                                              clang::CXXBasePath &Path,
                                              void *Name
                                              )
{
   clang::RecordDecl *BaseRecord = Specifier->getType()->getAs<clang::RecordType>()->getDecl();

   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(BaseRecord);
   if (clxx == 0) return false;
   
   const clang::FieldDecl *found = R__GetDataMemberFromAll(*clxx,(const char*)Name);
   if (found) {
      // Humm, this is somewhat bad (well really bad), oh well.
      // Let's hope Paths never things its own those (it should not as far as I can tell).
      Path.Decls.first  = (clang::NamedDecl**)found;
      Path.Decls.second = 0;
      return true;
   }
//   
// This is inspired from CXXInheritance.cpp:
/*   
       RecordDecl *BaseRecord =
         Specifier->getType()->castAs<RecordType>()->getDecl();
    
   const unsigned IDNS = clang::Decl::IDNS_Ordinary | clang::Decl::IDNS_Tag | clang::Decl::IDNS_Member;
   clang::DeclarationName N = clang::DeclarationName::getFromOpaquePtr(Name);
   for (Path.Decls = BaseRecord->lookup(N);
        Path.Decls.first != Path.Decls.second;
        ++Path.Decls.first) {
      if ((*Path.Decls.first)->isInIdentifierNamespace(IDNS))
         return true;
   }
*/   
   return false;
    
}
#include <stdio.h>

//////////////////////////////////////////////////////////////////////////
static const clang::FieldDecl *R__GetDataMemberFromAllParents(const clang::CXXRecordDecl &cl, const char *what)
{
   // Return a data member name 'what' in any of the base classes of the class described by 'cl' if any.
   
   clang::CXXBasePaths Paths;
   Paths.setOrigin(const_cast<clang::CXXRecordDecl*>(&cl));
   if (cl.lookupInBases(&CXXRecordDecl__FindOrdinaryMember,
                        (void*) what,
                        Paths) )
   {
      clang::CXXBasePaths::paths_iterator iter = Paths.begin();
      if (iter != Paths.end()) {
         // See CXXRecordDecl__FindOrdinaryMember, this is, well, awkward.
         const clang::FieldDecl *found = (clang::FieldDecl *)iter->Decls.first;
         return found;
      }
   }
   return 0;
}

//////////////////////////////////////////////////////////////////////////
// ValidArrayIndex return a static string (so use it or copy it immediatly, do not
// call GrabIndex twice in the same expression) containing the size of the
// array data member.
// In case of error, or if the size is not specified, GrabIndex returns 0.
// If errnum is not null, *errnum updated with the error number:
//   Cint::G__DataMemberInfo::G__VALID     : valid array index
//   Cint::G__DataMemberInfo::G__NOT_INT   : array index is not an int
//   Cint::G__DataMemberInfo::G__NOT_DEF   : index not defined before array 
//                                          (this IS an error for streaming to disk)
//   Cint::G__DataMemberInfo::G__IS_PRIVATE: index exist in a parent class but is private
//   Cint::G__DataMemberInfo::G__UNKNOWN   : index is not known
// If errstr is not null, *errstr is updated with the address of a static
//   string containing the part of the index with is invalid.
const char* ROOT::TMetaUtils::DataMemberInfo__ValidArrayIndex(const clang::FieldDecl &m, int *errnum, const char **errstr) 
{
   llvm::StringRef title;
   
   // Try to get the comment either from the annotation or the header file if present
   if (clang::AnnotateAttr *A = m.getAttr<clang::AnnotateAttr>())
      title = A->getAnnotation();
   else
      // Try to get the comment from the header file if present
      title = ROOT::TMetaUtils::GetComment( m );

   // Let's see if the user provided us with some information
   // with the format: //[dimension] this is the dim of the array
   // dimension can be an arithmetical expression containing, literal integer,
   // the operator *,+ and - and data member of integral type.  In addition the
   // data members used for the size of the array need to be defined prior to
   // the array.
   
   if (errnum) *errnum = VALID;
   
   size_t rightbracket = title.find(']');
   if ((title[0] != '[') ||
       (rightbracket == llvm::StringRef::npos)) return 0;
   
   std::string working;
   static std::string indexvar;
   indexvar = title.substr(1,rightbracket-1).str();
   
   // now we should have indexvar=dimension
   // Let's see if this is legal.
   // which means a combination of data member and digit separated by '*','+','-'
   // First we remove white spaces.
   unsigned int i;
   size_t indexvarlen = indexvar.length();
   for ( i=0; i<=indexvarlen; i++) {
      if (!isspace(indexvar[i])) {
         working += indexvar[i];
      }
   };
   
   // Now we go through all indentifiers
   const char *tokenlist = "*+-";
   char *current = const_cast<char*>(working.c_str());
   current = strtok(current,tokenlist);
   
   while (current!=0) {
      // Check the token
      if (isdigit(current[0])) {
         for(i=0;i<strlen(current);i++) {
            if (!isdigit(current[0])) {
               // Error we only access integer.
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not an interger\n",
               //        member.MemberOf()->Name(), member.Name(), current);
               if (errstr) *errstr = current;
               if (errnum) *errnum = NOT_INT;
               return 0;
            }
         }
      } else { // current token is not a digit
         // first let's see if it is a data member:
         int found = 0;
         const clang::CXXRecordDecl *parent_clxx = llvm::dyn_cast<clang::CXXRecordDecl>(m.getParent());
         const clang::FieldDecl *index1 = R__GetDataMemberFromAll(*parent_clxx, current );
         if ( index1 ) {
            if ( R__IsInt(index1) ) {
               found = 1;
               // Let's see if it has already been written down in the
               // Streamer.
               // Let's see if we already wrote it down in the
               // streamer.
               for(clang::RecordDecl::field_iterator field_iter = parent_clxx->field_begin(), end = parent_clxx->field_end();
                   field_iter != end;
                   ++field_iter)
               {
                  if ( field_iter->getNameAsString() == m.getNameAsString() ) {
                     // we reached the current data member before
                     // reaching the index so we have not written it yet!
                     //NOTE: *** Need to print an error;
                     //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) has not been defined before the array \n",
                     //        member.MemberOf()->Name(), member.Name(), current);
                     if (errstr) *errstr = current;
                     if (errnum) *errnum = NOT_DEF;
                     return 0;
                  }
                  if ( field_iter->getNameAsString() == index1->getNameAsString() ) {
                     break;
                  }
               } // end of while (m_local.Next())
            } else {
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
               //        member.MemberOf()->Name(), member.Name(), current);
               if (errstr) *errstr = current;
               if (errnum) *errnum = NOT_INT;
               return 0;
            }
         } else {
            // There is no variable by this name in this class, let see
            // the base classes!:
            index1 = R__GetDataMemberFromAllParents( *parent_clxx, current );
            if ( index1 ) {
               if ( R__IsInt(index1) ) {
                  found = 1;
               } else {
                  // We found a data member but it is the wrong type
                  //NOTE: *** Need to print an error;
                  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
                  //  member.MemberOf()->Name(), member.Name(), current);
                  if (errnum) *errnum = NOT_INT;
                  if (errstr) *errstr = current;
                  //NOTE: *** Need to print an error;
                  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not int \n",
                  //  member.MemberOf()->Name(), member.Name(), current);
                  if (errnum) *errnum = NOT_INT;
                  if (errstr) *errstr = current;
                  return 0;
               }
               if ( found && (index1->getAccess() == clang::AS_private) ) {
                  //NOTE: *** Need to print an error;
                  //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is a private member of %s \n",
                  if (errstr) *errstr = current;
                  if (errnum) *errnum = IS_PRIVATE;
                  return 0;
               }
            }
            if (!found) {
               //NOTE: *** Need to print an error;
               //fprintf(stderr,"*** Datamember %s::%s: size of array (%s) is not known \n",
               //        member.MemberOf()->Name(), member.Name(), indexvar);
               if (errstr) *errstr = indexvar.c_str();
               if (errnum) *errnum = UNKNOWN;
               return 0;
            } // end of if not found
         } // end of if is a data member of the class
      } // end of if isdigit
      
      current = strtok(0,tokenlist);
   } // end of while loop on tokens      
   
   return indexvar.c_str();
   
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
         out.resize(2*j+3);
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

// See also cling's AST.cpp
static
clang::NestedNameSpecifier* CreateNestedNameSpecifier(const clang::ASTContext& Ctx,
                                                      clang::NamespaceDecl* cl) {
   
   clang::NamespaceDecl* outer 
      = llvm::dyn_cast_or_null<clang::NamespaceDecl>(cl->getDeclContext());
   if (outer && outer->getName().size()) {
      clang::NestedNameSpecifier* outerNNS = CreateNestedNameSpecifier(Ctx,outer);
      return clang::NestedNameSpecifier::Create(Ctx,outerNNS,
                                         cl);
   } else {
      return clang::NestedNameSpecifier::Create(Ctx, 
                                         0, /* no starting '::'*/
                                         cl);        
   }
}

// See also cling's AST.cpp
static
clang::NestedNameSpecifier* CreateNestedNameSpecifier(const clang::ASTContext& Ctx,
                                                      clang::TagDecl *cl) {
   
   clang::NamedDecl* outer = llvm::dyn_cast_or_null<clang::NamedDecl>(cl->getDeclContext());
   if (outer && outer->getName().size()) {
      clang::NestedNameSpecifier *outerNNS;
      if (cl->getDeclContext()->isNamespace()) {
         outerNNS = CreateNestedNameSpecifier(Ctx,
                                              llvm::dyn_cast<clang::NamespaceDecl>(outer));
      } else {
         outerNNS = CreateNestedNameSpecifier(Ctx,
                                              llvm::dyn_cast<clang::TagDecl>(outer));
      }
      return clang::NestedNameSpecifier::Create(Ctx,outerNNS,
                                         false /* template keyword wanted */,
                                         Ctx.getTypeDeclType(cl).getTypePtr());
   } else {
      return clang::NestedNameSpecifier::Create(Ctx, 
                                         0, /* no starting '::'*/
                                         false /* template keyword wanted */,
                                         Ctx.getTypeDeclType(cl).getTypePtr());        
   }
}

//////////////////////////////////////////////////////////////////////////
clang::QualType ROOT::TMetaUtils::GetFullyQualifiedType(const clang::QualType &qtype,
                                                        const cling::Interpreter &interpreter)
{
   // Return the fully qualified type, if we need to recurse through any template parameter, 
   // this needs to be merged somehow with GetPartialDesugaredType.

   const clang::ASTContext& Ctx = interpreter.getCI()->getASTContext();

   if (llvm::isa<clang::TypedefType>(qtype.getTypePtr())) {
      clang::QualType QT(qtype);
      clang::Decl *decl = 0;
      const clang::TypedefType* typedeftype = 
         llvm::dyn_cast_or_null<clang::TypedefType>(QT.getTypePtr());
      decl = typedeftype->getDecl();
      if (decl) {
         clang::NamedDecl* outer 
            = llvm::dyn_cast_or_null<clang::NamedDecl>(decl->getDeclContext());
         clang::NamespaceDecl* outer_ns
            = llvm::dyn_cast_or_null<clang::NamespaceDecl>(decl->getDeclContext());
         if (outer && !(outer_ns && outer_ns->isAnonymousNamespace())) {
            clang::NestedNameSpecifier* prefix = 0;
            if (outer_ns) {
               prefix = CreateNestedNameSpecifier(Ctx,outer_ns);
            } else {
               assert(llvm::isa<clang::TagDecl>(outer)&& "not in namespace of TagDecl");
               prefix = CreateNestedNameSpecifier(Ctx,
                                                  llvm::dyn_cast<clang::TagDecl>(outer));
            }
            if (prefix) QT = Ctx.getElaboratedType(clang::ETK_None,prefix,QT);
         }
      }
      return QT;
   }
   else {
      return qtype;
   }
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
   
   clang::PrintingPolicy policy(ctxt.getPrintingPolicy());
   policy.SuppressTagKeyword = true; // Never get the class or struct keyword
   policy.SuppressScope = true;      // Force the scope to be coming from a clang::ElaboratedType.
   // The scope suppression is required for getting rid of the anonymous part of the name of a class defined in an anonymous namespace.
   // This gives us more control vs not using the clang::ElaboratedType and relying on the Policy.SuppressUnwrittenScope which would
   // strip both the anonymous and the inline namespace names (and we probably do not want the later to be suppressed).

   std::string normalizedNameStep1;
   normalizedType.getAsStringInternal(normalizedNameStep1,policy);
   
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
#if 0
         // Tell HeaderSearch that the header's directory has a module.map
         llvm::StringRef srHdrDir(hdrFileEntry->getName());
         srHdrDir = llvm::sys::path::parent_path(srHdrDir);
         const clang::DirectoryEntry* Dir
            = PP.getFileManager().getDirectory(srHdrDir);
         if (Dir) {
            HdrSearch.setDirectoryHasModuleMap(Dir);
         }
#endif
      }

      ModuleMap.addHeader(modCreation.first, hdrFileEntry);
   } // for headers
   return modCreation.first;
}

//////////////////////////////////////////////////////////////////////////
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

//////////////////////////////////////////////////////////////////////////
llvm::StringRef ROOT::TMetaUtils::GetClassComment(const clang::CXXRecordDecl &decl, clang::SourceLocation *loc, const cling::Interpreter &interpreter)
{
   // Find the class comment (after the ClassDef).

   using namespace clang;
   SourceLocation commentSLoc;
   llvm::StringRef comment;
   
   Sema& sema = interpreter.getCI()->getSema();

   for(CXXRecordDecl::decl_iterator I = decl.decls_begin(), 
       E = decl.decls_end(); I != E; ++I) {
      if (!(*I)->isImplicit() 
          && (isa<CXXMethodDecl>(*I) || isa<FieldDecl>(*I) || isa<VarDecl>(*I))) {
         // For now we allow only a special macro (ClassDef) to have meaningful comments
         SourceLocation maybeMacroLoc = (*I)->getLocation();
         bool isClassDefMacro = maybeMacroLoc.isMacroID() && sema.findMacroSpelling(maybeMacroLoc, "ClassDef");
         if (isClassDefMacro) {
            while (isa<NamedDecl>(*I) && cast<NamedDecl>(*I)->getName() != "DeclFileLine")
               ++I;
            comment = ROOT::TMetaUtils::GetComment(**I, &commentSLoc);
            if (comment.size()) {
               if (loc) 
                  *loc = commentSLoc;
               return comment;
            }
         }
      }
   }
   return llvm::StringRef();
}

//////////////////////////////////////////////////////////////////////////
clang::NestedNameSpecifier* 
ReSubstTemplateArgNNS(const clang::ASTContext &Ctxt, 
                      clang::NestedNameSpecifier *scope, 
                      const clang::Type *instance)
{
   // Check if 'scope' or any of its template parameter was substituted when
   // instantiating the class template instance and replace it with the 
   // partially sugared types we have from 'instance'.
 
   const clang::Type* scope_type = scope->getAsType();      
   if (scope_type) {
      clang::NestedNameSpecifier* outer_scope = scope->getPrefix();
      if (outer_scope) {
         outer_scope = ReSubstTemplateArgNNS(Ctxt, outer_scope, instance);
      }
      clang::QualType substScope = 
         ROOT::TMetaUtils::ReSubstTemplateArg(clang::QualType(scope_type,0), instance);
      // NOTE: Should check whether the type has changed or not.
      scope = clang::NestedNameSpecifier::Create(Ctxt,outer_scope,
                                                 false /* template keyword wanted */,
                                                 substScope.getTypePtr());
   }
   return scope;
}

//////////////////////////////////////////////////////////////////////////
clang::QualType ROOT::TMetaUtils::ReSubstTemplateArg(clang::QualType input, const clang::Type *instance)
{
   // Check if 'input' or any of its template parameter was substituted when
   // instantiating the class template instance and replace it with the 
   // partially sugared types we have from 'instance'.

   if (!instance) return input;
   
   // Treat scope (clang::ElaboratedType) if any.
   const clang::ElaboratedType* etype 
      = llvm::dyn_cast<clang::ElaboratedType>(input.getTypePtr());
   if (etype) {
      // We have to also handle the prefix.
      
      clang::Qualifiers scope_qualifiers = input.getLocalQualifiers();
      assert(instance->getAsCXXRecordDecl()!=0 && "ReSubstTemplateArg only makes sense with a type representing a class.");
      const clang::ASTContext &Ctxt = instance->getAsCXXRecordDecl()->getASTContext();
     
      clang::NestedNameSpecifier *scope = ReSubstTemplateArgNNS(Ctxt,etype->getQualifier(),instance);
      clang::QualType subTy = ReSubstTemplateArg(clang::QualType(etype->getNamedType().getTypePtr(),0),instance);
 
      subTy = Ctxt.getElaboratedType(clang::ETK_None,scope,subTy);
      subTy = Ctxt.getQualifiedType(subTy,scope_qualifiers);
      return subTy;
   }

   // If the instance is also an elaborated type, we need to skip
   etype = llvm::dyn_cast<clang::ElaboratedType>(instance);
   if (etype) {
      instance = etype->getNamedType().getTypePtr();
      if (!instance) return input;
   }

   const clang::TemplateSpecializationType* TST 
      = llvm::dyn_cast<const clang::TemplateSpecializationType>(instance);
   
   if (!TST) return input;
 
   const clang::ClassTemplateSpecializationDecl* TSTdecl
      = llvm::dyn_cast_or_null<const clang::ClassTemplateSpecializationDecl>(instance->getAsCXXRecordDecl());

   const clang::SubstTemplateTypeParmType *substType
      = llvm::dyn_cast<clang::SubstTemplateTypeParmType>(input.getTypePtr());

   if (substType) {
      // Make sure it got replaced from this template
      const clang::ClassTemplateDecl *replacedCtxt = 0;
                                               
      const clang::DeclContext *replacedDeclCtxt = substType->getReplacedParameter()->getDecl()->getDeclContext();
      const clang::CXXRecordDecl *decl = llvm::dyn_cast<clang::CXXRecordDecl>(replacedDeclCtxt);
      if (decl) {
         if (decl->getKind() == clang::Decl::ClassTemplatePartialSpecialization) {
            replacedCtxt = llvm::dyn_cast<clang::ClassTemplatePartialSpecializationDecl>(decl)->getSpecializedTemplate();
         } else {
            replacedCtxt = decl->getDescribedClassTemplate();
         }
      } else {
         replacedCtxt = llvm::dyn_cast<clang::ClassTemplateDecl>(replacedDeclCtxt);
      }
      unsigned int index = substType->getReplacedParameter()->getIndex();
      if (replacedCtxt->getCanonicalDecl() == TSTdecl->getSpecializedTemplate()->getCanonicalDecl() 
          || /* the following is likely just redundant */
          substType->getReplacedParameter()->getDecl() 
          == TSTdecl->getSpecializedTemplate ()->getTemplateParameters()->getParam(index))
      {
         if ( index >= TST->getNumArgs() ) {
            // The argument replaced was a default template argument that is
            // being listed as part of the instance ... 
            // so we probably don't really know how to spell it ... we would need to recreate it
            // (See AddDefaultParamters).
            return input;
         } else {
            return TST->getArg(index).getAsType();
         }
      }
   }
   // Maybe a class template instance, recurse and rebuild
   const clang::TemplateSpecializationType* inputTST 
      = llvm::dyn_cast<const clang::TemplateSpecializationType>(input.getTypePtr());
   const clang::ASTContext& astCtxt = TSTdecl->getASTContext(); 

   if (inputTST) {
      bool mightHaveChanged = false;
      llvm::SmallVector<clang::TemplateArgument, 4> desArgs;
      for(clang::TemplateSpecializationType::iterator I = inputTST->begin(), E = inputTST->end();
          I != E; ++I) {
         if (I->getKind() != clang::TemplateArgument::Type) {
            desArgs.push_back(*I);
            continue;
         }

         clang::QualType SubTy = I->getAsType();
         // Check if the type needs more desugaring and recurse.
         if (llvm::isa<clang::SubstTemplateTypeParmType>(SubTy) 
             || llvm::isa<clang::TemplateSpecializationType>(SubTy)) {
            mightHaveChanged = true;
            clang::QualType newSubTy = ReSubstTemplateArg(SubTy,instance);
            if (!newSubTy.isNull()) {
               desArgs.push_back(clang::TemplateArgument(newSubTy));
            }
         } else 
            desArgs.push_back(*I);
      }
      
      // If desugaring happened allocate new type in the AST.
      if (mightHaveChanged) {
         clang::Qualifiers qualifiers = input.getLocalQualifiers();
         input = astCtxt.getTemplateSpecializationType(inputTST->getTemplateName(), 
                                                       desArgs.data(),
                                                       desArgs.size(),
                                                       inputTST->getCanonicalTypeInternal());
         input = astCtxt.getQualifiedType(input, qualifiers);
      }
   }

   return input;
}
