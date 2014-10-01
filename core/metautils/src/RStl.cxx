// @(#)root/utils:$Id$
// Author: Philippe Canal 27/08/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun, Fons Rademakers, and al.          *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS              *
 *************************************************************************/

#include "RConfigure.h"
#include "RConfig.h"

#include "RStl.h"
#include "TClassEdit.h"
#include "TMetaUtils.h"
using namespace TClassEdit;

#include <stdio.h>

#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"

#include "Scanner.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"

#include "clang/Sema/Sema.h"
#include "clang/Sema/Template.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/AST/DeclTemplate.h"

#ifndef ROOT_Varargs
#include "Varargs.h"
#endif
// void Error(const char *location, const char *va_(fmt), ...);
// void Warning(const char *location, const char *va_(fmt), ...);

//
// ROOT::RStl is the rootcint STL handling class.
//

static int fgCount = 0;

ROOT::RStl& ROOT::RStl::Instance()
{
   // Return the singleton ROOT::RStl.

   static ROOT::RStl instance;
   return instance;

}

void ROOT::RStl::GenerateTClassFor(const clang::QualType &type, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Force the generation of the TClass for the given class.

   const clang::CXXRecordDecl *stlclass = type.getTypePtr()->getAsCXXRecordDecl();
   if (stlclass == 0) {
      return;
   }
   const clang::ClassTemplateSpecializationDecl *templateCl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(stlclass);

   if (templateCl == 0) {
      ROOT::TMetaUtils::Error("RStl::GenerateTClassFor","%s not in a template",
            ROOT::TMetaUtils::GetQualifiedName(*stlclass).c_str());
   }

   if ( TClassEdit::STLKind( stlclass->getName().str().c_str() )  == ROOT::kSTLvector ) {
      const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(0) );
      if (arg.getKind() == clang::TemplateArgument::Type) {
         const clang::NamedDecl *decl = arg.getAsType().getTypePtr()->getAsCXXRecordDecl();
         if (decl) {
            // NOTE: we should just compare the decl to the bool builtin!
            llvm::StringRef argname = decl->getName();
            if ( (argname.str() == "bool") || (argname.str() == "Bool_t") ) {
               ROOT::TMetaUtils::Warning("std::vector<bool>", " is not fully supported yet!\nUse std::vector<char> or std::deque<bool> instead.\n");
            }
         }
      }
   }

   fList.insert( ROOT::TMetaUtils::AnnotatedRecordDecl(++fgCount,type.getTypePtr(),stlclass,"",false /* for backward compatibility rather than 'true' .. neither really make a difference */,false,false,false,-1, interp, normCtxt) );

   // fprintf(stderr,"Registered the STL class %s as needing a dictionary\n",R__GetQualifiedName(*stlclass).c_str());

   for(unsigned int i=0; i <  templateCl->getTemplateArgs().size(); ++i) {
      const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(i) );
      if (arg.getKind() == clang::TemplateArgument::Type) {
         const clang::NamedDecl *decl = arg.getAsType().getTypePtr()->getAsCXXRecordDecl();

         if (decl && TClassEdit::STLKind( decl->getName().str().c_str() ) != 0 )
            {
               const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
               if (clxx) {
                  if (!clxx->isCompleteDefinition()) {
                     /* bool result = */ ROOT::TMetaUtils::RequireCompleteType(interp, clxx->getLocation(), arg.getAsType());
                  }
                  // Do we need to strip the qualifier?
                  GenerateTClassFor(arg.getAsType(),interp,normCtxt);
               }
            }
      }
   }
}

void ROOT::RStl::GenerateTClassFor(const char *requestedName, const clang::CXXRecordDecl *stlclass, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Force the generation of the TClass for the given class.

   const clang::ClassTemplateSpecializationDecl *templateCl = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl>(stlclass);

   if (templateCl == 0) {
      ROOT::TMetaUtils::Error("RStl::GenerateTClassFor","%s not in a template",
            ROOT::TMetaUtils::GetQualifiedName(*stlclass).c_str());
   }


   if ( TClassEdit::STLKind( stlclass->getName().str().c_str() )  == ROOT::kSTLvector ) {
      const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(0) );
      if (arg.getKind() == clang::TemplateArgument::Type) {
         const clang::NamedDecl *decl = arg.getAsType().getTypePtr()->getAsCXXRecordDecl();
         if (decl) {
            // NOTE: we should just compare the decl to the bool builtin!
            llvm::StringRef argname = decl->getName();
            if ( (argname.str() == "bool") || (argname.str() == "Bool_t") ) {
               ROOT::TMetaUtils::Warning("std::vector<bool>", " is not fully supported yet!\nUse std::vector<char> or std::deque<bool> instead.\n");
            }
         }
      }
   }

   fList.insert( ROOT::TMetaUtils::AnnotatedRecordDecl(++fgCount,stlclass,requestedName,true,false,false,false,-1, interp,normCtxt) );

   TClassEdit::TSplitType splitType( requestedName, (TClassEdit::EModType)(TClassEdit::kLong64 | TClassEdit::kDropStd) );
   for(unsigned int i=0; i <  templateCl->getTemplateArgs().size(); ++i) {
      const clang::TemplateArgument &arg( templateCl->getTemplateArgs().get(i) );
      if (arg.getKind() == clang::TemplateArgument::Type) {
         const clang::NamedDecl *decl = arg.getAsType().getTypePtr()->getAsCXXRecordDecl();

         if (decl && TClassEdit::STLKind( decl->getName().str().c_str() ) != 0 )
            {
               const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
               if (clxx) {
                  if (!clxx->isCompleteDefinition()) {
                     /* bool result = */ ROOT::TMetaUtils::RequireCompleteType(interp, clxx->getLocation (), arg.getAsType());
                     clxx = arg.getAsType().getTypePtr()->getAsCXXRecordDecl();
                  }
                  if (!splitType.fElements.empty()) {
                     GenerateTClassFor( splitType.fElements[i+1].c_str(), clxx, interp, normCtxt);
                  } else {
                     GenerateTClassFor( "", clxx, interp, normCtxt );
                  }
               }
            }
      }
   }

   //    fprintf(stderr,"ROOT::RStl registered %s as %s\n",stlclassname.c_str(),registername.c_str());
}

void ROOT::RStl::Print()
{
   // Print the content of the object
   fprintf(stderr,"ROOT::RStl singleton\n");
   list_t::iterator iter;
   for(iter = fList.begin(); iter != fList.end(); ++iter) {
      fprintf(stderr, "need TClass for %s\n", ROOT::TMetaUtils::GetQualifiedName(*(*iter)).c_str());
   }
}

void ROOT::RStl::WriteClassInit(std::ostream &ostr,
                                const cling::Interpreter &interp,
                                const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt,
                                const ROOT::TMetaUtils::RConstructorTypes& ctorTypes,
                                bool &needCollectionProxy,
                                void (*emitStreamerInfo)(const char*) )
{
   // This function writes the TGeneraticClassInfo initialiser
   // and the auxiliary functions (new and delete wrappers) for
   // each of the STL containers that have been registered

   list_t::iterator iter;
   for(iter = fList.begin(); iter != fList.end(); ++iter) {
      const clang::CXXRecordDecl* result;

      if (!iter->GetRecordDecl()->getDefinition()) {

         // We do not have a complete definition, we need to force the instantiation
         // and findScope can do that.
         const cling::LookupHelper& lh = interp.getLookupHelper();
         result = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(lh.findScope(iter->GetNormalizedName(),
                                                                            cling::LookupHelper::NoDiagnostics,
                                                                            0)
                                                               );

         if (!result || !iter->GetRecordDecl()->getDefinition()) {
            fprintf(stderr,"Error: incomplete definition for %s\n",iter->GetNormalizedName());
            continue;
         }
      }
      else
      {
         result = llvm::dyn_cast<clang::CXXRecordDecl>(iter->GetRecordDecl());
      }

//      std::string fullname = TMetaUtils::GetQualifiedName(*iter->GetRecordDecl());
//      fprintf(stderr,"RStl is generating TClass for %ld %s %s %s\n",iter->GetRuleIndex(),iter->GetRequestedName(),iter->GetNormalizedName(),fullname.c_str());

      ROOT::TMetaUtils::WriteClassInit(ostr, *iter, result, interp, normCtxt, ctorTypes, needCollectionProxy);
      ROOT::TMetaUtils::WriteAuxFunctions(ostr, *iter, result, interp, ctorTypes, normCtxt);

      if (emitStreamerInfo) emitStreamerInfo(iter->GetNormalizedName());
   }
}

// void ROOT::RStl::WriteStreamer(FILE *file, const clang::CXXRecordDecl *stlcl)
// {
//    // Write the free standing streamer function for the given
//    // STL container class.

//    std::string streamerName = "stl_streamer_";

//    std::string shortTypeName = GetLong64_Name( TClassEdit::ShortType(ROOT::TMetaUtils::GetQualifiedName(*stlcl).c_str(),TClassEdit::kDropStlDefault) );
//    std::string noConstTypeName( TClassEdit::CleanType(shortTypeName.c_str(),2) );

//    std::string typedefName;
//    ROOT::TMetaUtils::GetCppName(typedefName, shortTypeName.c_str());
//    streamerName += typedefName;

//    const clang::ClassTemplateSpecializationDecl *tmplt_specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl> (stlcl);
//    if (!tmplt_specialization) return;

//    int stltype = TClassEdit::STLKind(tmplt_specialization->getName().str().c_str());

//    const clang::TemplateArgument &arg0( tmplt_specialization->getTemplateArgs().get(0) );

//    clang::QualType firstType = arg0.getAsType();
//    clang::QualType secondType;

//    const char *tclFirst=0,*tclSecond=0;
//    std::string firstFullName, secondFullName;

//    if (ROOT::TMetaUtils::ElementStreamer(*stlcl, firstType,0,0)) {
//       tclFirst = "R__tcl1";
//       firstFullName = firstType.getAsString();
//    }
//    if (stltype==kMap || stltype==kMultiMap) {
//       const clang::TemplateArgument &arg1( tmplt_specialization->getTemplateArgs().get(1) );
//       secondType = arg1.getAsType();

//       if (ROOT::TMetaUtils::ElementStreamer(*stlcl, secondType,0,0)) {
//          tclSecond="R__tcl2";
//          secondFullName = secondType.getAsString();
//       }
//    }

//    fprintf(file, "//___________________________________________________________");
//    fprintf(file, "_____________________________________________________________\n");
//    fprintf(file, "namespace ROOT {\n");
//    fprintf(file, "   typedef %s %s;\n",shortTypeName.c_str(), typedefName.c_str());
//    fprintf(file, "   static void %s(TBuffer &R__b, void *R__p)\n",streamerName.c_str());
//    fprintf(file, "   {\n");
//    fprintf(file, "      if (gDebug>1) Info(__FILE__,\"Running compiled streamer for %s at %%p\",R__p);\n",shortTypeName.c_str());
//    fprintf(file, "      %s &R__stl = *(%s *)R__p;\n",shortTypeName.c_str(),shortTypeName.c_str());
//    fprintf(file, "      if (R__b.IsReading()) {\n");
//    fprintf(file, "         R__stl.clear();\n");

//    if (tclFirst)
//       fprintf(file, "         TClass *R__tcl1 = TBuffer::GetClass(typeid(%s));\n",
//               firstFullName.c_str());
//    if (tclSecond)
//       fprintf(file, "         TClass *R__tcl2 = TBuffer::GetClass(typeid(%s));\n",
//               secondFullName.c_str());

//    fprintf(file, "         int R__i, R__n;\n");
//    fprintf(file, "         R__b >> R__n;\n");

//    if (stltype==kVector) {
//       fprintf(file,"         R__stl.reserve(R__n);\n");
//    }
//    fprintf(file, "         for (R__i = 0; R__i < R__n; R__i++) {\n");

//    ROOT::TMetaUtils::ElementStreamer(*stlcl, firstType,"R__t",0,tclFirst);
//    if (stltype == kMap || stltype == kMultiMap) {     //Second Arg
//       ROOT::TMetaUtils::ElementStreamer(*stlcl, secondType,"R__t2",0,tclSecond);
//    }
//    switch (stltype) {

//       case kMap:
//       case kMultiMap:
//          fprintf(file, "            R__stl.insert(make_pair(R__t,R__t2));\n");
//          break;
//       case kSet:
//       case kMultiSet:
//          fprintf(file, "            R__stl.insert(R__t);\n");
//          break;
//       case kVector:
//       case kList:
//       case kDeque:
//          fprintf(file, "            R__stl.push_back(R__t);\n");
//          break;

//       default:
//             assert(0);
//    }
//    fprintf(file, "         }\n");

//    fprintf(file, "      } else {\n");

//    fprintf(file, "         int R__n=(&R__stl) ? int(R__stl.size()) : 0;\n");
//    fprintf(file, "         R__b << R__n;\n");
//    fprintf(file, "         if(R__n) {\n");

//    if (tclFirst) {
//       fprintf(file, "            TClass *R__tcl1 = TBuffer::GetClass(typeid(%s));\n",
//               firstFullName.c_str());
//       fprintf(file, "            if (R__tcl1==0) {\n");
//       fprintf(file, "               Error(\"%s streamer\",\"Missing the TClass object for %s!\");\n",
//               shortTypeName.c_str(), firstFullName.c_str());
//       fprintf(file, "               return;\n");
//       fprintf(file, "            }\n");
//    }
//    if (tclSecond) {
//       fprintf(file, "            TClass *R__tcl2 = TBuffer::GetClass(typeid(%s));\n",
//               secondFullName.c_str());
//       fprintf(file, "            if (R__tcl2==0) {\n");
//       fprintf(file, "               Error(\"%s streamer\",\"Missing the TClass object for %s!\");\n",
//               shortTypeName.c_str(), secondFullName.c_str());
//       fprintf(file, "               return;\n");
//       fprintf(file, "            }\n");
//    }
//    fprintf(file, "            %s::iterator R__k;\n", shortTypeName.c_str());
//    fprintf(file, "            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {\n");

//    if (stltype == kMap || stltype == kMultiMap) {
//       ROOT::TMetaUtils::ElementStreamer(*stlcl, firstType ,"((*R__k).first )",1,tclFirst);
//       ROOT::TMetaUtils::ElementStreamer(*stlcl, secondType,"((*R__k).second)",1,tclSecond);
//    } else {
//       ROOT::TMetaUtils::ElementStreamer(*stlcl, firstType ,"(*R__k)"         ,1,tclFirst);
//    }

//    fprintf(file, "            }\n");
//    fprintf(file, "         }\n");

//    fprintf(file, "      }\n");
//    fprintf(file, "   } // end of %s streamer\n",ROOT::TMetaUtils::GetQualifiedName(*stlcl).c_str());
//    fprintf(file, "} // close namespace ROOT\n\n");

//    fprintf(file, "// Register the streamer (a typedef is used to avoid problem with macro parameters\n");

//    //if ( 0 != ::getenv("MY_ROOT") && ::getenv("MY_ROOT")[0]>'1' )  {
//    //  fprintf(file, "// Disabled due customized build:\n// ");
//    //}
//    fprintf(file, "RootStlStreamer(%s,%s)\n", typedefName.c_str(), streamerName.c_str());
//    fprintf(file, "\n");

// }

// void ROOT::RStl::WriteStreamer(FILE *file)
// {
//    // Write the free standing streamer function for the registereed
//    // STL container classes

//    list_t::iterator iter;
//    for(iter = fList.begin(); iter != fList.end(); ++iter) {
//       WriteStreamer(file,llvm::dyn_cast<clang::CXXRecordDecl>(iter->GetRecordDecl()));
//    }
// }
