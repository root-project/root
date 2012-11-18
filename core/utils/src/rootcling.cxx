// @(#)root/utils:$Id$
// Author: Fons Rademakers   13/07/96

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/rootcint.            *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// CREDITS                                                              //
//                                                                      //
// This program generates the CINT dictionaries needed in order to      //
// get access to your classes via the interpreter.                      //
// In addition rootcint can generate the Streamer(),                    //
// TBuffer &operator>>() and ShowMembers() methods for ROOT classes,    //
// i.e. classes using the ClassDef and ClassImp macros.                 //
//                                                                      //
// Rootcint can be used like:                                           //
//                                                                      //
//  rootcint TAttAxis.h[{+,-}][!] ... [LinkDef.h] > AxisDict.cxx        //
//                                                                      //
// or                                                                   //
//                                                                      //
//  rootcint [-v[0-4]][-l][-f] dict.C [-c] [-p]                         //
//           file.h[{+,-}][!] ... [LinkDef.h]                           //
//                                                                      //
// The difference between the two is that in the first case only the    //
// Streamer() and ShowMembers() methods are generated while in the      //
// latter case a complete compileable file is generated (including      //
// the include statements). The first method also allows the            //
// output to be appended to an already existing file (using >>).        //
// The optional - behind the header file name tells rootcint to not     //
// generate the Streamer() method. A custom method must be provided     //
// by the user in that case. For the + and ! options see below.         //
// When using option -c also the interpreter method interface stubs     //
// will be written to the output file (AxisDict.cxx in the above case). //
// By default the output file will not be overwritten if it exists.     //
// Use the -f (force) option to overwite the output file. The output    //
// file must have one of the following extensions: .cxx, .C, .cpp,      //
// .cc, .cp.                                                            //
// Use the -p option to request the use of the compiler's preprocessor  //
// instead of CINT's preprocessor.  This is useful to handle header\n"  //
// files with macro construct not handled by CINT.\n\n"                 //
// Use the -l (long) option to prepend the pathname of the              //
// dictionary source file to the include of the dictionary header.      //
// This might be needed in case the dictionary file needs to be         //
// compiled with the -I- option that inhibits the use of the directory  //
// of the source file as the first search directory for                 //
// "#include "file"".                                                   //
// The flag --lib-list-prefix=xxx can be used to produce a list of      //
// libraries needed by the header files being parsed. Rootcint will     //
// read the content of xxx.in for a list of rootmap files (see          //
// rlibmap). Rootcint will read these files and use them to deduce a    //
// list of libraries that are needed to properly link and load this     //
// dictionary. This list of libraries is saved in the first line of the //
// file xxx.out; the remaining lines contains the list of classes for   //
// which this run of rootcint produced a dictionary.                    //
// This feature is used by ACliC (the automatic library generator).     //
// The verbose flags have the following meaning:                        //
//      -v   Display all messages                                       //
//      -v0  Display no messages at all.                                //
//      -v1  Display only error messages (default).                     //
//      -v2  Display error and warning messages.                        //
//      -v3  Display error, warning and note messages.                  //
//      -v4  Display all messages                                       //
// rootcint also support the other CINT options (see 'cint -h)          //
//                                                                      //
// Before specifying the first header file one can also add include     //
// file directories to be searched and preprocessor defines, like:      //
//   -I$MYPROJECT/include -DDebug=1                                     //
//                                                                      //
// The (optional) file LinkDef.h looks like:                            //
//                                                                      //
// #ifdef __CINT__                                                      //
//                                                                      //
// #pragma link off all globals;                                        //
// #pragma link off all classes;                                        //
// #pragma link off all functions;                                      //
//                                                                      //
// #pragma link C++ class TAxis;                                        //
// #pragma link C++ class TAttAxis-;                                    //
// #pragma link C++ class TArrayC-!;                                    //
// #pragma link C++ class AliEvent+;                                    //
//                                                                      //
// #pragma link C++ function StrDup;                                    //
// #pragma link C++ function operator+(const TString&,const TString&);  //
//                                                                      //
// #pragma link C++ global gROOT;                                       //
// #pragma link C++ global gEnv;                                        //
//                                                                      //
// #pragma link C++ enum EMessageTypes;                                 //
//                                                                      //
// #endif                                                               //
//                                                                      //
// This file tells rootcint for which classes the method interface      //
// stubs should be generated. A trailing - in the class name tells      //
// rootcint to not generate the Streamer() method. This is necessary    //
// for those classes that need a customized Streamer() method.          //
// A trailing ! in the class name tells rootcint to not generate the    //
// operator>>(TBuffer &b, MyClass *&obj) function. This is necessary to //
// be able to write pointers to objects of classes not inheriting from  //
// TObject. See for an example the source of the TArrayF class.         //
// If the class contains a ClassDef macro, a trailing + in the class    //
// name tells rootcint to generate an automatic Streamer(), i.e. a      //
// streamer that let ROOT do automatic schema evolution. Otherwise, a   //
// trailing + in the class name tells rootcint to generate a ShowMember //
// function and a Shadow Class. The + option is mutually exclusive with //
// the - option. For new classes the + option is the                    //
// preferred option. For legacy reasons it is not yet the default.      //
// When the linkdef file is not specified a default version exporting   //
// the classes with the names equal to the include files minus the .h   //
// is generated.                                                        //
//                                                                      //
// *** IMPORTANT ***                                                    //
// 1) LinkDef.h must be the last argument on the rootcint command line. //
// 2) Note that the LinkDef file name MUST contain the string:          //
//    LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine    //
//    just like, linkdef1.h. Linkdef.h is case sensitive.               //
//                                                                      //
// The default constructor used by the ROOT I/O can be customized by    //
// using the rootcint pragma:                                           //
//    #pragma link C++ ioctortype UserClass;                            //
// For example, with this pragma and a class named MyClass,             //
// this method will called the first of the following 3                 //
// constructors which exists and is public:                             //
//    MyClass(UserClass*);                                              //
//    MyClass(TRootIOCtor*);                                            //
//    MyClass(); // Or a constructor with all its arguments defaulted.  //
//                                                                      //
// When more than one pragma ioctortype is used, the first seen has     //
// priority.  For example with:                                         //
//    #pragma link C++ ioctortype UserClass1;                           //
//    #pragma link C++ ioctortype UserClass2;                           //
// We look in the following order:                                      //
//    MyClass(UserClass1*);                                             //
//    MyClass(UserClass2*);                                             //
//    MyClass(TRootIOCtor*);                                            //
//    MyClass(); // Or a constructor with all its arguments defaulted.  //
//                                                                      //
// ----------- historical ---------                                     //
//                                                                      //
// Note that the file rootcint.C is constructed in such a way that it   //
// can also be interpreted by CINT. The above two statements become in  //
// that case:                                                           //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            TAttAxis.h TAxis.h > AxisGen.C            //
//                                                                      //
// or                                                                   //
//                                                                      //
// cint -I$ROOTSYS/include +V TAttAxis.h TAxis.h LinkDef.h rootcint.C \ //
//                            AxisGen.C TAttAxis.h TAxis.h              //
//                                                                      //
// The +V and -I$ROOTSYS/include options are added to the list of       //
// arguments in the compiled version of rootcint.                       //
//////////////////////////////////////////////////////////////////////////

#include "RConfigure.h"
#include "RConfig.h"
#include "Rtypes.h"

#include <iostream>
#include <memory>
#include <vector>

#include "cintdictversion.h"

#include "cling/Interpreter/Interpreter.h"
#include "cling/Interpreter/LookupHelper.h"
#include "cling/Interpreter/Value.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Pragma.h"
#include "clang/Sema/Sema.h"
#include "clang/Serialization/ASTWriter.h"
#include "cling/Utils/AST.h"

#include "llvm/Bitcode/BitstreamWriter.h"

#ifdef __APPLE__
#include <libgen.h> // Needed for basename
#include <mach-o/dyld.h>
#endif

#if defined(R__WIN32)
#include "cygpath.h"
#endif

#ifdef ROOTBUILD
# define ROOTBUILDVAL true
#else
# define ROOTBUILDVAL false
#endif

#ifdef R__EXTERN_LLVMDIR
# define R__LLVMDIR R__EXTERN_LLVMDIR
#else
# define R__LLVMDIR "./interpreter/llvm/inst" // only works for rootbuild for now!
#endif

template <typename T> struct R__IsPointer { enum { kVal = 0 }; };

template <typename T> struct R__IsPointer<T*> { enum { kVal = 1 }; };

const char *ShortTypeName(const char *typeDesc);

const char *help =
"\n"
"This program generates the CINT dictionaries needed in order to\n"
"get access to your classes via the interpreter.\n"
"In addition rootcint can generate the Streamer(), TBuffer &operator>>()\n"
"and ShowMembers() methods for ROOT classes, i.e. classes using the\n"
"ClassDef and ClassImp macros.\n"
"\n"
"Rootcint can be used like:\n"
"\n"
"  rootcint TAttAxis.h[{+,-}][!] ... [LinkDef.h] > AxisDict.cxx\n"
"\n"
"or\n"
"\n"
"  rootcint [-v[0-4]] [-l] [-f] dict.C [-c] [-p] TAxis.h[{+,-}][!] ... [LinkDef.h] \n"
"\n"
"The difference between the two is that in the first case only the\n"
"Streamer() and ShowMembers() methods are generated while in the\n"
"latter case a complete compileable file is generated (including\n"
"the include statements). The first method also allows the\n"
"output to be appended to an already existing file (using >>).\n"
"The optional - behind the header file name tells rootcint\n"
"to not generate the Streamer() method. A custom method must be\n"
"provided by the user in that case. For the + and ! options see below.\n\n"
"When using option -c also the interpreter method interface stubs\n"
"will be written to the output file (AxisDict.cxx in the above case).\n"
"By default the output file will not be overwritten if it exists.\n"
"Use the -f (force) option to overwite the output file. The output\n"
"file must have one of the following extensions: .cxx, .C, .cpp,\n"
".cc, .cp.\n\n"
"Use the -p option to request the use of the compiler's preprocessor\n"
"instead of CINT's preprocessor.  This is useful to handle header\n"
"files with macro construct not handled by CINT.\n\n"
"Use the -l (long) option to prepend the pathname of the\n"
"dictionary source file to the include of the dictionary header.\n"
"This might be needed in case the dictionary file needs to be\n"
"compiled with the -I- option that inhibits the use of the directory\n"
"of the source file as the first search directory for\n"
"\"#include \"file\"\".\n"
"The flag --lib-list-prefix=xxx can be used to produce a list of\n"
"libraries needed by the header files being parsed. Rootcint will\n"
"read the content of xxx.in for a list of rootmap files (see\n"
"rlibmap). Rootcint will read these files and use them to deduce a\n"
"list of libraries that are needed to properly link and load this\n"
"dictionary. This list of libraries is saved in the file xxx.out.\n"
"This feature is used by ACliC (the automatic library generator).\n"
"The verbose flags have the following meaning:\n"
"      -v   Display all messages\n"
"      -v0  Display no messages at all.\n"
"      -v1  Display only error messages (default).\n"
"      -v2  Display error and warning messages.\n"
"      -v3  Display error, warning and note messages.\n"
"      -v4  Display all messages\n"
"rootcint also support the other CINT options (see 'cint -h)\n"
"\n"
"Before specifying the first header file one can also add include\n"
"file directories to be searched and preprocessor defines, like:\n"
"   -I../include -DDebug\n"
"\n"
"The (optional) file LinkDef.h looks like:\n"
"\n"
"#ifdef __CINT__\n"
"\n"
"#pragma link off all globals;\n"
"#pragma link off all classes;\n"
"#pragma link off all functions;\n"
"\n"
"#pragma link C++ class TAxis;\n"
"#pragma link C++ class TAttAxis-;\n"
"#pragma link C++ class TArrayC-!;\n"
"#pragma link C++ class AliEvent+;\n"
"\n"
"#pragma link C++ function StrDup;\n"
"#pragma link C++ function operator+(const TString&,const TString&);\n"
"\n"
"#pragma link C++ global gROOT;\n"
"#pragma link C++ global gEnv;\n"
"\n"
"#pragma link C++ enum EMessageTypes;\n"
"\n"
"#endif\n"
"\n"
"This file tells rootcint for which classes the method interface\n"
"stubs should be generated. A trailing - in the class name tells\n"
"rootcint to not generate the Streamer() method. This is necessary\n"
"for those classes that need a customized Streamer() method.\n"
"A trailing ! in the class name tells rootcint to not generate the\n"
"operator>>(TBuffer &b, MyClass *&obj) method. This is necessary to\n"
"be able to write pointers to objects of classes not inheriting from\n"
"TObject. See for an example the source of the TArrayF class.\n"
"If the class contains a ClassDef macro, a trailing + in the class\n"
"name tells rootcint to generate an automatic Streamer(), i.e. a\n"
"streamer that let ROOT do automatic schema evolution. Otherwise, a\n"
"trailing + in the class name tells rootcint to generate a ShowMember\n"
"function and a Shadow Class. The + option is mutually exclusive with\n"
"the - option. For new classes the + option is the\n"
"preferred option. For legacy reasons it is not yet the default.\n"
"When this linkdef file is not specified a default version exporting\n"
"the classes with the names equal to the include files minus the .h\n"
"is generated.\n"
"\n"
"*** IMPORTANT ***\n"
"1) LinkDef.h must be the last argument on the rootcint command line.\n"
"2) Note that the LinkDef file name MUST contain the string:\n"
"   LinkDef.h, Linkdef.h or linkdef.h, i.e. NA49_LinkDef.h is fine,\n"
"   just like linkdef1.h. Linkdef.h is case sensitive.\n";

#ifdef _WIN32
#ifdef system
#undef system
#endif
#include <windows.h>
#include <Tlhelp32.h> // for MAX_MODULE_NAME32
#include <process.h>
#endif

#include <errno.h>
#include <time.h>
#include <string>
#include <list>
#include <vector>
#include <sstream>
#include <map>
#include <fstream>
#include <sys/stat.h>

namespace std {}
using namespace std;

//#include <fstream>
//#include <strstream>

#include "TClassEdit.h"
using namespace TClassEdit;
#include "TMetaUtils.h"
using namespace ROOT;

#include "RClStl.h"
#include "RConversionRuleParser.h"
#include "XMLReader.h"
#include "LinkdefReader.h"
#include "SelectionRules.h"
#include "Scanner.h"

enum {
   TClassTable__kHasCustomStreamerMember = 0x10 // See TClassTable.h
};

cling::Interpreter *gInterp = 0;

// NOTE: This belongs in RConversionRules.cxx but can only be moved there if it is not shared with rootcint
   void R__GetQualifiedName(std::string &qual_name, const clang::QualType &type, const clang::NamedDecl &forcontext);

   //--------------------------------------------------------------------------
   void CreateNameTypeMap(const clang::CXXRecordDecl &cl, MembersTypeMap_t& nameType )
   {
      // Create the data member name-type map for given class

      std::stringstream dims;
      std::string typenameStr;

      // Loop over the non static data member.
      for(clang::RecordDecl::field_iterator field_iter = cl.field_begin(), end = cl.field_end();
          field_iter != end;
          ++field_iter)
      {
         // The CINT based code was filtering away static variables (they are not part of
         // the list starting with field_begin in clang), and const enums (which should
         // also not be part of this list).
         // It was also filtering out the 'G__virtualinfo' artificial member.

         typenameStr.clear();
         dims.clear();
         clang::QualType fieldType(field_iter->getType());
         if (fieldType->isConstantArrayType()) {
            const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(fieldType.getTypePtr());
            while (arrayType) {
               dims << "[" << arrayType->getSize().getLimitedValue() << "]";
               fieldType = arrayType->getElementType();
               arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
            }
         }
         R__GetQualifiedName(typenameStr, fieldType, *(*field_iter));

         nameType[field_iter->getName().str()] = TSchemaType(typenameStr.c_str(),dims.str().c_str());
      }

      // And now the base classes
      // We also need to look at the base classes.
      for(clang::CXXRecordDecl::base_class_const_iterator iter = cl.bases_begin(), end = cl.bases_end();
          iter != end;
          ++iter)
      {
         std::string basename( iter->getType()->getAsCXXRecordDecl()->getNameAsString() ); // Intentionally using only the unqualified name.
         nameType[basename] = TSchemaType(basename.c_str(),"");
      }         
   }

//______________________________________________________________________________
static void R__GetCurrentDirectory(std::string &output)
{
   char fixedLength[1024];
   char *currWorkDir = fixedLength;
   size_t len = 1024;
   char *result = currWorkDir;

   do {
      if (result == 0) {
         len = 2*len;
         if (fixedLength != currWorkDir) {
            delete [] currWorkDir;
         }
         currWorkDir = new char[len];
      }  
#ifdef WIN32
      result = ::_getcwd(currWorkDir, len);
#else
      result = getcwd(currWorkDir, len);
#endif
   } while ( result == 0 && errno == ERANGE );

   output = currWorkDir;
   output += '/';

   if (fixedLength != currWorkDir) {
      delete [] currWorkDir;
   }
}

//______________________________________________________________________________
static std::string R__GetRelocatableHeaderName(const char *header, const std::string &currentDirectory) 
{
   // Convert to path relative to $PWD.
   // If that's not what the caller wants, she should pass -I to rootcint and a
   // different relative path to the header files.
#ifdef ROOTBUILD
         // For ROOT, convert module directories like core/base/inc/ to include/
#endif

   std::string result( header );

   const char *currWorkDir = currentDirectory.c_str();
   size_t lenCurrWorkDir = strlen(currWorkDir);
   if (result.substr(0, lenCurrWorkDir) == currWorkDir) {
      // Convert to path relative to $PWD.
      // If that's not what the caller wants, she should pass -I to rootcint and a
      // different relative path to the header files.
      result.erase(0, lenCurrWorkDir);
   }
#ifdef ROOTBUILD
   // For ROOT, convert module directories like core/base/inc/ to include/
   int posInc = result.find("/inc/");
   if (posInc != -1) {
      result = /*std::string("include") +*/ result.substr(posInc + 5, -1);
   }
#endif
   return result;
}
using namespace ROOT;

std::ostream* dictSrcOut=&std::cout;
std::ostream* dictHdrOut=&std::cout;

bool gNeedCollectionProxy = false;

char *StrDup(const char *str);

class RConstructorType {
   std::string           fArgTypeName;
   const clang::CXXRecordDecl *fArgType;

public:
   RConstructorType(const char *type_of_arg) : fArgTypeName(type_of_arg),fArgType(0) 
   {
      const cling::LookupHelper& lh = gInterp->getLookupHelper();
      // We can not use findScope since the type we are given are usually,
      // only forward declared (and findScope explicitly reject them).
      clang::QualType instanceType = lh.findType(type_of_arg); 
      if (!instanceType.isNull()) 
         fArgType = instanceType->getAsCXXRecordDecl();
   }

   const char *GetName() { return fArgTypeName.c_str(); }
   const clang::CXXRecordDecl *GetType() { return fArgType; }
};

vector<RConstructorType> gIoConstructorTypes;
void AddConstructorType(const char *arg)
{
   if (arg) gIoConstructorTypes.push_back(RConstructorType(arg));
}

const clang::DeclContext *R__GetEnclosingSpace(const clang::RecordDecl &cl)
{
   const clang::DeclContext *ctxt = cl.getDeclContext();
   while(ctxt && !ctxt->isNamespace()) {
      ctxt = ctxt->getParent();
   }
   return ctxt;
}

bool ClassInfo__HasMethod(const clang::RecordDecl *cl, const char* name) 
{
   const clang::CXXRecordDecl* CRD = llvm::dyn_cast<clang::CXXRecordDecl>(cl);
   if (!CRD) {
      return false;
   }
   std::string given_name(name);
   for (
        clang::CXXRecordDecl::method_iterator M = CRD->method_begin(),
        MEnd = CRD->method_end();
        M != MEnd;
        ++M
        ) 
   {
      if (M->getNameAsString() == given_name) {
         return true;
      }
   }
   return false;
}

bool Namespace__HasMethod(const clang::NamespaceDecl *cl, const char* name) 
{
   std::string given_name(name);
   for (
        clang::DeclContext::decl_iterator M = cl->decls_begin(),
        MEnd = cl->decls_begin();
        M != MEnd;
        ++M
        ) {
      if (M->isFunctionOrFunctionTemplate()) {
         clang::NamedDecl *named = llvm::dyn_cast<clang::NamedDecl>(*M);
         if (named && named->getNameAsString() == given_name) {
            return true;
         }
      }
   }
   return false;
}

llvm::StringRef R__GetFileName(const clang::Decl *decl)
{
   // It looks like the template specialization decl actually contains _less_ information
   // on the location of the code than the decl (in case where there is forward declaration,
   // that is what the specialization points to.
   //
   // const clang::CXXRecordDecl* clxx = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
   // if (clxx) {
   //    switch(clxx->getTemplateSpecializationKind()) {
   //       case clang::TSK_Undeclared:
   //          // We want the default behavior
   //          break;
   //       case clang::TSK_ExplicitInstantiationDeclaration:
   //       case clang::TSK_ExplicitInstantiationDefinition:
   //       case clang::TSK_ImplicitInstantiation: {
   //          // We want the location of the template declaration:
   //          const clang::ClassTemplateSpecializationDecl *tmplt_specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl> (clxx);
   //          if (tmplt_specialization) {
   //             // return R__GetFileName(const_cast< clang::ClassTemplateSpecializationDecl *>(tmplt_specialization)->getSpecializedTemplate());
   //          }
   //          break;
   //       }
   //       case clang::TSK_ExplicitSpecialization:
   //          // We want the default behavior
   //          break;
   //       default:
   //          break;
   //    } 
   // }   
   clang::SourceLocation sourceLocation = decl->getLocation();
   clang::SourceManager& sourceManager = decl->getASTContext().getSourceManager();

   if (sourceLocation.isValid() && sourceLocation.isFileID()) {
      clang::PresumedLoc PLoc = sourceManager.getPresumedLoc(sourceLocation);
      return PLoc.getFilename();
   }
   else {
      return "invalid";
   }
}

long R__GetLineNumber(const clang::Decl *decl)
{
   // It looks like the template specialization decl actually contains _less_ information
   // on the location of the code than the decl (in case where there is forward declaration,
   // that is what the specialization points to.
   //
   // const clang::CXXRecordDecl* clxx = llvm::dyn_cast<clang::CXXRecordDecl>(decl);
   // if (clxx) {
   //    switch(clxx->getTemplateSpecializationKind()) {
   //       case clang::TSK_Undeclared:
   //          // We want the default behavior
   //          break;
   //       case clang::TSK_ExplicitInstantiationDeclaration:
   //       case clang::TSK_ExplicitInstantiationDefinition:
   //       case clang::TSK_ImplicitInstantiation: {
   //          // We want the location of the template declaration:
   //          const clang::ClassTemplateSpecializationDecl *tmplt_specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl> (clxx);
   //          if (tmplt_specialization) {
   //             return R__GetLineNumber(const_cast< clang::ClassTemplateSpecializationDecl *>(tmplt_specialization)->getSpecializedTemplate());
   //          }
   //          break;
   //       }
   //       case clang::TSK_ExplicitSpecialization:
   //          // We want the default behavior
   //          break;
   //       default:
   //          break;
   //    } 
   // }      
   clang::SourceLocation sourceLocation = decl->getLocation();
   clang::SourceManager& sourceManager = decl->getASTContext().getSourceManager();

   if (sourceLocation.isValid() && sourceLocation.isFileID()) {
      return sourceManager.getLineNumber(sourceManager.getFileID(sourceLocation),sourceManager.getFileOffset(sourceLocation));
   }
   else {
      return -1;
   }   
}

// In order to store the meaningful for the IO comments we have to transform 
// the comment into annotation of the given decl.
void R__AnnotateDecl(clang::CXXRecordDecl &CXXRD) 
{
   using namespace clang;
   SourceLocation commentSLoc;
   llvm::StringRef comment;

   ASTContext &C = CXXRD.getASTContext();
   Sema& S = gInterp->getCI()->getSema();

   SourceRange commentRange;

   for(CXXRecordDecl::decl_iterator I = CXXRD.decls_begin(), 
          E = CXXRD.decls_end(); I != E; ++I) {
      if (!(*I)->isImplicit() 
          && (isa<CXXMethodDecl>(*I) || isa<FieldDecl>(*I) || isa<VarDecl>(*I))) {
         // For now we allow only a special macro (ClassDef) to have meaningful comments
         SourceLocation maybeMacroLoc = (*I)->getLocation();
         bool isClassDefMacro = maybeMacroLoc.isMacroID() && S.findMacroSpelling(maybeMacroLoc, "ClassDef");
         if (isClassDefMacro) {
            while (isa<NamedDecl>(*I) && cast<NamedDecl>(*I)->getName() != "DeclFileLine")
               ++I;
         }

         comment = ROOT::TMetaUtils::GetComment(**I, &commentSLoc);
         if (comment.size()) {
            // Keep info for the source range of the comment in case we want to issue
            // nice warnings, eg. empty comment and so on.
            commentRange = SourceRange(commentSLoc, commentSLoc.getLocWithOffset(comment.size()));
            // The ClassDef annotation is for the class itself
            if (isClassDefMacro)
               CXXRD.addAttr(new (C) AnnotateAttr(commentRange, C, comment.str()));
            else
               (*I)->addAttr(new (C) AnnotateAttr(commentRange, C, comment.str()));
         }
      }
   }
}

std::string gResourceDir;

void R__GetQualifiedName(std::string &qual_name, const clang::QualType &type, const clang::NamedDecl &forcontext)
{
   clang::PrintingPolicy policy( forcontext.getASTContext().getPrintingPolicy() );
   policy.SuppressTagKeyword = true; // Never get the class or struct keyword
   policy.SuppressUnwrittenScope = true; // Don't write the inline or anonymous namespace names.
   type.getAsStringInternal(qual_name,policy);
}

void R__GetQualifiedName(std::string &qual_name, const clang::NamedDecl &cl)
{
   clang::PrintingPolicy policy( cl.getASTContext().getPrintingPolicy() );
   policy.SuppressTagKeyword = true; // Never get the class or struct keyword
   policy.SuppressUnwrittenScope = true; // Don't write the inline or anonymous namespace names.
  
   cl.getNameForDiagnostic(qual_name,policy,true);

   if ( strncmp(qual_name.c_str(),"<anonymous ",strlen("<anonymous ") ) == 0) {
      size_t pos = qual_name.find(':');
      qual_name.erase(0,pos+2);
   }
}

void R__GetQualifiedName(std::string &qual_name, const RScanner::AnnotatedRecordDecl &annotated)
{
   R__GetQualifiedName(qual_name,*annotated.GetRecordDecl());
}

std::string R__GetQualifiedName(const clang::QualType &type, const clang::NamedDecl &forcontext)
{
   std::string result;
   R__GetQualifiedName(result,type,forcontext);
   return result;
}

std::string R__GetQualifiedName(const clang::Type &type, const clang::NamedDecl &forcontext)
{
   std::string result;
   R__GetQualifiedName(result,clang::QualType(&type,0),forcontext);
   return result;
}

std::string R__GetQualifiedName(const clang::NamedDecl &cl)
{
   clang::PrintingPolicy policy( cl.getASTContext().getPrintingPolicy() );
   policy.SuppressTagKeyword = true; // Never get the class or struct keyword
   policy.SuppressUnwrittenScope = true; // Don't write the inline or anonymous namespace names.

   std::string result;
   cl.getNameForDiagnostic(result,policy,true); // qual_name = N->getQualifiedNameAsString();
   return result;
}

std::string R__GetQualifiedName(const clang::CXXBaseSpecifier &base)
{
   std::string result;
   R__GetQualifiedName(result,*base.getType()->getAsCXXRecordDecl());
   return result;
}

std::string R__GetQualifiedName(const RScanner::AnnotatedRecordDecl &annotated)
{
   return R__GetQualifiedName(*annotated.GetRecordDecl());
}

bool R__GetNameWithinNamespace(std::string &fullname, 
                               std::string &clsname, std::string &nsname, 
                               const clang::CXXRecordDecl *cl)
{
   // Return true one of the class' enclosing scope is a namespace and
   // set fullname to the fully qualified name,
   // clsname to the name within a namespace
   // and nsname to the namespace fully qualified name.

   fullname.clear();
   nsname.clear();

   R__GetQualifiedName(fullname,*cl);
   clsname = fullname;
   
   const clang::NamedDecl *ctxt = llvm::dyn_cast<clang::NamedDecl>(cl->getEnclosingNamespaceContext());
   if (ctxt && ctxt!=cl) {
      const clang::NamespaceDecl *nsdecl = llvm::dyn_cast<clang::NamespaceDecl>(ctxt);
      if (nsdecl == 0 || !nsdecl->isAnonymousNamespace()) {
         R__GetQualifiedName(nsname,*ctxt);
         clsname.erase (0, nsname.size() + 2);
         return true;
      }
   }
   return false;
}

std::string R__TrueName(const clang::FieldDecl &m)
{
   // TrueName strips the typedefs and array dimensions.
   
   const clang::Type *rawtype = m.getType()->getCanonicalTypeInternal().getTypePtr();   
   if (rawtype->isArrayType()) {
      rawtype = rawtype->getBaseElementTypeUnsafe ();
   }
   
   std::string result;
   R__GetQualifiedName(result, clang::QualType(rawtype,0), m);
   return result;
}

const clang::CXXRecordDecl *R__ScopeSearch(const char *name, const clang::Type** resultType = 0) 
{
   // Return the scope corresponding to 'name' or std::'name'
   const cling::LookupHelper& lh = gInterp->getLookupHelper();
   const clang::CXXRecordDecl *result = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(lh.findScope(name,resultType));
   if (!result) {
      std::string std_name("std::");
      std_name += name;
      result = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(lh.findScope(std_name,resultType));
   }
   return result;
}

const clang::Type *R__GetUnderlyingType(clang::QualType type)
{
   // Return the base/underlying type of a chain of array or pointers type.
   // Does not yet support the array and pointer part being intermixed.
   
   const clang::Type *rawtype = type.getTypePtr();

   // NOTE: We probably meant isa<clang::ElaboratedType>
   if (rawtype->isElaboratedTypeSpecifier() ) {
      rawtype = rawtype->getCanonicalTypeInternal().getTypePtr();
   }
   if (rawtype->isArrayType()) {
      rawtype = type.getTypePtr()->getBaseElementTypeUnsafe ();
   }   
   if (rawtype->isPointerType() || rawtype->isReferenceType() ) {
      //Get to the 'raw' type.
      clang::QualType pointee;
      while ( (pointee = rawtype->getPointeeType()) , pointee.getTypePtrOrNull() && pointee.getTypePtr() != rawtype)
      {
         rawtype = pointee.getTypePtr();
      }
   }
   if (rawtype->isArrayType()) {
      rawtype = type.getTypePtr()->getBaseElementTypeUnsafe ();
   }
   return rawtype;
}


clang::RecordDecl *R__GetUnderlyingRecordDecl(clang::QualType type)
{
   const clang::Type *rawtype = R__GetUnderlyingType(type);

   if (rawtype->isFundamentalType() || rawtype->isEnumeralType()) {
      // not an ojbect.
      return 0;
   }
   return rawtype->getAsCXXRecordDecl();
}

size_t R__GetFullArrayLength(const clang::ConstantArrayType *arrayType)
{
   llvm::APInt len = arrayType->getSize();
   while(const clang::ConstantArrayType *subArrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual()) )
   {
      len *= subArrayType->getSize();
      arrayType = subArrayType;
   }
   return len.getLimitedValue();
}


class FDVisitor : public clang::RecursiveASTVisitor<FDVisitor> {
private:
   clang::FunctionDecl* fFD;
public:
   clang::FunctionDecl* getFD() const { return fFD; }
   bool VisitDeclRefExpr(clang::DeclRefExpr* DRE) {
      fFD = llvm::dyn_cast<clang::FunctionDecl>(DRE->getDecl());
      return true;
   }
   FDVisitor() : fFD(0) {}
};

//______________________________________________________________________________
const clang::FunctionDecl *R__GetFuncWithProto(const clang::Decl* cinfo, 
                                               const char *method, const char *proto)
{
   return gInterp->getLookupHelper().findFunctionProto(cinfo, method, proto);
}

//______________________________________________________________________________
const clang::CXXMethodDecl *R__GetMethodWithProto(const clang::Decl* cinfo, 
                                                  const char *method, const char *proto)
{
   const clang::FunctionDecl* funcD 
      = gInterp->getLookupHelper().findFunctionProto(cinfo, method, proto);
   if (funcD) {
      return llvm::dyn_cast<const clang::CXXMethodDecl>(funcD);
   }
   return 0;
}

bool R__CheckPublicFuncWithProto(const clang::CXXRecordDecl *cl, const char *methodname, const char *proto)
{
   // Return true, if the function (defined by the name and prototype) exists and is public

   const clang::CXXMethodDecl *method = R__GetMethodWithProto(cl,methodname,proto);
   if (method && method->getAccess() == clang::AS_public) {
      return true;
   }
   return false;
}

bool ClassInfo__IsBase(const clang::RecordDecl *cl, const char* basename)
{
   const clang::CXXRecordDecl* CRD = llvm::dyn_cast<clang::CXXRecordDecl>(cl);
   if (!CRD) {
      return false;
   }
   const clang::NamedDecl *base = R__ScopeSearch(basename);
   if (base) {
      const clang::CXXRecordDecl* baseCRD = llvm::dyn_cast<clang::CXXRecordDecl>( base ); 
      if (baseCRD) return CRD->isDerivedFrom(baseCRD);
   }
   return false;
}

bool R__IsBase(const clang::CXXRecordDecl *cl, const clang::CXXRecordDecl *base)
{
   if (!cl || !base) {
      return false;
   }
   return cl->isDerivedFrom(base);
}

bool R__IsBase(const clang::FieldDecl &m, const char* basename)
{
   const clang::CXXRecordDecl* CRD = llvm::dyn_cast<clang::CXXRecordDecl>(R__GetUnderlyingRecordDecl(m.getType()));
   if (!CRD) {
      return false;
   }
   
   const clang::NamedDecl *base = R__ScopeSearch(basename);
   
   if (base) {
      const clang::CXXRecordDecl* baseCRD = llvm::dyn_cast<clang::CXXRecordDecl>( base ); 
      if (baseCRD) return CRD->isDerivedFrom(baseCRD);
   }
   return false;
}


bool InheritsFromTObject(const clang::RecordDecl *cl)
{
   static const clang::CXXRecordDecl *TObject_decl = R__ScopeSearch("TObject");

   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl);
   return R__IsBase(clxx, TObject_decl);
}

bool InheritsFromTSelector(const clang::RecordDecl *cl)
{
   static const clang::CXXRecordDecl *TObject_decl = R__ScopeSearch("TSelector");

   return R__IsBase(llvm::dyn_cast<clang::CXXRecordDecl>(cl), TObject_decl);
}

bool IsStdClass(const clang::RecordDecl &cl)
{
   // Return true, if the decl is part of the std namespace.

   const clang::DeclContext *ctx = cl.getDeclContext();

   if (ctx->isNamespace())
   {
      const clang::NamedDecl *parent = llvm::dyn_cast<clang::NamedDecl> (ctx);
      if (parent) {
         if (parent->getQualifiedNameAsString()=="std") {
            return true;
         }
      }
   }
   return false;
}

void R__GetName(std::string &qual_name, const clang::NamedDecl *cl)
{
   cl->getNameForDiagnostic(qual_name,cl->getASTContext().getPrintingPolicy(),false); // qual_name = N->getQualifiedNameAsString();
}

inline bool R__IsTemplate(const clang::Decl &cl)
{
   return (cl.getKind() == clang::Decl::ClassTemplatePartialSpecialization
           || cl.getKind() == clang::Decl::ClassTemplateSpecialization);
}

//inline bool R__IsTemplate(const clang::CXXRecordDecl *cl)
//{
//   return cl->getTemplateSpecializationKind() != clang::TSK_Undeclared;
//}

bool R__IsSelectionXml(const char *filename)
{
   size_t len = strlen(filename);
   size_t xmllen = 4; /* strlen(".xml"); */
   if (strlen(filename) >= xmllen ) {
      return (0 == strcasecmp( filename + (len - xmllen), ".xml"));
   } else {
      return false;
   }
}

bool R__IsLinkdefFile(const char *filename)
{
   if ((strstr(filename,"LinkDef") || strstr(filename,"Linkdef") ||
        strstr(filename,"linkdef")) && strstr(filename,".h")) {
      return true;
   }
   size_t len = strlen(filename);
   size_t linkdeflen = 9; /* strlen("linkdef.h") */
   if (len >= 9) {
      if (0 == strncasecmp( filename + (len - linkdeflen), "linkdef", linkdeflen-2)
          && 0 == strcmp(filename + (len - 2),".h")
          ) {
         return true;
      } else {
         return false;
      }
   } else {
      return false;
   }
}

bool R__IsSelectionFile(const char *filename)
{
   return R__IsLinkdefFile(filename) || R__IsSelectionXml(filename);
}

//const char* root_style()  {
//  static const char* s = ::getenv("MY_ROOT");
//  return s;
//}

//______________________________________________________________________________

#ifndef ROOT_Varargs
#include "Varargs.h"
#endif

const int kInfo     =      0;
const int kNote     =    500;
const int kWarning  =   1000;
const int kError    =   2000;
const int kSysError =   3000;
const int kFatal    =   4000;
const int kMaxLen   =   1024;

static int gErrorIgnoreLevel = kError;

//______________________________________________________________________________
void LevelPrint(bool prefix, int level, const char *location,
                const char *fmt, va_list ap)
{
   if (level < gErrorIgnoreLevel)
      return;

   const char *type = 0;

   if (level >= kInfo)
      type = "Info";
   if (level >= kNote)
      type = "Note";
   if (level >= kWarning)
      type = "Warning";
   if (level >= kError)
      type = "Error";
   if (level >= kSysError)
      type = "SysError";
   if (level >= kFatal)
      type = "Fatal";

   if (!location || strlen(location) == 0) {
      if (prefix) fprintf(stderr, "%s: ", type);
      vfprintf(stderr, (char*)va_(fmt), ap);
   } else {
      if (prefix) fprintf(stderr, "%s in <%s>: ", type, location);
      else fprintf(stderr, "In <%s>: ", location);
      vfprintf(stderr, (char*)va_(fmt), ap);
   }

   fflush(stderr);
}

//______________________________________________________________________________
void Error(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case an error occured.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void SysError(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case a system (OS or GUI) related error occured.

   va_list ap;
   va_start(ap, va_(fmt));
   LevelPrint(true, kSysError, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Info(const char *location, const char *va_(fmt), ...)
{
   // Use this function for informational messages.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kInfo, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Warning(const char *location, const char *va_(fmt), ...)
{
   // Use this function in warning situations.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kWarning, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void Fatal(const char *location, const char *va_(fmt), ...)
{
   // Use this function in case of a fatal error. It will abort the program.

   va_list ap;
   va_start(ap,va_(fmt));
   LevelPrint(true, kFatal, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
const char *GetExePath()
{
   // Returns the executable path name, used by SetRootSys().

   static std::string exepath;
   if (exepath == "") {
#ifdef __APPLE__
      exepath = _dyld_get_image_name(0);
#endif
#ifdef __linux
      char linkname[PATH_MAX];  // /proc/<pid>/exe
      char buf[PATH_MAX];     // exe path name
      pid_t pid;

      // get our pid and build the name of the link in /proc
      pid = getpid();
      snprintf(linkname,PATH_MAX, "/proc/%i/exe", pid);
      int ret = readlink(linkname, buf, 1024);
      if (ret > 0 && ret < 1024) {
         buf[ret] = 0;
         exepath = buf;
      }
#endif
#ifdef _WIN32
   char *buf = new char[MAX_MODULE_NAME32 + 1];
   ::GetModuleFileName(NULL, buf, MAX_MODULE_NAME32 + 1);
   char* p = buf;
   while ((p = strchr(p, '\\')))
      *(p++) = '/';
   exepath = buf;
   delete[] buf;
#endif
   }
   return exepath.c_str();
}

//______________________________________________________________________________
void SetRootSys()
{
   // Set the ROOTSYS env var based on the executable location.

   const char *exepath = GetExePath();
   if (exepath && *exepath) {
#if !defined(_WIN32)
      char *ep = new char[PATH_MAX];
      if (!realpath(exepath, ep)) {
         if (getenv("ROOTSYS")) {
            delete [] ep;
            return;
         } else {
            fprintf(stderr, "rootcint: error getting realpath of rootcint, please set ROOTSYS in the shell");
            strlcpy(ep, exepath,PATH_MAX);
         }
      }
#else
      int nche = strlen(exepath)+1;
      char *ep = new char[nche];
      strlcpy(ep, exepath,nche);
#endif
      char *s;
      if ((s = strrchr(ep, '/'))) {
         // $ROOTSYS/bin/rootcint
         int removesubdirs = 2;
         if (!strncmp(s, "rootcint_tmp", 12))
            // $ROOTSYS/core/utils/src/rootcint_tmp
            removesubdirs = 4;
         for (int i = 1; s && i < removesubdirs; ++i) {
            *s = 0;
            s = strrchr(ep, '/');
         }
         if (s) *s = 0;
      } else {
         // There was no slashes at all let now change ROOTSYS
         return;
      }
      int ncha = strlen(ep) + 10;
      char *env = new char[ncha];
      snprintf(env, ncha, "ROOTSYS=%s", ep);
      putenv(env);
      delete [] ep;
   }
}

//______________________________________________________________________________
bool ParsePragmaLine(const std::string& line, const char* expectedTokens[],
                     size_t* end = 0) {
   // Check whether the #pragma line contains expectedTokens (0-terminated array).
   if (end) *end = 0;
   if (line[0] != '#') return false;
   size_t pos = 1;
   for (const char** iToken = expectedTokens; *iToken; ++iToken) {
      while (isspace(line[pos])) ++pos;
      size_t lenToken = strlen(*iToken);
      if (line.compare(pos, lenToken, *iToken)) {
         if (end) *end = pos;
         return false;
      }
      pos += lenToken;
   }
   if (end) *end = pos;
   return true;
}

#ifdef _WIN32
//______________________________________________________________________________
// defined in newlink.c
extern "C" FILE *FOpenAndSleep(const char *filename, const char *mode);

# ifdef fopen
#  undef fopen
# endif
# define fopen(A,B) FOpenAndSleep((A),(B))
#endif


//______________________________________________________________________________
typedef map<string,string> Recmap_t;
Recmap_t gAutoloads;
string gLiblistPrefix;
string gLibsNeeded;

void RecordDeclCallback(const char *c)
{
   string need( gAutoloads[c] );
   if (need.length() && gLibsNeeded.find(need)==string::npos) {
      gLibsNeeded += " " + need;
   }
}

void LoadLibraryMap()
{
   string filelistname = gLiblistPrefix + ".in";
   ifstream filelist(filelistname.c_str());

   string filename;
   static char *buffer = 0;
   static unsigned int sbuffer = 0;

   while ( filelist >> filename ) {
#ifdef WIN32
      struct _stati64 finfo;

      if (_stati64(filename.c_str(), &finfo) < 0 ||
          finfo.st_mode & S_IFDIR) {
         continue;
      }
#else
      struct stat finfo;
      if (stat(filename.c_str(), &finfo) < 0 ||
          S_ISDIR(finfo.st_mode)) {
         continue;
      }

#endif

      ifstream file(filename.c_str());

      string line;
      string classname;

      while ( file >> line ) {

         if (line.substr(0,8)=="Library.") {

            int pos = line.find(":",8);
            classname = line.substr(8,pos-8);

            pos = 0;
            while ( (pos=classname.find("@@",pos)) >= 0 ) {
               classname.replace(pos,2,"::");
            }
            pos = 0;
            while ( (pos=classname.find("-",pos)) >= 0 ) {
               classname.replace(pos,1," ");
            }

            getline(file,line,'\n');

            while( line[0]==' ' ) line.replace(0,1,"");

            if ( strchr(classname.c_str(),':')!=0 ) {
               // We have a namespace and we have to check it first

               int slen = classname.size();
               for(int k=0;k<slen;++k) {
                  if (classname[k]==':') {
                     if (k+1>=slen || classname[k+1]!=':') {
                        // we expected another ':'
                        break;
                     }
                     if (k) {
                        string base = classname.substr(0,k);
                        if (base=="std") {
                           // std is not declared but is also ignored by CINT!
                           break;
                        } else {
                           gAutoloads[base] = ""; // We never load namespaces on their own.
                        }
                        ++k;
                     }
                  } else if (classname[k] == '<') {
                     // We do not want to look at the namespace inside the template parameters!
                     break;
                  }
               }
            }

            if (strncmp("ROOT::TImpProxy",classname.c_str(),strlen("ROOT::TImpProxy"))==0) {
               // Do not register the ROOT::TImpProxy so that they can be instantiated.
               continue;
            }
            gAutoloads[classname] = line;
            if (sbuffer < classname.size()+20) {
               delete [] buffer;
               sbuffer = classname.size()+20;
               buffer = new char[sbuffer];
            }
            strlcpy(buffer,classname.c_str(),sbuffer);
         }
      }
      file.close();
   }
}

//______________________________________________________________________________
bool CheckInputOperator(const char *what, const char *proto, const string &fullname, const clang::RecordDecl *cl)
{
   // Check if the specificed operator (what) has been properly declared if the user has
   // resquested a custom version.

 
   const clang::FunctionDecl *method = R__GetFuncWithProto(llvm::dyn_cast<clang::Decl>(cl->getDeclContext()), what, proto);
   if (!method) {
      // This intended to find the global scope.
      clang::TranslationUnitDecl *TU =
         cl->getASTContext().getTranslationUnitDecl();
      method = R__GetFuncWithProto(TU, what, proto);
   }
   bool has_input_error = false;
   if (method != 0 && (method->getAccess() == clang::AS_public || method->getAccess() == clang::AS_none) ) {
      std::string filename = R__GetFileName(method);
      if (strstr(filename.c_str(),"TBuffer.h")!=0 ||
          strstr(filename.c_str(),"Rtypes.h" )!=0) {

         has_input_error = true;
      }
   } else {
      has_input_error = true;
   }
   if (has_input_error) {
      // We don't want to generate duplicated error messages in several dictionaries (when generating temporaries)
      const char *maybeconst = "";
      const char *mayberef = "&";
      if (what[strlen(what)-1]=='<') {
         maybeconst = "const ";
         mayberef = "";
      }
      Error(0,
            "in this version of ROOT, the option '!' used in a linkdef file\n"
            "       implies the actual existence of customized operators.\n"
            "       The following declaration is now required:\n"
            "   TBuffer &%s(TBuffer &,%s%s *%s);\n",what,maybeconst,fullname.c_str(),mayberef);
   }
   return has_input_error;
 
}

//______________________________________________________________________________
bool CheckInputOperator(const clang::RecordDecl *cl)
{
   // Check if the operator>> has been properly declared if the user has
   // resquested a custom version.

   string fullname;
   R__GetQualifiedName(fullname,*cl);
   int ncha = fullname.length()+13;
   char *proto = new char[ncha];
   snprintf(proto,ncha,"TBuffer&,%s*&",fullname.c_str());
   
   Info(0, "Class %s: Do not generate operator>>()\n",
        fullname.c_str());

   // We do want to call both CheckInputOperator all the times.
   bool has_input_error = CheckInputOperator("operator>>",proto,fullname,cl);
   has_input_error = CheckInputOperator("operator<<",proto,fullname,cl) || has_input_error;
   return has_input_error;
}

//______________________________________________________________________________
bool CheckClassDef(const clang::RecordDecl *cl)
{
   // Return false if the class does not have ClassDef even-though it should.


   // Detect if the class has a ClassDef
   bool hasClassDef = ClassInfo__HasMethod(cl,"Class_Version");

   const clang::CXXRecordDecl* clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl);
   if (!clxx) {
      return false;
   }
   bool isAbstract = clxx->isAbstract();

   bool result = true;
   if (!isAbstract && InheritsFromTObject(clxx) && !InheritsFromTSelector(clxx)
       && !hasClassDef) {
      Error(R__GetQualifiedName(*cl).c_str(),"CLING: %s inherits from TObject but does not have its own ClassDef\n",R__GetQualifiedName(*cl).c_str());
      // We do want to always output the message (hence the Error level)
      // but still want rootcint to succeed.
      result = true;
   }

   return result;
}

//______________________________________________________________________________
bool HasDirectoryAutoAdd(const clang::CXXRecordDecl *cl)
{
   // Return true if the class has a method DirectoryAutoAdd(TDirectory *)

   // Detect if the class has a DirectoryAutoAdd

   // Detect if the class or one of its parent has a DirectoryAutoAdd
   const char *proto = "TDirectory*";
   const char *name = "DirectoryAutoAdd";

   return R__CheckPublicFuncWithProto(cl,name,proto);
}


//______________________________________________________________________________
bool HasNewMerge(const clang::CXXRecordDecl *cl)
{
   // Return true if the class has a method Merge(TCollection*,TFileMergeInfo*)

   // Detect if the class has a 'new' Merge function.

   // Detect if the class or one of its parent has a DirectoryAutoAdd
   const char *proto = "TCollection*,TFileMergeInfo*";
   const char *name = "Merge";

   return R__CheckPublicFuncWithProto(cl,name,proto);
}

//______________________________________________________________________________
bool HasOldMerge(const clang::CXXRecordDecl *cl)
{
   // Return true if the class has a method Merge(TCollection*)

   // Detect if the class has an old fashion Merge function.

   // Detect if the class or one of its parent has a DirectoryAutoAdd
   const char *proto = "TCollection*";
   const char *name = "Merge";

   return R__CheckPublicFuncWithProto(cl,name,proto);
}


//______________________________________________________________________________
bool HasResetAfterMerge(const clang::CXXRecordDecl *cl)
{
   // Return true if the class has a method ResetAfterMerge(TFileMergeInfo *)

   // Detect if the class has a 'new' Merge function.
   // bool hasMethod = cl.HasMethod("DirectoryAutoAdd");

   // Detect if the class or one of its parent has a DirectoryAutoAdd
   const char *proto = "TFileMergeInfo*";
   const char *name = "ResetAfterMerge";

   return R__CheckPublicFuncWithProto(cl,name,proto);
}

//______________________________________________________________________________
int GetClassVersion(const clang::RecordDecl *cl)
{
   // Return the version number of the class or -1
   // if the function Class_Version does not exist.

   if (!ClassInfo__HasMethod(cl,"Class_Version")) return -1;

   const clang::CXXRecordDecl* CRD = llvm::dyn_cast<clang::CXXRecordDecl>(cl);
   if (!CRD) {
      // Must be an enum or namespace.
      // FIXME: Make it work for a namespace!
      return false;
   }
   // Class_Version is know to be inline and we constrol (via the ClassDef macros)
   // it's structure, so this is apriori fine, but we could consider replacing it
   // with the slower but simpler:
   //   gInterp->evaluate( classname + "::Class_Version()", &Value);
   std::string given_name("Class_Version");
   for (
        clang::CXXRecordDecl::method_iterator M = CRD->method_begin(),
        MEnd = CRD->method_end();
        M != MEnd;
        ++M
        ) {
      if (M->getNameAsString() == given_name) {
         clang::CompoundStmt *func = 0;
         if (M->getBody()) {
            func = llvm::dyn_cast<clang::CompoundStmt>(M->getBody());
         } else {
            const clang::FunctionDecl *inst = M->getInstantiatedFromMemberFunction();
            if (inst && inst->getBody()) {
               func = llvm::dyn_cast<clang::CompoundStmt>(inst->getBody());
            } else {
               Error("GetClassVersion","Could not find the body for %s::ClassVersion!\n",R__GetQualifiedName(*cl).c_str());
            }
         }
         if (func && !func->body_empty()) {
            clang::ReturnStmt *ret = llvm::dyn_cast<clang::ReturnStmt>(*func->body_begin());
            if (ret) {
               clang::IntegerLiteral *val;
               clang::ImplicitCastExpr *cast = llvm::dyn_cast<clang::ImplicitCastExpr>( ret->getRetValue() );
               if (cast) {
                  val = llvm::dyn_cast<clang::IntegerLiteral>( cast->getSubExprAsWritten() );
               } else {
                  val = llvm::dyn_cast<clang::IntegerLiteral>( ret->getRetValue() );
               }
               if (val) {
                  return (int)val->getValue().getLimitedValue(~0);
               }
            }
         }
         return 0;
      }
   }
   return 0;   
}

//______________________________________________________________________________
string GetNonConstMemberName(const clang::FieldDecl &m, const string &prefix = "")
{
   // Return the name of the data member so that it can be used
   // by non-const operation (so it includes a const_cast if necessary).

   if (m.getType().isConstQualified()) {      
      string ret = "const_cast< ";
      string type_name;
      R__GetQualifiedName(type_name, m.getType(), m);
      ret += type_name;
      ret += " &>( ";
      ret += prefix;
      ret += m.getName().str();
      ret += " )";
      return ret;
   } else {
      return prefix+m.getName().str();
   }
}

//______________________________________________________________________________
bool NeedExternalShowMember(const RScanner::AnnotatedRecordDecl &cl_input)
{
   if (IsStdClass(*cl_input.GetRecordDecl())) {
      // getName() return the template name without argument!
      llvm::StringRef name = (*cl_input).getName();
      
      if (name == "pair") return true;
      if (name == "complex") return true;
      if (name == "auto_ptr") return true;
      if (STLKind(name.str().c_str())) return false;
      if (name == "string" || name == "basic_string") return false;
   }
   
   // This means templated classes hiding members won't have
   // a proper shadow class, and the user has no chance of
   // veto-ing a shadow, as we need it for ShowMembers :-/
   if (ClassInfo__HasMethod(cl_input,"ShowMembers"))
      return R__IsTemplate(*cl_input);

   // no streamer, no shadow
   if (cl_input.RequestNoStreamer()) return false;

   return (cl_input.RequestStreamerInfo());
}

//______________________________________________________________________________
bool NeedTemplateKeyword(const clang::CXXRecordDecl *cl)
{
   clang::TemplateSpecializationKind kind = cl->getTemplateSpecializationKind();
   if (kind == clang::TSK_Undeclared ) {
      // Note a template;
      return false;
   } else if (kind == clang::TSK_ExplicitSpecialization) {
      // This is a specialized templated class
      return false;
   } else {
      // This is an automatically or explicitly instantiated templated class.
      return true;
   }  
}

//______________________________________________________________________________
bool HasCustomOperatorNewPlacement(const char *which, const clang::RecordDecl &cl)
{
   // return true if we can find a custom operator new with placement

   const char *name = which;
   const char *proto = "size_t";
   const char *protoPlacement = "size_t,void*";

   // First search in the enclosing namespaces
   const clang::FunctionDecl *operatornew = R__GetFuncWithProto(llvm::dyn_cast<clang::Decl>(cl.getDeclContext()), name, proto);
   const clang::FunctionDecl *operatornewPlacement = R__GetFuncWithProto(llvm::dyn_cast<clang::Decl>(cl.getDeclContext()), name, protoPlacement);

   const clang::DeclContext *ctxtnew = 0;
   const clang::DeclContext *ctxtnewPlacement = 0;
   
   if (operatornew) {
      ctxtnew = operatornew->getParent();
   }
   if (operatornewPlacement) {
      ctxtnewPlacement = operatornewPlacement->getParent();
   }

   // Then in the class and base classes
   operatornew = R__GetFuncWithProto(&cl, name, proto);
   operatornewPlacement = R__GetFuncWithProto(&cl, name, protoPlacement);

   if (operatornew) {
      ctxtnew = operatornew->getParent();
   }
   if (operatornewPlacement) {
      ctxtnewPlacement = operatornewPlacement->getParent();
   }

   if (ctxtnewPlacement == 0) {
      return false;
   }
   if (ctxtnew == 0) {
      // Only a new with placement, no hiding
      return true;
   }
   // Both are non zero
   if (ctxtnew == ctxtnewPlacement) {
      // Same declaration ctxt, no hiding
      return true;
   }
   const clang::CXXRecordDecl* clnew = llvm::dyn_cast<clang::CXXRecordDecl>(ctxtnew);         
   const clang::CXXRecordDecl* clnewPlacement = llvm::dyn_cast<clang::CXXRecordDecl>(ctxtnewPlacement);
   if (clnew == 0 && clnewPlacement == 0) {
      // They are both in different namespaces, I am not sure of the rules.
      // we probably ought to find which one is closest ... for now bail
      // (because rootcint was also bailing on that).
      return true;
   }
   if (clnew != 0 && clnewPlacement == 0) {
      // operator new is class method hiding the outer scope operator new with placement.
      return false;
   }
   if (clnew == 0 && clnewPlacement != 0) {
      // operator new is a not class method and can not hide new with placement which is a method
      return true;
   }
   // Both are class methods   
   if (clnew->isDerivedFrom(clnewPlacement)) {
      // operator new is in a more derived part of the hierarchy, it is hiding operator new with placement.
      return false;
   }
   // operator new with placement is in a more derived part of the hierarchy, it can't be hidden by operator new. 
   return true;
}

//______________________________________________________________________________
bool HasCustomOperatorNewPlacement(const clang::RecordDecl &cl)
{
   // return true if we can find a custom operator new with placement

   return HasCustomOperatorNewPlacement("operator new",cl);
}

//______________________________________________________________________________
bool HasCustomOperatorNewArrayPlacement(const clang::RecordDecl &cl)
{
   // return true if we can find a custom operator new with placement

   return HasCustomOperatorNewPlacement("operator new[]",cl);
}

//______________________________________________________________________________
bool CheckConstructor(const clang::CXXRecordDecl *cl, RConstructorType &ioctortype)
{
   const char *arg = ioctortype.GetName();
   if ( (arg == 0 || arg[0] == '\0') && !cl->hasUserDeclaredConstructor() ) {
      return true;
   }

   if (ioctortype.GetType() ==0 && (arg == 0 || arg[0] == '\0')) {
      // We are looking for a constructor with zero non-default arguments.

      for(clang::CXXRecordDecl::ctor_iterator iter = cl->ctor_begin(), end = cl->ctor_end();
          iter != end;
          ++iter)
      {
         if (iter->getAccess() != clang::AS_public)
            continue;
         // We can reach this constructor.
            
         if (iter->getNumParams() == 0) {
            return true;
         }
         if ( (*iter->param_begin())->hasDefaultArg()) {
            return true;
         }
      } // For each constructor.
   }
   else {
      for(clang::CXXRecordDecl::ctor_iterator iter = cl->ctor_begin(), end = cl->ctor_end();
          iter != end;
          ++iter) 
      {
         if (iter->getAccess() != clang::AS_public)
            continue;

         // We can reach this constructor.
         if (iter->getNumParams() == 1) {
            clang::QualType argType( (*iter->param_begin())->getType() );
            argType = argType.getDesugaredType(cl->getASTContext());
            if (argType->isPointerType()) {
               argType = argType->getPointeeType();
               argType = argType.getDesugaredType(cl->getASTContext());
               
               const clang::CXXRecordDecl *argDecl = argType->getAsCXXRecordDecl();
               if (argDecl && ioctortype.GetType()) {
                  if (argDecl->getCanonicalDecl() == ioctortype.GetType()->getCanonicalDecl()) {
                     return true;
                  }
               } else {
                  std::string realArg = argType.getAsString();
                  std::string clarg("class ");
                  clarg += arg;
                  if (realArg == clarg) {
                     return true;
                     
                  }
               }
            }
         } // has one argument.
      } // for each constructor 

      // Look for a potential templated constructor.
      for(clang::CXXRecordDecl::decl_iterator iter = cl->decls_begin(), end = cl->decls_end();
          iter != end;
          ++iter) 
      {
         const clang::FunctionTemplateDecl *func = llvm::dyn_cast<clang::FunctionTemplateDecl>(*iter);
         if (func) {
            const clang::CXXConstructorDecl *ctor = llvm::dyn_cast<clang::CXXConstructorDecl>(func->getTemplatedDecl ());
            if (ctor && ctor->getNumParams() == 1) {
               clang::QualType argType( (*ctor->param_begin())->getType() );
               argType = argType.getDesugaredType(cl->getASTContext());

               // Check for either of:
               //   ClassName::ClassName( T *&);
               //   ClassName::ClassName( T *);
               //   ClassName::ClassName( T );
               // which all could be used for a call to new ClassName( (ioctor*)0)

               // // Strip one reference type
               if (argType->isReferenceType()) {
                  if (argType->getPointeeType()->isPointerType()) {
                     argType = argType->getPointeeType();
                     argType = argType.getDesugaredType(cl->getASTContext());
                  }
               }
               // Strip one pointer type
               if (argType->isPointerType()) {
                  argType = argType->getPointeeType();
                  argType = argType.getDesugaredType(cl->getASTContext());
               }
               if (argType->isTemplateTypeParmType()) {
                  return true;
               }
            }
         }
      }
   }

   return false;
}

//______________________________________________________________________________
bool HasDefaultConstructor(const clang::CXXRecordDecl *cl, string *arg)
{
   // return true if we can find an constructor calleable without any arguments

   bool result = false;

   if (cl->isAbstract()) return false;

   for(unsigned int i=0; i<gIoConstructorTypes.size(); ++i) {
      string proto( gIoConstructorTypes[i].GetName() );
      int extra = (proto.size()==0) ? 0 : 1;
      if (extra==0) {
         // Looking for default constructor
         result = true;
      } else {
         proto += " *";
      }

      result = CheckConstructor(cl,gIoConstructorTypes[i]);
      if (result && extra && arg) {
         *arg = "( (";
         *arg += proto;
         *arg += ")0 )";
      }

      // Check for private operator new
      if (result) {
         const char *name = "operator new";
         proto = "size_t";
         const clang::CXXMethodDecl *method = R__GetMethodWithProto(cl,name,proto.c_str());
         if (method && method->getAccess() != clang::AS_public) {
            result = false;
         }
         if (result) return true;
      }
   }
   return result;
}


//______________________________________________________________________________
bool NeedDestructor(const clang::CXXRecordDecl *cl)
{
   if (!cl) return false;

   if (cl->hasUserDeclaredDestructor()) {

      clang::CXXDestructorDecl *dest = cl->getDestructor();
      if (dest) {
         return (dest->getAccess() == clang::AS_public);
      } else {
         return true; // no destructor, so let's assume it means default?
      }
   }
   return true;
}

//______________________________________________________________________________
bool HasCustomStreamerMemberFunction(const RScanner::AnnotatedRecordDecl &cl)
{
   // Return true if the class has a custom member function streamer.

   static const char *proto = "TBuffer&";

   const clang::CXXRecordDecl* clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   const clang::CXXMethodDecl *method = R__GetMethodWithProto(clxx,"Streamer",proto);
   const clang::DeclContext *clxx_as_context = llvm::dyn_cast<clang::DeclContext>(clxx);

   return (method && method->getDeclContext() == clxx_as_context && ( cl.RequestNoStreamer() || !cl.RequestStreamerInfo()));
}


//______________________________________________________________________________
bool hasOpaqueTypedef(clang::QualType instanceType, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Return true if the type is a Double32_t or Float16_t or
   // is a instance template that depends on Double32_t or Float16_t.
  
   while (llvm::isa<clang::PointerType>(instanceType.getTypePtr())
       || llvm::isa<clang::ReferenceType>(instanceType.getTypePtr()))
   {
      instanceType = instanceType->getPointeeType();
   }
   
   const clang::ElaboratedType* etype 
      = llvm::dyn_cast<clang::ElaboratedType>(instanceType.getTypePtr());
   if (etype) {
      instanceType = clang::QualType(etype->getNamedType().getTypePtr(),0);
   }

   // There is no typedef to worried about, except for the opaque ones.
   
   // Technically we should probably used our own list with just
   // Double32_t and Float16_t
   if (normCtxt.GetTypeWithAlternative().count(instanceType.getTypePtr())) {
      return true;
   }

   
   bool result = false;
   const clang::CXXRecordDecl* clxx = instanceType->getAsCXXRecordDecl();
   if (clxx && clxx->getTemplateSpecializationKind() != clang::TSK_Undeclared) {
      // do the template thing.
      const clang::TemplateSpecializationType* TST 
      = llvm::dyn_cast<const clang::TemplateSpecializationType>(instanceType.getTypePtr());
      if (TST==0) {
//         std::string type_name;
//         type_name =  R__GetQualifiedName( instanceType, *clxx ); 
//         fprintf(stderr,"ERROR: Could not findS TST for %s\n",type_name.c_str());
         return false;
      }
      for(clang::TemplateSpecializationType::iterator 
          I = TST->begin(), E = TST->end();
          I!=E; ++I)
      {
         if (I->getKind() == clang::TemplateArgument::Type) {
//            std::string arg;
//            arg = R__GetQualifiedName( I->getAsType(), *clxx ); 
//            fprintf(stderr,"DEBUG: looking at %s\n", arg.c_str());
            result |= hasOpaqueTypedef(I->getAsType(), normCtxt);
         }
      }
   }
   return result;   
}

//______________________________________________________________________________
bool hasOpaqueTypedef(const RScanner::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Return true if any of the argument is or contains a double32.

   const clang::CXXRecordDecl* clxx =  llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   if (clxx->getTemplateSpecializationKind() == clang::TSK_Undeclared) return 0;

   clang::QualType instanceType = interp.getLookupHelper().findType(cl.GetNormalizedName());
   if (instanceType.isNull()) {
      Error(0,"Could not find the clang::Type for %s\n",cl.GetNormalizedName());
      return false;
   } else {
      return hasOpaqueTypedef(instanceType, normCtxt);
   }
}

//______________________________________________________________________________
int IsSTLContainer(const RScanner::AnnotatedRecordDecl &annotated)
{
   // Is this an STL container.
   
   const char *name = annotated.GetRequestedName()[0] ? annotated.GetRequestedName() : annotated.GetNormalizedName();
   
   int k = TClassEdit::IsSTLCont(name,1);
   
   return k;   
}

//______________________________________________________________________________
int IsSTLContainer(const clang::FieldDecl &m)
{
   // Is this an STL container?

   clang::QualType type = m.getType();
   std::string type_name = type.getAsString(m.getASTContext().getPrintingPolicy()); // m.Type()->TrueName();

   int k = TClassEdit::IsSTLCont(type_name.c_str(),1);

   //    if (k) printf(" %s==%d\n",type.c_str(),k);

   return k;
}


//______________________________________________________________________________
int IsSTLContainer(const clang::CXXBaseSpecifier &base)
{
   // Is this an STL container?

   if (!IsStdClass(*base.getType()->getAsCXXRecordDecl())) {
      return kNotSTL;
   }

   int k = TClassEdit::IsSTLCont(R__GetQualifiedName(*base.getType()->getAsCXXRecordDecl()).c_str(),1);
   //   if (k) printf(" %s==%d\n",type.c_str(),k);
   return k;
}


//______________________________________________________________________________
bool IsStreamableObject(const clang::FieldDecl &m)
{
   const char *comment = ROOT::TMetaUtils::GetComment( m ).data();

   // Transient
   if (comment[0] == '!') return false;

   clang::QualType type = m.getType();

   if (type->isReferenceType()) {
      // Reference can not be streamed.
      return false;
   }

   std::string mTypeName = type.getAsString(m.getASTContext().getPrintingPolicy());
   if (!strcmp(mTypeName.c_str(), "string") || !strcmp(mTypeName.c_str(), "string*")) {
      return true;
   }
   if (!strcmp(mTypeName.c_str(), "std::string") || !strcmp(mTypeName.c_str(), "std::string*")) {
      return true;
   }

   if (IsSTLContainer(m)) {
      return true;
   }

   const clang::Type *rawtype = type.getTypePtr()->getBaseElementTypeUnsafe ();

   if (rawtype->isPointerType()) {
      //Get to the 'raw' type.
      clang::QualType pointee;
      while ( (pointee = rawtype->getPointeeType()) , pointee.getTypePtrOrNull() && pointee.getTypePtr() != rawtype)
      {
        rawtype = pointee.getTypePtr();
      }      
   }

   if (rawtype->isFundamentalType() || rawtype->isEnumeralType()) {
      // not an ojbect.
      return false;
   }

   const clang::CXXRecordDecl *cxxdecl = rawtype->getAsCXXRecordDecl();
   if (cxxdecl && ClassInfo__HasMethod(cxxdecl,"Streamer")) {
      if (!(ClassInfo__HasMethod(cxxdecl,"Class_Version"))) return true;
      int version = GetClassVersion(cxxdecl);
      if (version > 0) return true;
   }
   return false;
}

//______________________________________________________________________________
void WriteAuxFunctions(const RScanner::AnnotatedRecordDecl &cl)
{
   // Write the functions that are need for the TGenericClassInfo.
   // This includes
   //    IsA
   //    operator new
   //    operator new[]
   //    operator delete
   //    operator delete[]

   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   if (!clxx) {
      return;
   }

   string classname( GetLong64_Name(cl.GetNormalizedName()) );
   string mappedname; 
   TMetaUtils::GetCppName(mappedname,classname.c_str());

   if ( ! TClassEdit::IsStdClass( classname.c_str() ) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      classname.insert(0,"::");
   }

   (*dictSrcOut) << "namespace ROOT {" << std::endl;

   string args;
   if (HasDefaultConstructor(clxx,&args)) {
      // write the constructor wrapper only for concrete classes
      (*dictSrcOut) << "   // Wrappers around operator new" << std::endl
                    << "   static void *new_" << mappedname.c_str() << "(void *p) {" << std::endl
                    << "      return  p ? ";
      if (HasCustomOperatorNewPlacement(*clxx)) {
         (*dictSrcOut) << "new(p) " << classname.c_str() << args << " : ";
      } else {
         (*dictSrcOut) << "::new((::ROOT::TOperatorNewHelper*)p) " << classname.c_str() << args << " : ";
      }
      (*dictSrcOut) << "new " << classname.c_str() << args << ";" << std::endl
                    << "   }" << std::endl;

      if (args.size()==0 && NeedDestructor(clxx)) {
         // Can not can newArray if the destructor is not public.
         (*dictSrcOut) << "   static void *newArray_" << mappedname.c_str() << "(Long_t nElements, void *p) {" << std::endl;
         (*dictSrcOut) << "      return p ? ";
         if (HasCustomOperatorNewArrayPlacement(*clxx)) {
            (*dictSrcOut) << "new(p) " << classname.c_str() << "[nElements] : ";
         } else {
            (*dictSrcOut) << "::new((::ROOT::TOperatorNewHelper*)p) " << classname.c_str() << "[nElements] : ";
         }
         (*dictSrcOut) << "new " << classname.c_str() << "[nElements];" << std::endl;
         (*dictSrcOut) << "   }" << std::endl;
      }
   }

   if (NeedDestructor(clxx)) {
      (*dictSrcOut) << "   // Wrapper around operator delete" << std::endl
                    << "   static void delete_" << mappedname.c_str() << "(void *p) {" << std::endl
                    << "      delete ((" << classname.c_str() << "*)p);" << std::endl
                    << "   }" << std::endl

                    << "   static void deleteArray_" << mappedname.c_str() << "(void *p) {" << std::endl
                    << "      delete [] ((" << classname.c_str() << "*)p);" << std::endl
                    << "   }" << std::endl

                    << "   static void destruct_" << mappedname.c_str() << "(void *p) {" << std::endl
                    << "      typedef " << classname.c_str() << " current_t;" << std::endl
                    << "      ((current_t*)p)->~current_t();" << std::endl
                    << "   }" << std::endl;
   }

   if (HasDirectoryAutoAdd(clxx)) {
       (*dictSrcOut) << "   // Wrapper around the directory auto add." << std::endl
                     << "   static void directoryAutoAdd_" << mappedname.c_str() << "(void *p, TDirectory *dir) {" << std::endl
                     << "      ((" << classname.c_str() << "*)p)->DirectoryAutoAdd(dir);" << std::endl
                     << "   }" << std::endl;
   }

   if (HasCustomStreamerMemberFunction(cl)) {
      (*dictSrcOut) << "   // Wrapper around a custom streamer member function." << std::endl
      << "   static void streamer_" << mappedname.c_str() << "(TBuffer &buf, void *obj) {" << std::endl
      << "      ((" << classname.c_str() << "*)obj)->" << classname.c_str() << "::Streamer(buf);" << std::endl
      << "   }" << std::endl;
   }

   if (HasNewMerge(clxx)) {
      (*dictSrcOut) << "   // Wrapper around the merge function." << std::endl
      << "   static Long64_t merge_" << mappedname.c_str() << "(void *obj,TCollection *coll,TFileMergeInfo *info) {" << std::endl
      << "      return ((" << classname.c_str() << "*)obj)->Merge(coll,info);" << std::endl
      << "   }" << std::endl;
   } else if (HasOldMerge(clxx)) {
      (*dictSrcOut) << "   // Wrapper around the merge function." << std::endl
      << "   static Long64_t  merge_" << mappedname.c_str() << "(void *obj,TCollection *coll,TFileMergeInfo *) {" << std::endl
      << "      return ((" << classname.c_str() << "*)obj)->Merge(coll);" << std::endl
      << "   }" << std::endl;
   }

   if (HasResetAfterMerge(clxx)) {
      (*dictSrcOut) << "   // Wrapper around the Reset function." << std::endl
      << "   static void reset_" << mappedname.c_str() << "(void *obj,TFileMergeInfo *info) {" << std::endl
      << "      ((" << classname.c_str() << "*)obj)->ResetAfterMerge(info);" << std::endl
      << "   }" << std::endl;
   }
   (*dictSrcOut) << "} // end of namespace ROOT for class " << classname.c_str() << std::endl << std::endl;
}

//______________________________________________________________________________
int ElementStreamer(const clang::NamedDecl &forcontext, const clang::QualType &qti, const char *R__t,int rwmode,const char *tcl=0)
{

   static const clang::CXXRecordDecl *TObject_decl = R__ScopeSearch("TObject");
   enum {
      kBIT_ISTOBJECT     = 0x10000000,
      kBIT_HASSTREAMER   = 0x20000000,
      kBIT_ISSTRING      = 0x40000000,
      
      kBIT_ISPOINTER     = 0x00001000,
      kBIT_ISFUNDAMENTAL = 0x00000020,
      kBIT_ISENUM        = 0x00000008
   };

   const clang::Type &ti( * qti.getTypePtr() );
   string tiName;
   R__GetQualifiedName(tiName, clang::QualType(&ti,0), forcontext);
   
   string objType(ShortTypeName(tiName.c_str()));

   const clang::Type *rawtype = R__GetUnderlyingType(clang::QualType(&ti,0));
   string rawname;
   R__GetQualifiedName(rawname, clang::QualType(rawtype,0), forcontext);
   
   clang::CXXRecordDecl *cxxtype = rawtype->getAsCXXRecordDecl() ;
   int isStre = cxxtype && ClassInfo__HasMethod(cxxtype,"Streamer");
   int isTObj = cxxtype && (R__IsBase(cxxtype,TObject_decl) || rawname == "TObject");
 
   long kase = 0;   

   if (ti.isPointerType())           kase |= kBIT_ISPOINTER;
   if (rawtype->isFundamentalType()) kase |= kBIT_ISFUNDAMENTAL;
   if (rawtype->isEnumeralType())    kase |= kBIT_ISENUM;


   if (isTObj)              kase |= kBIT_ISTOBJECT;
   if (isStre)              kase |= kBIT_HASSTREAMER;
   if (tiName == "string")  kase |= kBIT_ISSTRING;
   if (tiName == "string*") kase |= kBIT_ISSTRING;
   
   
   if (tcl == 0) {
      tcl = " internal error in rootcint ";
   }
   //    if (strcmp(objType,"string")==0) RStl::Instance().GenerateTClassFor( "string", interp, normCtxt  );

   if (rwmode == 0) {  //Read mode

      if (R__t) (*dictSrcOut) << "            " << tiName << " " << R__t << ";" << std::endl;
      switch (kase) {

      case kBIT_ISFUNDAMENTAL:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            R__b >> " << R__t << ";" << std::endl;
         break;

      case kBIT_ISPOINTER|kBIT_ISTOBJECT|kBIT_HASSTREAMER:
         if (!R__t)  return 1;
         (*dictSrcOut) << "            " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");"  << std::endl;
         break;

      case kBIT_ISENUM:
         if (!R__t)  return 0;
         //             fprintf(fp, "            R__b >> (Int_t&)%s;\n",R__t);
         // On some platforms enums and not 'Int_t' and casting to a reference to Int_t
         // induces the silent creation of a temporary which is 'filled' __instead of__
         // the desired enum.  So we need to take it one step at a time.
         (*dictSrcOut) << "            Int_t readtemp;" << std::endl
                       << "            R__b >> readtemp;" << std::endl
                       << "            " << R__t << " = static_cast<" << tiName << ">(readtemp);" << std::endl;
         break;

      case kBIT_HASSTREAMER:
      case kBIT_HASSTREAMER|kBIT_ISTOBJECT:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            " << R__t << ".Streamer(R__b);" << std::endl;
         break;

      case kBIT_HASSTREAMER|kBIT_ISPOINTER:
         if (!R__t)  return 1;
         //fprintf(fp, "            fprintf(stderr,\"info is %%p %%d\\n\",R__b.GetInfo(),R__b.GetInfo()?R__b.GetInfo()->GetOldVersion():-1);\n");
         (*dictSrcOut) << "            if (R__b.GetInfo() && R__b.GetInfo()->GetOldVersion()<=3) {" << std::endl;
         if (cxxtype && cxxtype->isAbstract()) {
            (*dictSrcOut) << "               R__ASSERT(0);// " << objType << " is abstract. We assume that older file could not be produced using this streaming method." << std::endl;
         } else {
            (*dictSrcOut) << "               " << R__t << " = new " << objType << ";" << std::endl
                          << "               " << R__t << "->Streamer(R__b);" << std::endl;
         }
         (*dictSrcOut) << "            } else {" << std::endl
                       << "               " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");" << std::endl
                       << "            }" << std::endl;
         break;

      case kBIT_ISSTRING:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            {TString R__str;" << std::endl
                       << "             R__str.Streamer(R__b);" << std::endl
                       << "             " << R__t << " = R__str.Data();}" << std::endl;
         break;

      case kBIT_ISSTRING|kBIT_ISPOINTER:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            {TString R__str;"  << std::endl
                       << "             R__str.Streamer(R__b);" << std::endl
                       << "             " << R__t << " = new string(R__str.Data());}" << std::endl;
         break;

      case kBIT_ISPOINTER:
         if (!R__t)  return 1;
         (*dictSrcOut) << "            " << R__t << " = (" << tiName << ")R__b.ReadObjectAny(" << tcl << ");" << std::endl;
         break;

      default:
         if (!R__t) return 1;
         (*dictSrcOut) << "            R__b.StreamObject(&" << R__t << "," << tcl << ");" << std::endl;
         break;
      }

   } else {     //Write case

      switch (kase) {

      case kBIT_ISFUNDAMENTAL:
      case kBIT_ISPOINTER|kBIT_ISTOBJECT|kBIT_HASSTREAMER:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            R__b << " << R__t << ";" << std::endl;
         break;

      case kBIT_ISENUM:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            {  void *ptr_enum = (void*)&" << R__t << ";\n";
         (*dictSrcOut) << "               R__b >> *reinterpret_cast<Int_t*>(ptr_enum); }" << std::endl;
         break;

      case kBIT_HASSTREAMER:
      case kBIT_HASSTREAMER|kBIT_ISTOBJECT:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            ((" << objType << "&)" << R__t << ").Streamer(R__b);" << std::endl;
         break;

      case kBIT_HASSTREAMER|kBIT_ISPOINTER:
         if (!R__t)  return 1;
         (*dictSrcOut) << "            R__b.WriteObjectAny(" << R__t << "," << tcl << ");" << std::endl;
         break;

      case kBIT_ISSTRING:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            {TString R__str(" << R__t << ".c_str());" << std::endl
                       << "             R__str.Streamer(R__b);};" << std::endl;
         break;

      case kBIT_ISSTRING|kBIT_ISPOINTER:
         if (!R__t)  return 0;
         (*dictSrcOut) << "            {TString R__str(" << R__t << "->c_str());" << std::endl
                       << "             R__str.Streamer(R__b);}" << std::endl;
         break;

      case kBIT_ISPOINTER:
         if (!R__t)  return 1;
         (*dictSrcOut) << "            R__b.WriteObjectAny(" << R__t << "," << tcl <<");" << std::endl;
         break;

      default:
         if (!R__t)  return 1;
         (*dictSrcOut) << "            R__b.StreamObject((" << objType << "*)&" << R__t << "," << tcl << ");" << std::endl;
         break;
      }
   }
   return 0;
}

//______________________________________________________________________________
int STLContainerStreamer(const clang::FieldDecl &m, int rwmode, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Create Streamer code for an STL container. Returns 1 if data member
   // was an STL container and if Streamer code has been created, 0 otherwise.

   int stltype = abs(IsSTLContainer(m));
   std::string mTypename;
   R__GetQualifiedName(mTypename, m.getType(), m);
   
   const clang::CXXRecordDecl* clxx = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(R__GetUnderlyingRecordDecl(m.getType()));

   if (stltype!=0) {
      //        fprintf(stderr,"Add %s (%d) which is also %s\n",
      //                m.Type()->Name(), stltype, m.Type()->TrueName() );
      clang::QualType utype(R__GetUnderlyingType(m.getType()),0);      
      RStl::Instance().GenerateTClassFor(utype,interp,normCtxt);
   }
   if (stltype<=0) return 0;
   if (clxx->getTemplateSpecializationKind() == clang::TSK_Undeclared) return 0;
   
   const clang::ClassTemplateSpecializationDecl *tmplt_specialization = llvm::dyn_cast<clang::ClassTemplateSpecializationDecl> (clxx);
   if (!tmplt_specialization) return 0;


   string stlType( ShortTypeName(mTypename.c_str()) );
   string stlName;
   stlName = ShortTypeName(m.getName().str().c_str());

   string fulName1,fulName2;
   const char *tcl1=0,*tcl2=0;
   const clang::TemplateArgument &arg0( tmplt_specialization->getTemplateArgs().get(0) );
   clang::QualType ti = arg0.getAsType();

   if (ElementStreamer(m, ti, 0, rwmode)) {
      tcl1="R__tcl1";
      fulName1 = ti.getAsString(); // Should we be passing a context?
   }
   if (stltype==kMap || stltype==kMultiMap) {
      const clang::TemplateArgument &arg1( tmplt_specialization->getTemplateArgs().get(1) );
      clang::QualType tmplti = arg1.getAsType();
      if (ElementStreamer(m, tmplti, 0, rwmode)) {
         tcl2="R__tcl2";
         fulName2 = tmplti.getAsString(); // Should we be passing a context?
      }
   }

   int isArr = 0;
   int len = 1;
   int pa = 0;
   const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(m.getType().getTypePtr());
   if (arrayType) {
      isArr = 1;
      len =  R__GetFullArrayLength(arrayType);
      pa = 1;
      while (arrayType) {
         if (arrayType->getArrayElementTypeNoTypeQual()->isPointerType()) {
            pa = 3;
            break;
         }
         arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
      }
   } else if (m.getType()->isPointerType()) {
      pa = 2;
   }
   if (rwmode == 0) {
      // create read code
      (*dictSrcOut) << "      {" << std::endl;
      if (isArr) {
         (*dictSrcOut) << "         for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
      }

      switch (pa) {
      case 0:         //No pointer && No array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << ";" << std::endl;
         break;
      case 1:         //No pointer && array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << "[R__l];" << std::endl;
         break;
      case 2:         //pointer && No array
         (*dictSrcOut) << "         delete *" << stlName.c_str() << ";"<< std::endl
                       << "         *" << stlName.c_str() << " = new " << stlType.c_str() << ";" << std::endl
                       << "         " << stlType.c_str() << " &R__stl = **" << stlName.c_str() << ";" << std::endl;
         break;
      case 3:         //pointer && array
         (*dictSrcOut) << "         delete " << stlName.c_str() << "[R__l];" << std::endl
                       << "         " << stlName.c_str() << "[R__l] = new " << stlType.c_str() << ";" << std::endl
                       << "         " << stlType.c_str() << " &R__stl = *" << stlName.c_str() << "[R__l];" << std::endl;
         break;
      }

      (*dictSrcOut) << "         R__stl.clear();" << std::endl;

      if (tcl1) {
         (*dictSrcOut) << "         TClass *R__tcl1 = TBuffer::GetClass(typeid(" << fulName1.c_str() << "));" << std::endl
                       << "         if (R__tcl1==0) {" << std::endl
                       << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
                       << fulName1.c_str() << "!\");"  << std::endl
                       << "            return;" << std::endl
                       << "         }" << std::endl;
      }
      if (tcl2) {
         (*dictSrcOut) << "         TClass *R__tcl2 = TBuffer::GetClass(typeid(" << fulName2.c_str() << "));" << std::endl
                       << "         if (R__tcl2==0) {" << std::endl
                       << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
                       << fulName2.c_str() <<"!\");" << std::endl
                       << "            return;" << std::endl
                       << "         }" << std::endl;
      }

      (*dictSrcOut) << "         int R__i, R__n;" << std::endl
                    << "         R__b >> R__n;" << std::endl;

      if (stltype==kVector) {
         (*dictSrcOut) << "         R__stl.reserve(R__n);" << std::endl;
      }
      (*dictSrcOut) << "         for (R__i = 0; R__i < R__n; R__i++) {" << std::endl;

      ElementStreamer(m, arg0.getAsType(), "R__t", rwmode, tcl1);
      if (stltype == kMap || stltype == kMultiMap) {     //Second Arg
         const clang::TemplateArgument &arg1( tmplt_specialization->getTemplateArgs().get(1) );
         ElementStreamer(m, arg1.getAsType(), "R__t2", rwmode, tcl2);
      }

      /* Need to go from
         type R__t;
         R__t.Stream;
         vec.push_back(R__t);
         to
         vec.push_back(type());
         R__t_p = &(vec.last());
         *R__t_p->Stream;

      */
      switch (stltype) {

      case kMap:
      case kMultiMap: {
         std::string keyName( ti.getAsString() );
         (*dictSrcOut) << "            typedef " << keyName << " Value_t;" << std::endl
                       << "            std::pair<Value_t const, " << tmplt_specialization->getTemplateArgs().get(1).getAsType().getAsString() << " > R__t3(R__t,R__t2);" << std::endl
                       << "            R__stl.insert(R__t3);" << std::endl;
         //fprintf(fp, "            R__stl.insert(%s::value_type(R__t,R__t2));\n",stlType.c_str());
         break;
      }
      case kSet:
      case kMultiSet:
         (*dictSrcOut) << "            R__stl.insert(R__t);" << std::endl;
         break;
      case kVector:
      case kList:
      case kDeque:
         (*dictSrcOut) << "            R__stl.push_back(R__t);" << std::endl;
         break;

      default:
         assert(0);
      }
      (*dictSrcOut) << "         }" << std::endl
                    << "      }" << std::endl;
      if (isArr) (*dictSrcOut) << "    }" << std::endl;

   } else {

      // create write code
      if (isArr) {
         (*dictSrcOut) << "         for (Int_t R__l = 0; R__l < " << len << "; R__l++) {" << std::endl;
      }
      (*dictSrcOut) << "      {" << std::endl;
      switch (pa) {
      case 0:         //No pointer && No array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << ";" << std::endl;
         break;
      case 1:         //No pointer && array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl =  " << stlName.c_str() << "[R__l];" << std::endl;
         break;
      case 2:         //pointer && No array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl = **" << stlName.c_str() << ";" << std::endl;
         break;
      case 3:         //pointer && array
         (*dictSrcOut) << "         " << stlType.c_str() << " &R__stl = *" << stlName.c_str() << "[R__l];" << std::endl;
         break;
      }

      (*dictSrcOut) << "         int R__n=(&R__stl) ? int(R__stl.size()) : 0;" << std::endl
                    << "         R__b << R__n;" << std::endl
                    << "         if(R__n) {" << std::endl;

      if (tcl1) {
         (*dictSrcOut) << "         TClass *R__tcl1 = TBuffer::GetClass(typeid(" << fulName1.c_str() << "));" << std::endl
                       << "         if (R__tcl1==0) {" << std::endl
                       << "            Error(\"" << stlName.c_str() << " streamer\",\"Missing the TClass object for "
                       << fulName1.c_str() << "!\");" << std::endl
                       << "            return;" << std::endl
                       << "         }" << std::endl;
      }
      if (tcl2) {
         (*dictSrcOut) << "         TClass *R__tcl2 = TBuffer::GetClass(typeid(" << fulName2.c_str() << "));" << std::endl
                       << "         if (R__tcl2==0) {" << std::endl
                       << "            Error(\"" << stlName.c_str() << "streamer\",\"Missing the TClass object for " << fulName2.c_str() << "!\");" << std::endl
                       << "            return;" << std::endl
                       << "         }" << std::endl;
      }

      (*dictSrcOut) << "            " << stlType.c_str() << "::iterator R__k;" << std::endl
                    << "            for (R__k = R__stl.begin(); R__k != R__stl.end(); ++R__k) {" << std::endl;
      if (stltype == kMap || stltype == kMultiMap) {
         const clang::TemplateArgument &arg1( tmplt_specialization->getTemplateArgs().get(1) );
         clang::QualType tmplti = arg1.getAsType();
         ElementStreamer(m, ti, "((*R__k).first )",rwmode,tcl1);
         ElementStreamer(m, tmplti, "((*R__k).second)",rwmode,tcl2);
      } else {
         ElementStreamer(m, ti, "(*R__k)"         ,rwmode,tcl1);
      }

      (*dictSrcOut) << "            }" << std::endl
                    << "         }" << std::endl
                    << "      }" << std::endl;
      if (isArr) (*dictSrcOut) << "    }" << std::endl;
   }
   return 1;
}

//______________________________________________________________________________
int STLStringStreamer(const clang::FieldDecl &m, int rwmode)
{
   // Create Streamer code for a standard string object. Returns 1 if data
   // member was a standard string and if Streamer code has been created,
   // 0 otherwise.

   std::string mTypenameStr;
   R__GetQualifiedName(mTypenameStr, m.getType(),m);
   // Note: here we could to a direct type comparison!
   const char *mTypeName = ShortTypeName(mTypenameStr.c_str());
   if (!strcmp(mTypeName, "string")) {
      
      std::string fieldname =  m.getName().str();
      if (rwmode == 0) {
         // create read mode
         if (m.getType()->isConstantArrayType()) {
            if (m.getType().getTypePtr()->getArrayElementTypeNoTypeQual()->isPointerType()) {
               (*dictSrcOut) << "// Array of pointer to std::string are not supported (" << fieldname << "\n";
            } else {
               std::stringstream fullIdx;
               const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(m.getType().getTypePtr());
               int dim = 0;
               while (arrayType) {
                  (*dictSrcOut) << "      for (int R__i" << dim << "=0; R__i" << dim << "<"
                                << arrayType->getSize().getLimitedValue() << "; ++R__i" << dim << " )" << std::endl;
                  fullIdx << "[R__i" << dim << "]";
                  arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
                  ++dim;
               }
               (*dictSrcOut) << "         { TString R__str; R__str.Streamer(R__b); "
                             << fieldname << fullIdx.str() << " = R__str.Data();}" << std::endl;
            }
         } else {
            (*dictSrcOut) << "      { TString R__str; R__str.Streamer(R__b); ";
            if (m.getType()->isPointerType())
               (*dictSrcOut) << "if (*" << fieldname << ") delete *" << fieldname << "; (*"
                             << fieldname << " = new string(R__str.Data())); }" << std::endl;
            else
               (*dictSrcOut) << fieldname << " = R__str.Data(); }" << std::endl;
         }
      } else {
         // create write mode
         if (m.getType()->isPointerType())
            (*dictSrcOut) << "      { TString R__str; if (*" << fieldname << ") R__str = (*"
                          << fieldname << ")->c_str(); R__str.Streamer(R__b);}" << std::endl;
         else if (m.getType()->isConstantArrayType()) {
            std::stringstream fullIdx;
            const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(m.getType().getTypePtr());
            int dim = 0;
            while (arrayType) {
               (*dictSrcOut) << "      for (int R__i" << dim << "=0; R__i" << dim << "<"
                             << arrayType->getSize().getLimitedValue() << "; ++R__i" << dim << " )" << std::endl;
               fullIdx << "[R__i" << dim << "]";
               arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
               ++dim;
            }
            (*dictSrcOut) << "         { TString R__str(" << fieldname << fullIdx.str() << ".c_str()); R__str.Streamer(R__b);}" << std::endl;
         } else
            (*dictSrcOut) << "      { TString R__str = " << fieldname << ".c_str(); R__str.Streamer(R__b);}" << std::endl;
      }
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
bool isPointerToPointer(const clang::FieldDecl &m)
{
   if (m.getType()->isPointerType()) {
      if (m.getType()->getPointeeType()->isPointerType()) {
         return true;
      }
   }
   return false;
}

//______________________________________________________________________________
void WriteArrayDimensions(const clang::QualType &type)
{
   // Write "[0]" for all but the 1st dimension.

   const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(type.getTypePtr());
   if (arrayType) {
      arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
      while(arrayType) {
         (*dictSrcOut) << "[0]";
         arrayType = llvm::dyn_cast<clang::ConstantArrayType>(arrayType->getArrayElementTypeNoTypeQual());
      }
   }
}

//______________________________________________________________________________
int WriteNamespaceHeader(std::ostream &out, const clang::DeclContext *ctxt)
{
   // Write all the necessary opening part of the namespace and
   // return the number of closing brackets needed
   // For example for Space1::Space2
   // we write: namespace Space1 { namespace Space2 {
   // and return 2.
   
   int closing_brackets = 0;

   //fprintf(stderr,"DEBUG: in WriteNamespaceHeader for %s with %s\n",
   //    cl.Fullname(),namespace_obj.Fullname());
   if (ctxt && ctxt->isNamespace()) {
      closing_brackets = WriteNamespaceHeader(out,ctxt->getParent());
      for (int indent = 0; indent < closing_brackets; ++indent) {
         out << "   ";
      }
      const clang::NamespaceDecl *ns = llvm::dyn_cast<clang::NamespaceDecl>(ctxt);
      out << "namespace " << ns->getNameAsString() << " {" << std::endl;
      closing_brackets++;
   }
   
   return closing_brackets;
}

//______________________________________________________________________________
int WriteNamespaceHeader(std::ostream &out, const clang::RecordDecl &cl)
{
   const clang::DeclContext *ctxt = R__GetEnclosingSpace(cl);
   return WriteNamespaceHeader(out,ctxt);
}

//______________________________________________________________________________
void WriteClassFunctions(const clang::CXXRecordDecl *cl)
{
   // Write the code to set the class name and the initialization object.

   bool add_template_keyword = NeedTemplateKeyword(cl);

   string fullname;
   string clsname;
   string nsname;
   int enclSpaceNesting = 0;

   if (R__GetNameWithinNamespace(fullname,clsname,nsname,cl)) {
      enclSpaceNesting = WriteNamespaceHeader(*dictSrcOut,*cl);
   }

   (*dictSrcOut) << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "TClass *" << clsname.c_str() << "::fgIsA = 0;  // static to hold class pointer" << std::endl
                 << std::endl

                 << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "const char *" << clsname.c_str() << "::Class_Name()" << std::endl << "{" << std::endl
                 << "   return \"" << fullname.c_str() << "\";"  << std::endl <<"}" << std::endl << std::endl;

   (*dictSrcOut) << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "const char *" << clsname.c_str() << "::ImplFileName()"  << std::endl << "{" << std::endl
                 << "   return ::ROOT::GenerateInitInstanceLocal((const ::" << fullname.c_str()
                 << "*)0x0)->GetImplFileName();" << std::endl << "}" << std::endl << std::endl

                 << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) <<"template <> ";
   (*dictSrcOut) << "int " << clsname.c_str() << "::ImplFileLine()" << std::endl << "{" << std::endl
                 << "   return ::ROOT::GenerateInitInstanceLocal((const ::" << fullname.c_str()
                 << "*)0x0)->GetImplFileLine();" << std::endl << "}" << std::endl << std::endl

                 << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "void " << clsname.c_str() << "::Dictionary()" << std::endl << "{" << std::endl
                 << "   fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::" << fullname.c_str()
                 << "*)0x0)->GetClass();" << std::endl
                 << "}" << std::endl << std::endl

                 << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "TClass *" << clsname.c_str() << "::Class()" << std::endl << "{" << std::endl;
   (*dictSrcOut) << "   if (!fgIsA) fgIsA = ::ROOT::GenerateInitInstanceLocal((const ::";
   (*dictSrcOut) << fullname.c_str() << "*)0x0)->GetClass();" << std::endl
                 << "   return fgIsA;" << std::endl
                 << "}" << std::endl << std::endl;

   while (enclSpaceNesting) {
      (*dictSrcOut) << "} // namespace " << nsname << std::endl;
      --enclSpaceNesting;
   }
}
//______________________________________________________________________________
void WriteClassInit(const RScanner::AnnotatedRecordDecl &cl_input, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Write the code to initialize the class name and the initialization object.

   const clang::CXXRecordDecl* cl = llvm::dyn_cast<clang::CXXRecordDecl>(cl_input.GetRecordDecl());

   if (cl==0) {
      return;
   }

   // coverity[fun_call_w_exception] - that's just fine.
   string classname = GetLong64_Name( cl_input.GetNormalizedName() );
   string mappedname;
   TMetaUtils::GetCppName(mappedname,classname.c_str());
   string csymbol = classname;
   string args;

   if ( ! TClassEdit::IsStdClass( classname.c_str() ) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      csymbol.insert(0,"::");
   }

   int stl = TClassEdit::IsSTLCont(classname.c_str());
   bool bset = TClassEdit::IsSTLBitset(classname.c_str());

   (*dictSrcOut) << "namespace ROOT {" << std::endl
   << "   void " << mappedname.c_str() << "_ShowMembers(void *obj, TMemberInspector &R__insp);"
   << std::endl;

   if (!ClassInfo__HasMethod(cl,"Dictionary") || R__IsTemplate(*cl))
      (*dictSrcOut) << "   static void " << mappedname.c_str() << "_Dictionary();" << std::endl;

   if (HasDefaultConstructor(cl,&args)) {
      (*dictSrcOut) << "   static void *new_" << mappedname.c_str() << "(void *p = 0);" << std::endl;
      if (args.size()==0 && NeedDestructor(cl))
         (*dictSrcOut) << "   static void *newArray_" << mappedname.c_str()
         << "(Long_t size, void *p);" << std::endl;
   }
   if (NeedDestructor(cl)) {
      (*dictSrcOut) << "   static void delete_" << mappedname.c_str() << "(void *p);" << std::endl
      << "   static void deleteArray_" << mappedname.c_str() << "(void *p);" << std::endl
      << "   static void destruct_" << mappedname.c_str() << "(void *p);" << std::endl;
   }
   if (HasDirectoryAutoAdd(cl)) {
      (*dictSrcOut)<< "   static void directoryAutoAdd_" << mappedname.c_str() << "(void *obj, TDirectory *dir);" << std::endl;
   }
   if (HasCustomStreamerMemberFunction(cl_input)) {
      (*dictSrcOut)<< "   static void streamer_" << mappedname.c_str() << "(TBuffer &buf, void *obj);" << std::endl;
   }
   if (HasNewMerge(cl) || HasOldMerge(cl)) {
      (*dictSrcOut)<< "   static Long64_t merge_" << mappedname.c_str() << "(void *obj, TCollection *coll,TFileMergeInfo *info);" << std::endl;
   }
   if (HasResetAfterMerge(cl)) {
      (*dictSrcOut)<< "   static void reset_" << mappedname.c_str() << "(void *obj, TFileMergeInfo *info);" << std::endl;
   }

   //--------------------------------------------------------------------------
   // Check if we have any schema evolution rules for this class
   //--------------------------------------------------------------------------
   SchemaRuleClassMap_t::iterator rulesIt1 = gReadRules.find( R__GetQualifiedName(*cl).c_str() );
   SchemaRuleClassMap_t::iterator rulesIt2 = gReadRawRules.find( R__GetQualifiedName(*cl).c_str() );

   MembersTypeMap_t nameTypeMap;
   CreateNameTypeMap( *cl, nameTypeMap );

   //--------------------------------------------------------------------------
   // Process the read rules
   //--------------------------------------------------------------------------
   if( rulesIt1 != gReadRules.end() ) {
      int i = 0;
      (*dictSrcOut) << std::endl;
      (*dictSrcOut) << "   // Schema evolution read functions" << std::endl;
      std::list<SchemaRuleMap_t>::iterator rIt = rulesIt1->second.begin();
      while( rIt != rulesIt1->second.end() ) {

         //--------------------------------------------------------------------
         // Check if the rules refer to valid data members
         //--------------------------------------------------------------------
         if( !HasValidDataMembers( *rIt, nameTypeMap ) ) {
            rIt = rulesIt1->second.erase(rIt);
            continue;
         }

         //---------------------------------------------------------------------
         // Write the conversion function if necassary
         //---------------------------------------------------------------------
         if( rIt->find( "code" ) != rIt->end() ) {
            WriteReadRuleFunc( *rIt, i++, mappedname, nameTypeMap, *dictSrcOut );
         }
         ++rIt;
      }
   }

   //--------------------------------------------------------------------------
   // Process the read raw rules
   //--------------------------------------------------------------------------
   if( rulesIt2 != gReadRawRules.end() ) {
      int i = 0;
      (*dictSrcOut) << std::endl;
      (*dictSrcOut) << "   // Schema evolution read raw functions" << std::endl;
      std::list<SchemaRuleMap_t>::iterator rIt = rulesIt2->second.begin();
      while( rIt != rulesIt2->second.end() ) {

         //--------------------------------------------------------------------
         // Check if the rules refer to valid data members
         //--------------------------------------------------------------------
         if( !HasValidDataMembers( *rIt, nameTypeMap ) ) {
            rIt = rulesIt2->second.erase(rIt);
            continue;
         }

         //---------------------------------------------------------------------
         // Write the conversion function
         //---------------------------------------------------------------------
         if( rIt->find( "code" ) == rIt->end() )
            continue;

         WriteReadRawRuleFunc( *rIt, i++, mappedname, nameTypeMap, *dictSrcOut );
         ++rIt;
      }
   }

   (*dictSrcOut) << std::endl << "   // Function generating the singleton type initializer" << std::endl;

   (*dictSrcOut) << "   static TGenericClassInfo *GenerateInitInstanceLocal(const " << csymbol.c_str() << "*)" << std::endl << "   {" << std::endl;



   (*dictSrcOut) << "      " << csymbol.c_str() << " *ptr = 0;" << std::endl;

   //fprintf(fp, "      static ::ROOT::ClassInfo< %s > \n",classname.c_str());
   if (ClassInfo__HasMethod(cl,"IsA") ) {
      (*dictSrcOut) << "      static ::TVirtualIsAProxy* isa_proxy = new ::TInstrumentedIsAProxy< "
      << csymbol.c_str() << " >(0);" << std::endl;
   }
   else {
      (*dictSrcOut) << "      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid("
      << csymbol.c_str() << "),0);" << std::endl;
   }
   (*dictSrcOut) << "      static ::ROOT::TGenericClassInfo " << std::endl

   << "         instance(\"" << classname.c_str() << "\", ";

   if (ClassInfo__HasMethod(cl,"Class_Version")) {
      (*dictSrcOut) << csymbol.c_str() << "::Class_Version(), ";
   } else if (bset) {
      (*dictSrcOut) << "2, "; // bitset 'version number'
   } else if (stl) {
      (*dictSrcOut) << "-2, "; // "::TStreamerInfo::Class_Version(), ";
   } else if( cl_input.HasClassVersion() ) {
      (*dictSrcOut) << cl_input.RequestedVersionNumber() << ", ";
   } else { // if (cl_input.RequestStreamerInfo()) {

      // Need to find out if the operator>> is actually defined for this class.
      static const char *versionFunc = "GetClassVersion";
//      int ncha = strlen(classname.c_str())+strlen(versionFunc)+5;
//      char *funcname= new char[ncha];
//      snprintf(funcname,ncha,"%s<%s >",versionFunc,classname.c_str());
      std::string proto = classname + "*";
      const clang::Decl* ctxt = llvm::dyn_cast<clang::Decl>((*cl_input).getDeclContext());
      const clang::FunctionDecl *methodinfo = R__GetFuncWithProto(ctxt, versionFunc, proto.c_str());
//      delete [] funcname;

      if (methodinfo &&
          R__GetFileName(methodinfo).find("Rtypes.h") == llvm::StringRef::npos) {

         // GetClassVersion was defined in the header file.
         //fprintf(fp, "GetClassVersion((%s *)0x0), ",classname.c_str());
         (*dictSrcOut) << "GetClassVersion< " << classname.c_str() << " >(), ";
      }
      //static char temporary[1024];
      //sprintf(temporary,"GetClassVersion<%s>( (%s *) 0x0 )",classname.c_str(),classname.c_str());
      //fprintf(stderr,"DEBUG: %s has value %d\n",classname.c_str(),(int)G__int(G__calc(temporary)));
   }

   std::string filename = R__GetFileName(cl_input);
   if (filename.length() > 0) {
      for (unsigned int i=0; i<filename.length(); i++) {
         if (filename[i]=='\\') filename[i]='/';
      }
   }
   (*dictSrcOut) << "\"" << filename << "\", " << R__GetLineNumber(cl_input) << "," << std::endl
                 << "                  typeid(" << csymbol.c_str() << "), DefineBehavior(ptr, ptr)," << std::endl
                 << "                  ";
   //   fprintf(fp, "                  (::ROOT::ClassInfo< %s >::ShowMembersFunc_t)&::ROOT::ShowMembers,%d);\n", classname.c_str(),cl_input.RootFlag());
   if (!NeedExternalShowMember(cl_input)) {
      if (!ClassInfo__HasMethod(cl,"ShowMembers")) (*dictSrcOut) << "0, ";
   } else {
      if (!ClassInfo__HasMethod(cl,"ShowMembers"))
         (*dictSrcOut) << "&" << mappedname.c_str() << "_ShowMembers, ";
   }

   if (ClassInfo__HasMethod(cl,"Dictionary") && !R__IsTemplate(*cl)) {
      (*dictSrcOut) << "&" << csymbol.c_str() << "::Dictionary, ";
   } else {
      (*dictSrcOut) << "&" << mappedname.c_str() << "_Dictionary, ";
   }

   Int_t rootflag = cl_input.RootFlag();
   if (HasCustomStreamerMemberFunction(cl_input)) {
      rootflag = rootflag | TClassTable__kHasCustomStreamerMember;
   }
   (*dictSrcOut) << "isa_proxy, " << rootflag << "," << std::endl
                 << "                  sizeof(" << csymbol.c_str() << ") );" << std::endl;
   if (HasDefaultConstructor(cl,&args)) {
      (*dictSrcOut) << "      instance.SetNew(&new_" << mappedname.c_str() << ");" << std::endl;
      if (args.size()==0 && NeedDestructor(cl))
         (*dictSrcOut) << "      instance.SetNewArray(&newArray_" << mappedname.c_str() << ");" << std::endl;
   }
   if (NeedDestructor(cl)) {
      (*dictSrcOut) << "      instance.SetDelete(&delete_" << mappedname.c_str() << ");" << std::endl
                    << "      instance.SetDeleteArray(&deleteArray_" << mappedname.c_str() << ");" << std::endl
                    << "      instance.SetDestructor(&destruct_" << mappedname.c_str() << ");" << std::endl;
   }
   if (HasDirectoryAutoAdd(cl)) {
      (*dictSrcOut) << "      instance.SetDirectoryAutoAdd(&directoryAutoAdd_" << mappedname.c_str() << ");" << std::endl;
   }
   if (HasCustomStreamerMemberFunction(cl_input)) {
      // We have a custom member function streamer or an older (not StreamerInfo based) automatic streamer.
      (*dictSrcOut) << "      instance.SetStreamerFunc(&streamer_" << mappedname.c_str() << ");" << std::endl;
   }
   if (HasNewMerge(cl) || HasOldMerge(cl)) {
      (*dictSrcOut) << "      instance.SetMerge(&merge_" << mappedname.c_str() << ");" << std::endl;
   }
   if (HasResetAfterMerge(cl)) {
      (*dictSrcOut) << "      instance.SetResetAfterMerge(&reset_" << mappedname.c_str() << ");" << std::endl;      
   }
   if (bset) {
      (*dictSrcOut) << "      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::"
                    << "Pushback" << "<TStdBitsetHelper< " << classname.c_str() << " > >()));" << std::endl;

      // (*dictSrcOut) << "      instance.SetStreamer(::ROOT::std_bitset_helper" << strchr(csymbol.c_str(),'<') << "::Streamer);\n";
      gNeedCollectionProxy = true;
   } else if (stl != 0 && ((stl>0 && stl<8) || (stl<0 && stl>-8)) )  {
      int idx = classname.find("<");
      int stlType = (idx!=(int)std::string::npos) ? TClassEdit::STLKind(classname.substr(0,idx).c_str()) : 0;
      const char* methodTCP=0;
      switch(stlType)  {
         case TClassEdit::kVector:
         case TClassEdit::kList:
         case TClassEdit::kDeque:
            methodTCP="Pushback";
            break;
         case TClassEdit::kMap:
         case TClassEdit::kMultiMap:
            methodTCP="MapInsert";
            break;
         case TClassEdit::kSet:
         case TClassEdit::kMultiSet:
            methodTCP="Insert";
            break;
      }
      (*dictSrcOut) << "      instance.AdoptCollectionProxyInfo(TCollectionProxyInfo::Generate(TCollectionProxyInfo::"
      << methodTCP << "< " << classname.c_str() << " >()));" << std::endl;

      gNeedCollectionProxy = true;
   }

   //---------------------------------------------------------------------------
   // Pass the schema evolution rules to TGenericClassInfo
   //---------------------------------------------------------------------------
   if( (rulesIt1 != gReadRules.end() && rulesIt1->second.size()>0) || (rulesIt2 != gReadRawRules.end()  && rulesIt2->second.size()>0) ) {
      (*dictSrcOut) << std::endl << "      ROOT::TSchemaHelper* rule;" << std::endl;
   }

   if( rulesIt1 != gReadRules.end() ) {
      (*dictSrcOut) << std::endl;
      (*dictSrcOut) << "      // the io read rules" << std::endl;
      (*dictSrcOut) << "      std::vector<ROOT::TSchemaHelper> readrules(";
      (*dictSrcOut) << rulesIt1->second.size() << ");" << std::endl;
      WriteSchemaList( rulesIt1->second, "readrules", *dictSrcOut );
      (*dictSrcOut) << "      instance.SetReadRules( readrules );" << std::endl;
   }

   if( rulesIt2 != gReadRawRules.end() ) {
      (*dictSrcOut) << std::endl;
      (*dictSrcOut) << "      // the io read raw rules" << std::endl;
      (*dictSrcOut) << "      std::vector<ROOT::TSchemaHelper> readrawrules(";
      (*dictSrcOut) << rulesIt2->second.size() << ");" << std::endl;
      WriteSchemaList( rulesIt2->second, "readrawrules", *dictSrcOut );
      (*dictSrcOut) << "      instance.SetReadRawRules( readrawrules );" << std::endl;
   }

   (*dictSrcOut) << "      return &instance;"  << std::endl
   << "   }" << std::endl;

   if (!stl && !bset && !hasOpaqueTypedef(cl_input, interp, normCtxt)) {
      // The GenerateInitInstance for STL are not unique and should not be externally accessible
      (*dictSrcOut) << "   TGenericClassInfo *GenerateInitInstance(const " << csymbol.c_str() << "*)" << std::endl
      << "   {\n      return GenerateInitInstanceLocal((" <<  csymbol.c_str() << "*)0);\n   }"
      << std::endl;
   }

   (*dictSrcOut) << "   // Static variable to force the class initialization" << std::endl;
   // must be one long line otherwise R__UseDummy does not work


   (*dictSrcOut)
   << "   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const "
   << csymbol.c_str() << "*)0x0); R__UseDummy(_R__UNIQUE_(Init));" << std::endl;

   if (!ClassInfo__HasMethod(cl,"Dictionary") || R__IsTemplate(*cl)) {
      (*dictSrcOut) <<  std::endl << "   // Dictionary for non-ClassDef classes" << std::endl
      << "   static void " << mappedname.c_str() << "_Dictionary() {" << std::endl;
      (*dictSrcOut) << "      ::ROOT::GenerateInitInstanceLocal((const " << csymbol.c_str();
      (*dictSrcOut) << "*)0x0)->GetClass();" << std::endl
      << "   }" << std::endl << std::endl;
   }

   (*dictSrcOut) << "} // end of namespace ROOT" << std::endl << std::endl;

}

//______________________________________________________________________________
void WriteNamespaceInit(const clang::NamespaceDecl *cl)
{
   // Write the code to initialize the namespace name and the initialization object.

   if (cl->isAnonymousNamespace()) {
      // Don't write a GenerateInitInstance for the anonymous namespaces.
      return;
   }

   // coverity[fun_call_w_exception] - that's just fine.
   string classname = R__GetQualifiedName(*cl).c_str();
   string mappedname;
   TMetaUtils::GetCppName(mappedname,classname.c_str());

   int nesting = 0;
   // We should probably unwind the namespace to properly nest it.
   if (classname!="ROOT") {
      string right = classname;
      int pos = right.find(":");
      if (pos==0) {
         right = right.substr(2);
         pos = right.find(":");
      }
      while(pos>=0) {
         string left = right.substr(0,pos);
         right = right.substr(pos+2);
         pos = right.find(":");
         ++nesting;
         (*dictSrcOut) << "namespace " << left << " {" << std::endl;
      }

      ++nesting;
      (*dictSrcOut) << "namespace " << right << " {" << std::endl;
   }

   (*dictSrcOut) << "   namespace ROOT {" << std::endl;

#if !defined(R__AIX)
   (*dictSrcOut) << "      inline ::ROOT::TGenericClassInfo *GenerateInitInstance();" << std::endl;
#endif

   if (!Namespace__HasMethod(cl,"Dictionary"))
      (*dictSrcOut) << "      static void " << mappedname.c_str() << "_Dictionary();" << std::endl;
   (*dictSrcOut) << std::endl

   << "      // Function generating the singleton type initializer" << std::endl

#if !defined(R__AIX)
   << "      inline ::ROOT::TGenericClassInfo *GenerateInitInstance()" << std::endl
   << "      {" << std::endl
#else
   << "      ::ROOT::TGenericClassInfo *GenerateInitInstance()" << std::endl
   << "      {" << std::endl
#endif

   << "         static ::ROOT::TGenericClassInfo " << std::endl

   << "            instance(\"" << classname.c_str() << "\", ";

   if (Namespace__HasMethod(cl,"Class_Version")) {
      (*dictSrcOut) << "::" << classname.c_str() << "::Class_Version(), ";
   } else {
      (*dictSrcOut) << "0 /*version*/, ";
   }

   std::string filename = R__GetFileName(cl);
   for (unsigned int i=0; i<filename.length(); i++) {
      if (filename[i]=='\\') filename[i]='/';
   }
   (*dictSrcOut) << "\"" << filename << "\", " << R__GetLineNumber(cl) << "," << std::endl
                 << "                     ::ROOT::DefineBehavior((void*)0,(void*)0)," << std::endl
                 << "                     ";

   if (Namespace__HasMethod(cl,"Dictionary")) {
      (*dictSrcOut) << "&::" << classname.c_str() << "::Dictionary, ";
   } else {
      (*dictSrcOut) << "&" << mappedname.c_str() << "_Dictionary, ";
   }

   (*dictSrcOut) << 0 << ");" << std::endl

   << "         return &instance;" << std::endl
   << "      }" << std::endl
   << "      // Insure that the inline function is _not_ optimized away by the compiler\n"
   << "      ::ROOT::TGenericClassInfo *(*_R__UNIQUE_(InitFunctionKeeper))() = &GenerateInitInstance;  " << std::endl
   << "      // Static variable to force the class initialization" << std::endl
   // must be one long line otherwise R__UseDummy does not work
   << "      static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstance();"
   << " R__UseDummy(_R__UNIQUE_(Init));" << std::endl;

   if (!Namespace__HasMethod(cl,"Dictionary")) {
      (*dictSrcOut) <<  std::endl << "      // Dictionary for non-ClassDef classes" << std::endl
      << "      static void " << mappedname.c_str() << "_Dictionary() {" << std::endl
      << "         GenerateInitInstance()->GetClass();" << std::endl
      << "      }" << std::endl << std::endl;
   }

   (*dictSrcOut) << "   }" << std::endl;
   while(nesting--) {
      (*dictSrcOut) << "}" << std::endl;
   }
   (*dictSrcOut) <<  std::endl;
}

//______________________________________________________________________________
const char *ShortTypeName(const char *typeDesc)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // we remove * and const keywords. (we do not want to remove & ).
   // You need to use the result immediately before it is being overwritten.

   static char t[4096];
   static const char* constwd = "const ";
   static const char* constwdend = "const";

   const char *s;
   char *p=t;
   int lev=0;
   for (s=typeDesc;*s;s++) {
      if (*s=='<') lev++;
      if (*s=='>') lev--;
      if (lev==0 && *s=='*') continue;
      if (lev==0 && (strncmp(constwd,s,strlen(constwd))==0
                     ||strcmp(constwdend,s)==0 ) ) {
         s+=strlen(constwd)-1; // -1 because the loop adds 1
         continue;
      }
      if (lev==0 && *s==' ' && *(s+1)!='*') { p = t; continue;}
      if (p - t > (long)sizeof(t)) {
         printf("ERROR (rootcint): type name too long for StortTypeName: %s\n",
                typeDesc);
         p[0] = 0;
         return t;
      }
      *p++ = *s;
   }
   p[0]=0;

   return t;
}

//______________________________________________________________________________
std::string ShortTypeName(const clang::FieldDecl &m)
{
   // Return the absolute type of typeDesc.
   // E.g.: typeDesc = "class TNamed**", returns "TNamed".
   // we remove * and const keywords. (we do not want to remove & ).
   // You need to use the result immediately before it is being overwritten.
   
   const clang::Type *rawtype = m.getType().getTypePtr();
   
   //Get to the 'raw' type.
   clang::QualType pointee;
   while ( rawtype->isPointerType() && ((pointee = rawtype->getPointeeType()) , pointee.getTypePtrOrNull()) && pointee.getTypePtr() != rawtype)
   {
      rawtype = pointee.getTypePtr();
   }

   std::string result;
   R__GetQualifiedName(result, clang::QualType(rawtype,0), m);
   return result;
}
   
//______________________________________________________________________________
const char *GrabIndex(const clang::FieldDecl &member, int printError)
{
   // GrabIndex returns a static string (so use it or copy it immediatly, do not
   // call GrabIndex twice in the same expression) containing the size of the
   // array data member.
   // In case of error, or if the size is not specified, GrabIndex returns 0.

   int error;
   const char *where = 0;

   const char *index = ROOT::TMetaUtils::DataMemberInfo__ValidArrayIndex(member,&error, &where);
   if (index==0 && printError) {
      const char *errorstring;
      switch (error) {
      case TMetaUtils::NOT_INT:
         errorstring = "is not an integer";
         break;
      case TMetaUtils::NOT_DEF:
         errorstring = "has not been defined before the array";
         break;
      case TMetaUtils::IS_PRIVATE:
         errorstring = "is a private member of a parent class";
         break;
      case TMetaUtils::UNKNOWN:
         errorstring = "is not known";
         break;
      default:
         errorstring = "UNKNOWN ERROR!!!!";
      }

      if (where==0) {
         Error(0, "*** Datamember %s::%s: no size indication!\n",
               member.getParent()->getName().str().c_str(), member.getName().str().c_str());
      } else {
         Error(0,"*** Datamember %s::%s: size of array (%s) %s!\n",
               member.getParent()->getName().str().c_str(), member.getName().str().c_str(), where, errorstring);
      }
   }
   return index;
}

//______________________________________________________________________________
void WriteStreamer(const RScanner::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   if (clxx == 0) return;
   
   bool add_template_keyword = NeedTemplateKeyword(clxx);
   
   string fullname;
   string clsname;
   string nsname;
   int enclSpaceNesting = 0;

   if (R__GetNameWithinNamespace(fullname,clsname,nsname,clxx)) {
      enclSpaceNesting = WriteNamespaceHeader(*dictSrcOut,*cl);
   }
   
   (*dictSrcOut) << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "void " << clsname << "::Streamer(TBuffer &R__b)"  << std::endl << "{" << std::endl
                 << "   // Stream an object of class " << fullname << "." << std::endl << std::endl;

   // In case of VersionID<=0 write dummy streamer only calling
   // its base class Streamer(s). If no base class(es) let Streamer
   // print error message, i.e. this Streamer should never have been called.
   int version = GetClassVersion(clxx);
   if (version <= 0) {
      // We also need to look at the base classes.
      int basestreamer = 0;
      for(clang::CXXRecordDecl::base_class_const_iterator iter = clxx->bases_begin(), end = clxx->bases_end();
          iter != end;
          ++iter)
      {
         if (ClassInfo__HasMethod(iter->getType()->getAsCXXRecordDecl (),"Streamer")) {
            string base_fullname;
            R__GetQualifiedName(base_fullname,* iter->getType()->getAsCXXRecordDecl ());

            if (strstr(base_fullname.c_str(),"::")) {
               // there is a namespace involved, trigger MS VC bug workaround
               (*dictSrcOut) << "   //This works around a msvc bug and should be harmless on other platforms" << std::endl
                             << "   typedef " << base_fullname << " baseClass" << basestreamer << ";" << std::endl
                             << "   baseClass" << basestreamer << "::Streamer(R__b);" << std::endl;
            }
            else {
               (*dictSrcOut) << "   " << base_fullname << "::Streamer(R__b);" << std::endl;
            }
            basestreamer++;
         }
      }
      if (!basestreamer) {
         (*dictSrcOut) << "   ::Error(\"" << fullname << "::Streamer\", \"version id <=0 in ClassDef,"
            " dummy Streamer() called\"); if (R__b.IsReading()) { }" << std::endl;
      }
      (*dictSrcOut) << "}" << std::endl << std::endl;
      while (enclSpaceNesting) {
         (*dictSrcOut) << "} // namespace " << nsname.c_str() << std::endl;
         --enclSpaceNesting;
      }
      return;
   }

   // loop twice: first time write reading code, second time writing code
   string classname = fullname;
   if (strstr(fullname.c_str(),"::")) {
      // there is a namespace involved, trigger MS VC bug workaround
      (*dictSrcOut) << "   //This works around a msvc bug and should be harmless on other platforms" << std::endl
                    << "   typedef ::" << fullname << " thisClass;" << std::endl;
      classname = "thisClass";
   }
   for (int i = 0; i < 2; i++) {

      int decli = 0;

      if (i == 0) {
         (*dictSrcOut) << "   UInt_t R__s, R__c;" << std::endl;
         (*dictSrcOut) << "   if (R__b.IsReading()) {" << std::endl;
         (*dictSrcOut) << "      Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }" << std::endl;
      } else {
         (*dictSrcOut) << "      R__b.CheckByteCount(R__s, R__c, " << classname.c_str() << "::IsA());" << std::endl;
         (*dictSrcOut) << "   } else {" << std::endl;
         (*dictSrcOut) << "      R__c = R__b.WriteVersion(" << classname.c_str() << "::IsA(), kTRUE);" << std::endl;
      }

      // Stream base class(es) when they have the Streamer() method
      int base=0;
      for(clang::CXXRecordDecl::base_class_const_iterator iter = clxx->bases_begin(), end = clxx->bases_end();
          iter != end;
          ++iter)
      {
         if (ClassInfo__HasMethod(iter->getType()->getAsCXXRecordDecl (),"Streamer")) {
            string base_fullname;
            R__GetQualifiedName(base_fullname,* iter->getType()->getAsCXXRecordDecl ());
            
            if (strstr(base_fullname.c_str(),"::")) {
               // there is a namespace involved, trigger MS VC bug workaround
               (*dictSrcOut) << "      //This works around a msvc bug and should be harmless on other platforms" << std::endl
                             << "      typedef " << base_fullname << " baseClass" << base << ";" << std::endl
                             << "      baseClass" << base << "::Streamer(R__b);" << std::endl;
               ++base;
            }
            else {
               (*dictSrcOut) << "      " << base_fullname << "::Streamer(R__b);" << std::endl;
            }
         }
      }
      // Stream data members
      // Loop over the non static data member.
      for(clang::RecordDecl::field_iterator field_iter = clxx->field_begin(), end = clxx->field_end();
          field_iter != end;
          ++field_iter)
      {
         const char *comment = ROOT::TMetaUtils::GetComment( **field_iter ).data();

         clang::QualType type = field_iter->getType();
         std::string type_name = type.getAsString(clxx->getASTContext().getPrintingPolicy());

         const clang::Type *underling_type = R__GetUnderlyingType(type);
         
         // we skip:
         //  - static members
         //  - members with an ! as first character in the title (comment) field
 
         //special case for Float16_t
         int isFloat16=0;
         if (strstr(type_name.c_str(),"Float16_t")) isFloat16=1;

         //special case for Double32_t
         int isDouble32=0;
         if (strstr(type_name.c_str(),"Double32_t")) isDouble32=1;

         // No need to test for static, there are not in this list.
         if (strncmp(comment, "!", 1)) {

            // fundamental type: short, int, long, etc....
            if (underling_type->isFundamentalType() || underling_type->isEnumeralType()) {
               if (type.getTypePtr()->isConstantArrayType() &&
                   type.getTypePtr()->getArrayElementTypeNoTypeQual()->isPointerType() ) 
               {
                  const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(type.getTypePtr());
                  int s = R__GetFullArrayLength(arrayType);

                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  if (i == 0) {
                     Error(0, "*** Datamember %s::%s: array of pointers to fundamental type (need manual intervention)\n", fullname.c_str(), field_iter->getName().str().c_str());
                     (*dictSrcOut) << "         ;//R__b.ReadArray(" << field_iter->getName().str() << ");" << std::endl;
                  } else {
                     (*dictSrcOut) << "         ;//R__b.WriteArray(" << field_iter->getName().str() << ", __COUNTER__);" << std::endl;
                  }
               } else if (type.getTypePtr()->isPointerType()) {
                  const char *indexvar = GrabIndex(**field_iter, i==0);
                  if (indexvar==0) {
                     if (i == 0) {
                        Error(0,"*** Datamember %s::%s: pointer to fundamental type (need manual intervention)\n", fullname.c_str(), field_iter->getName().str().c_str());
                        (*dictSrcOut) << "      //R__b.ReadArray(" << field_iter->getName().str() << ");" << std::endl;
                     } else {
                        (*dictSrcOut) << "      //R__b.WriteArray(" << field_iter->getName().str() << ", __COUNTER__);" << std::endl;
                     }
                  } else {
                     if (i == 0) {
                        (*dictSrcOut) << "      delete [] " << field_iter->getName().str() << ";" << std::endl
                                      << "      " << GetNonConstMemberName(**field_iter) << " = new "
                                      << ShortTypeName(**field_iter) << "[" << indexvar << "];" << std::endl;
                        if (isFloat16) {
                           (*dictSrcOut) << "      R__b.ReadFastArrayFloat16(" <<  GetNonConstMemberName(**field_iter)
                                         << "," << indexvar << ");" << std::endl;
                        } else if (isDouble32) {
                           (*dictSrcOut) << "      R__b.ReadFastArrayDouble32(" <<  GetNonConstMemberName(**field_iter)
                                         << "," << indexvar << ");" << std::endl;
                        } else {
                           (*dictSrcOut) << "      R__b.ReadFastArray(" << GetNonConstMemberName(**field_iter)
                                         << "," << indexvar << ");" << std::endl;
                        }
                     } else {
                        if (isFloat16) {
                           (*dictSrcOut) << "      R__b.WriteFastArrayFloat16("
                                         << field_iter->getName().str() << "," << indexvar << ");" << std::endl;
                        } else if (isDouble32) {
                           (*dictSrcOut) << "      R__b.WriteFastArrayDouble32("
                                         << field_iter->getName().str() << "," << indexvar << ");" << std::endl;
                        } else {
                           (*dictSrcOut) << "      R__b.WriteFastArray("
                                         << field_iter->getName().str() << "," << indexvar << ");" << std::endl;
                        }
                     }
                  }
               } else if (type.getTypePtr()->isArrayType()) {
                  if (i == 0) {
                     if (type.getTypePtr()->getArrayElementTypeNoTypeQual()->isArrayType()) { // if (m.ArrayDim() > 1) {
                        if ( underling_type->isEnumeralType() )
                           (*dictSrcOut) << "      R__b.ReadStaticArray((Int_t*)" << field_iter->getName().str() << ");" << std::endl;
                        else {
                           if (isFloat16) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayFloat16((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ");" << std::endl;
                           } else if (isDouble32) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayDouble32((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.ReadStaticArray((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ");" << std::endl;
                           }
                        }
                     } else {
                        if ( underling_type->isEnumeralType() ) {
                           (*dictSrcOut) << "      R__b.ReadStaticArray((Int_t*)" << field_iter->getName().str() << ");" << std::endl;
                        } else {
                           if (isFloat16) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayFloat16(" << field_iter->getName().str() << ");" << std::endl;
                           } else if (isDouble32) {
                              (*dictSrcOut) << "      R__b.ReadStaticArrayDouble32(" << field_iter->getName().str() << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.ReadStaticArray((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ");" << std::endl;
                           }
                        }
                     }
                  } else {
                     const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(type.getTypePtr());
                     int s = R__GetFullArrayLength(arrayType);

                     if (type.getTypePtr()->getArrayElementTypeNoTypeQual()->isArrayType()) {// if (m.ArrayDim() > 1) {
                        if ( underling_type->isEnumeralType() )
                           (*dictSrcOut) << "      R__b.WriteArray((Int_t*)" << field_iter->getName().str() << ", "
                                         << s << ");" << std::endl;
                        else
                           if (isFloat16) {
                              (*dictSrcOut) << "      R__b.WriteArrayFloat16((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           } else if (isDouble32) {
                              (*dictSrcOut) << "      R__b.WriteArrayDouble32((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.WriteArray((" << R__TrueName(**field_iter)
                                            << "*)" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           }
                     } else {
                        if ( underling_type->isEnumeralType() )
                           (*dictSrcOut) << "      R__b.WriteArray((Int_t*)" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                        else
                           if (isFloat16) {
                              (*dictSrcOut) << "      R__b.WriteArrayFloat16(" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           } else if (isDouble32) {
                              (*dictSrcOut) << "      R__b.WriteArrayDouble32(" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           } else {
                              (*dictSrcOut) << "      R__b.WriteArray(" << field_iter->getName().str() << ", " << s << ");" << std::endl;
                           }
                     }
                  }
               } else if ( underling_type->isEnumeralType() ) {
                  if (i == 0) {
                     (*dictSrcOut) << "      void *ptr_" << field_iter->getName().str() << " = (void*)&" << field_iter->getName().str() << ";\n";
                     (*dictSrcOut) << "      R__b >> *reinterpret_cast<Int_t*>(ptr_" << field_iter->getName().str() << ");" << std::endl;
                  } else
                     (*dictSrcOut) << "      R__b << (Int_t)" << field_iter->getName().str() << ";" << std::endl;
               } else {
                  if (isFloat16) {
                     if (i == 0)
                        (*dictSrcOut) << "      {float R_Dummy; R__b >> R_Dummy; " << GetNonConstMemberName(**field_iter)
                                      << "=Float16_t(R_Dummy);}" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << float(" << GetNonConstMemberName(**field_iter) << ");" << std::endl;
                  } else if (isDouble32) {
                     if (i == 0)
                        (*dictSrcOut) << "      {float R_Dummy; R__b >> R_Dummy; " << GetNonConstMemberName(**field_iter)
                                      << "=Double32_t(R_Dummy);}" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << float(" << GetNonConstMemberName(**field_iter) << ");" << std::endl;
                  } else {
                     if (i == 0)
                        (*dictSrcOut) << "      R__b >> " << GetNonConstMemberName(**field_iter) << ";" << std::endl;
                     else
                        (*dictSrcOut) << "      R__b << " << GetNonConstMemberName(**field_iter) << ";" << std::endl;
                  }
               }
            } else {
               // we have an object...

               // check if object is a standard string
               if (STLStringStreamer(**field_iter, i))
                  continue;

               // check if object is an STL container
               if (STLContainerStreamer(**field_iter, i, interp, normCtxt))
                  continue;

               // handle any other type of objects
               if (type.getTypePtr()->isConstantArrayType() &&
                   type.getTypePtr()->getArrayElementTypeNoTypeQual()->isPointerType()) 
               {
                  const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(type.getTypePtr());
                  int s = R__GetFullArrayLength(arrayType);

                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  if (i == 0)
                     (*dictSrcOut) << "         R__b >> " << GetNonConstMemberName(**field_iter);
                  else {
                     if (R__IsBase(**field_iter,"TObject") && R__IsBase(**field_iter,"TArray"))
                        (*dictSrcOut) << "         R__b << (TObject*)" << field_iter->getName().str();
                     else
                        (*dictSrcOut) << "         R__b << " << GetNonConstMemberName(**field_iter);
                  }
                  WriteArrayDimensions(field_iter->getType());
                  (*dictSrcOut) << "[R__i];" << std::endl;
               } else if (type.getTypePtr()->isPointerType()) {
                  // This is always good. However, in case of a pointer
                  // to an object that is guarenteed to be there and not
                  // being referenced by other objects we could use
                  //     xx->Streamer(b);
                  // Optimize this with control statement in title.
                  if (isPointerToPointer(**field_iter)) {
                     if (i == 0) {
                        Error(0, "*** Datamember %s::%s: pointer to pointer (need manual intervention)\n", fullname.c_str(), field_iter->getName().str().c_str());
                        (*dictSrcOut) << "      //R__b.ReadArray(" << field_iter->getName().str() << ");" << std::endl;
                     } else {
                        (*dictSrcOut) << "      //R__b.WriteArray(" << field_iter->getName().str() << ", __COUNTER__);";
                     }
                  } else {
                     if (R__GetQualifiedName(*R__GetUnderlyingType(field_iter->getType()),**field_iter) == "TClonesArray") {
                        (*dictSrcOut) << "      " << field_iter->getName().str() << "->Streamer(R__b);" << std::endl;
                     } else {
                        if (i == 0) {
                           // The following:
                           //    if (strncmp(m.Title(),"->",2) != 0) fprintf(fp, "      delete %s;\n", GetNonConstMemberName(**field_iter).c_str());
                           // could be used to prevent a memory leak since the next statement could possibly create a new object.
                           // In the TStreamerInfo based I/O we made the previous statement conditional on TStreamerInfo::CanDelete
                           // to allow the user to prevent some inadvisable deletions.  So we should be offering this flexibility
                           // here to and should not (technically) rely on TStreamerInfo for it, so for now we leave it as is.
                           // Note that the leak should happen from here only if the object is stored in an unsplit object
                           // and either the user request an old branch or the streamer has been customized.
                           (*dictSrcOut) << "      R__b >> " << GetNonConstMemberName(**field_iter) << ";" << std::endl;
                        } else {
                           if (R__IsBase(**field_iter,"TObject") && R__IsBase(**field_iter,"TArray"))
                              (*dictSrcOut) << "      R__b << (TObject*)" << field_iter->getName().str() << ";" << std::endl;
                           else
                              (*dictSrcOut) << "      R__b << " << GetNonConstMemberName(**field_iter) << ";" << std::endl;
                        }
                     }
                  }
               } else if (type.getTypePtr()->isArrayType()) {
                  const clang::ConstantArrayType *arrayType = llvm::dyn_cast<clang::ConstantArrayType>(type.getTypePtr());
                  int s = R__GetFullArrayLength(arrayType);

                  if (!decli) {
                     (*dictSrcOut) << "      int R__i;" << std::endl;
                     decli = 1;
                  }
                  (*dictSrcOut) << "      for (R__i = 0; R__i < " << s << "; R__i++)" << std::endl;
                  std::string mTypeNameStr;
                  R__GetQualifiedName(mTypeNameStr,field_iter->getType(),**field_iter);
                  const char *mTypeName = mTypeNameStr.c_str();
                  const char *constwd = "const ";
                  if (strncmp(constwd,mTypeName,strlen(constwd))==0) {
                     mTypeName += strlen(constwd);
                     (*dictSrcOut) << "         const_cast< " << mTypeName << " &>(" << field_iter->getName().str();
                     WriteArrayDimensions(field_iter->getType());
                     (*dictSrcOut) << "[R__i]).Streamer(R__b);" << std::endl;
                  } else {
                     (*dictSrcOut) << "         " << GetNonConstMemberName(**field_iter);
                     WriteArrayDimensions(field_iter->getType());
                     (*dictSrcOut) << "[R__i].Streamer(R__b);" << std::endl;
                  }
               } else {
                  if (ClassInfo__HasMethod(R__GetUnderlyingRecordDecl(field_iter->getType()),"Streamer")) 
                     (*dictSrcOut) << "      " << GetNonConstMemberName(**field_iter) << ".Streamer(R__b);" << std::endl;
                  else {
                     (*dictSrcOut) << "      R__b.StreamObject(&(" << field_iter->getName().str() << "),typeid("
                                   << field_iter->getName().str() << "));" << std::endl;               //R__t.Streamer(R__b);\n");
                     //VP                     if (i == 0)
                     //VP                        Error(0, "*** Datamember %s::%s: object has no Streamer() method (need manual intervention)\n",
                     //VP                                  fullname, field_iter->getName().str());
                     //VP                     fprintf(fp, "      //%s.Streamer(R__b);\n", m.Name());
                  }
               }
            }
         }
      }
   }
   (*dictSrcOut) << "      R__b.SetByteCount(R__c, kTRUE);" << std::endl;
   (*dictSrcOut) << "   }" << std::endl
                 << "}" << std::endl << std::endl;

   while (enclSpaceNesting) {
      (*dictSrcOut) << "} // namespace " << nsname.c_str() << std::endl;
      --enclSpaceNesting;
   }
}

//______________________________________________________________________________
void WriteAutoStreamer(const RScanner::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{

   // Write Streamer() method suitable for automatic schema evolution.

   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   if (clxx == 0) return;
   
   bool add_template_keyword = NeedTemplateKeyword(clxx);
   
   // We also need to look at the base classes.
   for(clang::CXXRecordDecl::base_class_const_iterator iter = clxx->bases_begin(), end = clxx->bases_end();
       iter != end;
       ++iter)
   {
      int k = IsSTLContainer(*iter);
      if (k!=0) {
         RStl::Instance().GenerateTClassFor( iter->getType(), interp, normCtxt );
      }
   }
   
   string fullname;
   string clsname;
   string nsname;
   int enclSpaceNesting = 0;

   if (R__GetNameWithinNamespace(fullname,clsname,nsname,clxx)) {
      enclSpaceNesting = WriteNamespaceHeader(*dictSrcOut,*cl);
   }

   (*dictSrcOut) << "//_______________________________________"
                 << "_______________________________________" << std::endl;
   if (add_template_keyword) (*dictSrcOut) << "template <> ";
   (*dictSrcOut) << "void " << clsname << "::Streamer(TBuffer &R__b)" << std::endl
                 << "{" << std::endl
                 << "   // Stream an object of class " << fullname << "." << std::endl << std::endl
                 << "   if (R__b.IsReading()) {" << std::endl
                 << "      R__b.ReadClassBuffer(" << fullname << "::Class(),this);" << std::endl
                 << "   } else {" << std::endl
                 << "      R__b.WriteClassBuffer(" << fullname << "::Class(),this);" << std::endl
                 << "   }" << std::endl
                 << "}" << std::endl << std::endl;

   while (enclSpaceNesting) {
      (*dictSrcOut) << "} // namespace " << nsname << std::endl;
      --enclSpaceNesting;
   }
}

//______________________________________________________________________________
void WritePointersSTL(const RScanner::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   // Write interface function for STL members

   string a;
   string clName;
   TMetaUtils::GetCppName(clName, R__GetFileName(cl.GetRecordDecl()).str().c_str());
   int version = GetClassVersion(cl.GetRecordDecl());
   if (version == 0) return;
   if (version < 0 && !(cl.RequestStreamerInfo()) ) return;


   const clang::CXXRecordDecl *clxx = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());
   if (clxx == 0) return;

   // We also need to look at the base classes.
   for(clang::CXXRecordDecl::base_class_const_iterator iter = clxx->bases_begin(), end = clxx->bases_end();
       iter != end;
       ++iter)
   {
      int k = IsSTLContainer(*iter);
      if (k!=0) {
         RStl::Instance().GenerateTClassFor( iter->getType(), interp, normCtxt);
      }
   }

   // Loop over the non static data member.
   for(clang::RecordDecl::field_iterator field_iter = clxx->field_begin(), end = clxx->field_end();
       field_iter != end;
       ++field_iter)
   {
      std::string mTypename;
      R__GetQualifiedName(mTypename, field_iter->getType(), *clxx);

      //member is a string
      {
         const char*shortTypeName = ShortTypeName(mTypename.c_str());
         if (!strcmp(shortTypeName, "string")) {
            continue;
         }
      }

      if (!IsStreamableObject(**field_iter)) continue;

      int k = IsSTLContainer( **field_iter );
      if (k!=0) {
         //          fprintf(stderr,"Add %s which is also",m.Type()->Name());
         //          fprintf(stderr," %s\n",R__TrueName(**field_iter) );
         clang::QualType utype(R__GetUnderlyingType(field_iter->getType()),0);
         RStl::Instance().GenerateTClassFor(utype, interp, normCtxt);
      }      
   }

}

//______________________________________________________________________________
void WriteBodyShowMembers(const RScanner::AnnotatedRecordDecl &cl, bool outside)
{
   string csymbol;
   R__GetQualifiedName(csymbol,*cl);

   if ( ! IsStdClass(*cl) ) {

      // Prefix the full class name with '::' except for the STL
      // containers and std::string.  This is to request the
      // real class instead of the class in the namespace ROOT::Shadow
      csymbol.insert(0,"::");
   }

   std::string getClass;
   if (ClassInfo__HasMethod(cl,"IsA") && !outside) {
      getClass = csymbol + "::IsA()";
   } else {
      getClass = "::ROOT::GenerateInitInstanceLocal((const ";
      getClass += csymbol + "*)0x0)->GetClass()";
   }
   if (outside) {
      (*dictSrcOut) << "   gInterpreter->InspectMembers(R__insp, obj, "
                    << getClass << ");" << std::endl;
   } else {
      (*dictSrcOut) << "   gInterpreter->InspectMembers(R__insp, this, "
                    << getClass << ");" << std::endl;
   }
}

//______________________________________________________________________________
void WriteShowMembers(const RScanner::AnnotatedRecordDecl &cl, bool outside = false)
{
   const clang::CXXRecordDecl* cxxdecl = llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl());

   (*dictSrcOut) << "//_______________________________________";
   (*dictSrcOut) << "_______________________________________" << std::endl;

   string classname = GetLong64_Name( cl.GetNormalizedName() );
   string mappedname;
   TMetaUtils::GetCppName(mappedname, classname.c_str());

   if (outside || R__IsTemplate(*cl)) {
      (*dictSrcOut) << "namespace ROOT {" << std::endl

                    << "   void " << mappedname.c_str() << "_ShowMembers(void *obj, TMemberInspector &R__insp)"
                    << std::endl << "   {" << std::endl;
      WriteBodyShowMembers(cl, outside || R__IsTemplate(*cl));
      (*dictSrcOut) << "   }" << std::endl << std::endl;
      (*dictSrcOut) << "}" << std::endl << std::endl;
   }

   if (!outside) {
      string fullname;
      string clsname;
      string nsname;
      int enclSpaceNesting = 0;
      
      if (R__GetNameWithinNamespace(fullname,clsname,nsname,cxxdecl)) {
         enclSpaceNesting = WriteNamespaceHeader(*dictSrcOut,*cl);
      }
   
      bool add_template_keyword = NeedTemplateKeyword(cxxdecl);
      if (add_template_keyword) (*dictSrcOut) << "template <> ";
      (*dictSrcOut) << "void " << clsname << "::ShowMembers(TMemberInspector &R__insp)"
                    << std::endl << "{" << std::endl;
      if (!R__IsTemplate(*cl)) {
         WriteBodyShowMembers(cl, outside);
      } else {
         string clnameNoDefArg = GetLong64_Name( cl.GetNormalizedName() );
         string mappednameNoDefArg;
         TMetaUtils::GetCppName(mappednameNoDefArg, clnameNoDefArg.c_str());

         (*dictSrcOut) <<  "   ::ROOT::" << mappednameNoDefArg.c_str() << "_ShowMembers(this, R__insp);" << std::endl;
      }
      (*dictSrcOut) << "}" << std::endl << std::endl;

      while (enclSpaceNesting) {
         (*dictSrcOut) << "} // namespace " << nsname << std::endl;
         --enclSpaceNesting;
      }
   }
}

//______________________________________________________________________________
void WriteClassCode(const RScanner::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt)
{
   std::string fullname;
   R__GetQualifiedName(fullname,cl);
   if (TClassEdit::IsSTLCont(fullname.c_str()) ) {
      RStl::Instance().GenerateTClassFor(cl.GetNormalizedName(), llvm::dyn_cast<clang::CXXRecordDecl>(cl.GetRecordDecl()), interp, normCtxt);
      return;
   }

   if (ClassInfo__HasMethod(cl,"Streamer")) {
      if (cl.RootFlag()) WritePointersSTL(cl, interp, normCtxt); // In particular this detect if the class has a version number.
      if (!(cl.RequestNoStreamer())) {
         if ((cl.RequestStreamerInfo() /*G__AUTOSTREAMER*/)) {
            WriteAutoStreamer(cl, interp, normCtxt);
         } else {
            WriteStreamer(cl, interp, normCtxt);
         }
      } else
         Info(0, "Class %s: Do not generate Streamer() [*** custom streamer ***]\n",fullname.c_str());
   } else {
      Info(0, "Class %s: Streamer() not declared\n", fullname.c_str());

      if (cl.RequestStreamerInfo()) WritePointersSTL(cl, interp, normCtxt);
   }
   if (ClassInfo__HasMethod(cl,"ShowMembers")) {
      WriteShowMembers(cl);
   } else {
      if (NeedExternalShowMember(cl)) {
         WriteShowMembers(cl, true);
      }
   }
   WriteAuxFunctions(cl);
}

//______________________________________________________________________________
void GenerateLinkdef(int *argc, char **argv, int iv, std::string &code_for_parser)
{
   code_for_parser += "#ifdef __CINT__\n\n";
   code_for_parser += "#pragma link off all globals;\n";
   code_for_parser += "#pragma link off all classes;\n";
   code_for_parser += "#pragma link off all functions;\n\n";

   for (int i = iv; i < *argc; i++) {
      char *s, trail[3];
      int   nostr = 0, noinp = 0, bcnt = 0, l = strlen(argv[i])-1;
      for (int j = 0; j < 3; j++) {
         if (argv[i][l] == '-') {
            argv[i][l] = '\0';
            nostr = 1;
            l--;
         }
         if (argv[i][l] == '!') {
            argv[i][l] = '\0';
            noinp = 1;
            l--;
         }
         if (argv[i][l] == '+') {
            argv[i][l] = '\0';
            bcnt = 1;
            l--;
         }
      }
      if (nostr || noinp) {
         trail[0] = 0;
         if (nostr) strlcat(trail, "-",3);
         if (noinp) strlcat(trail, "!",3);
      }
      if (bcnt) {
         strlcpy(trail, "+",3);
         if (nostr)
            Error(0, "option + mutual exclusive with -\n");
      }
      char *cls = strrchr(argv[i], '/');
      if (!cls) cls = strrchr(argv[i], '\\');
      if (cls)
         cls++;
      else
         cls = argv[i];
      if ((s = strrchr(cls, '.'))) *s = '\0';
      code_for_parser += "#pragma link C++ class ";
      code_for_parser += cls;
      if (nostr || noinp || bcnt)
         code_for_parser += trail;
      code_for_parser += ";\n";
      if (s) *s = '.';
   }

   code_for_parser += "\n#endif\n";
}

//______________________________________________________________________________
bool Which(cling::Interpreter &interp, const char *fname, string& pname)
{
   // Find file name in path specified via -I statements to Cling.
   // Return false if the file can not be found.
   // If the file is found, set pname to the full path name and return true.

   FILE *fp = 0;

   pname = fname;
#ifdef WIN32
   fp = fopen(pname.c_str(), "rb");
#else
   fp = fopen(pname.c_str(), "r");
#endif
   if (fp) {
      fclose(fp);
      return true;
   }

   llvm::SmallVector<std::string, 10> includePaths;//Why 10? Hell if I know.
   //false - no system header, false - with flags.
   interp.GetIncludePaths(includePaths, false, false);

   const size_t nPaths = includePaths.size();
   for (size_t i = 0; i < nPaths; i += 1 /* 2 */) {

      pname = includePaths[i].c_str();
#ifdef WIN32
      pname += "\\";
      static const char* fopenopts = "rb";
#else
      pname += "/";
      static const char* fopenopts = "r";
#endif
      pname += fname;
      fp = fopen(pname.c_str(), fopenopts);
      if (fp) {
         fclose(fp);
         return true;
      }         
   }
   pname = "";
   return false;
}

//______________________________________________________________________________
char *StrDup(const char *str)
{
   // Duplicate the string str. The returned string has to be deleted by
   // the user.

   if (!str) return 0;

   // allocate 20 extra characters in case of eg, vector<vector<T>>
   int nch = strlen(str)+20;
   char *s = new char[nch];
   if (s) strlcpy(s, str,nch);

   return s;
}

//______________________________________________________________________________
char *Compress(const char *str)
{
   // Remove all blanks from the string str. The returned string has to be
   // deleted by the user.

   if (!str) return 0;

   const char *p = str;
   // allocate 20 extra characters in case of eg, vector<vector<T>>
   char *s, *s1 = new char[strlen(str)+20];
   s = s1;

   while (*p) {
      // keep space for A<const B>!
      if (*p != ' ' || (p - str > 0 && isalnum(*(p-1))))
         *s++ = *p;
      p++;
   }
   *s = '\0';

   return s1;
}

//______________________________________________________________________________
const char *CopyArg(const char *original)
{
   // If the argument starts with MODULE/inc, strip it
   // to make it the name we can use in #includes.

#ifdef ROOTBUILD
   if (R__IsSelectionFile(original)) {
      return original;
   }

   const char *inc = strstr(original, "\\inc\\");
   if (!inc)
      inc = strstr(original, "/inc/");
   if (inc && strlen(inc) > 5)
      return inc + 5;
   return original;
#else
   return original;
#endif
}

//______________________________________________________________________________
void StrcpyWithEsc(string& escaped, const char *original)
{
   // Copy original into escaped BUT make sure that the \ characters
   // are properly escaped (on Windows temp files have \'s).

   int j = 0;
   escaped = "";
   while (original[j] != '\0') {
      if (original[j] == '\\')
         escaped += '\\';
      escaped += original[j++];
   }
}

//______________________________________________________________________________
void StrcpyArg(string& dest, const char *original)
{
   // Copy the command line argument, stripping MODULE/inc if
   // necessary.

   dest = CopyArg( original );
}

//______________________________________________________________________________
void StrcpyArgWithEsc(string& escaped, const char *original)
{
   // Copy the command line argument, stripping MODULE/inc if
   // necessary and then escaping string.

   escaped = CopyArg( original );
}

string dictsrc;

//______________________________________________________________________________
void CleanupOnExit(int code)
{
   // Removes tmp files, and (if code!=0) output files.

   if (code) {
      if (!dictsrc.empty()) {
         unlink(dictsrc.c_str());
         // also remove the .d file belonging to dictsrc
         size_t posExt=dictsrc.rfind('.');
         if (posExt!=string::npos) {
            dictsrc.replace(posExt, dictsrc.length(), ".d");
            unlink(dictsrc.c_str());
         }
      }
   }
   // also remove the .def file created by CINT.
   {
      size_t posExt=dictsrc.rfind('.');
      if (posExt!=string::npos) {
         dictsrc.replace(posExt, dictsrc.length(), ".def");
         unlink(dictsrc.c_str());

         size_t posSlash=dictsrc.rfind('/');
         if (posSlash==string::npos) {
            posSlash=dictsrc.rfind('\\');
         }
         if (posSlash!=string::npos) {
            dictsrc.replace(0,posSlash+1,"");
            unlink(dictsrc.c_str());
         }
      }
   }
}

enum ESourceFileKind {
   kSFKNotC,
   kSFKHeader,
   kSFKSource,
   kSFKLinkdef
};

//______________________________________________________________________________
static ESourceFileKind GetSourceFileKind(const char* filename)
{
   // Check whether the file's extension is compatible with C or C++.
   // Return whether source, header, Linkdef or nothing.
   if (filename[0] == '-') return kSFKNotC;

   const size_t len = strlen(filename);
   const char* ext = filename + len - 1;
   while (ext >= filename && *ext != '.') --ext;
   if (ext < filename || *ext != '.') return kSFKNotC;
   ++ext;
   const size_t lenExt = filename + len - ext;

   ESourceFileKind ret = kSFKNotC;
   switch (lenExt) {
   case 1: {
      const char last = toupper(filename[len - 1]);
      if (last == 'H') ret = kSFKHeader;
      else if (last == 'C') ret = kSFKSource;
      break;
   }
   case 2: {
      if (filename[len - 2] == 'h' && filename[len - 1] == 'h')
         ret = kSFKHeader;
      else if (filename[len - 2] == 'c' && filename[len - 1] == 'c')
         ret = kSFKSource;
      break;
   }
   case 3: {
      const char last = filename[len - 1];
      if ((last == 'x' || last == 'p')
          && filename[len - 2] == last) {
         if (filename[len - 3] == 'h') ret = kSFKHeader;
         else if (filename[len - 3] == 'c') ret = kSFKSource;
      }
   }
   } // switch extension length

   static const size_t lenLinkdefdot = 8;
   if (ret == kSFKHeader && len - lenExt >= lenLinkdefdot) {
      if ((strstr(filename,"LinkDef") || strstr(filename,"Linkdef") ||
           strstr(filename,"linkdef")) && strstr(filename,".h")) {
         ret = kSFKLinkdef;
      }
   }
   return ret;
}


//______________________________________________________________________________
static int GenerateModule(const char* dictSrcFile, const std::vector<std::string>& args, const std::string &currentDirectory)
{
   // Generate the clang module given the arguments.
   // Returns != 0 on error.

   std::string dictname = llvm::sys::path::stem(dictSrcFile);

   // Parse Arguments
   vector<std::string> headers;
   std::vector<const char*> compI;
   std::vector<const char*> compD;
   std::vector<const char*> compU;
   for (size_t iPcmArg = 1 /*skip argv0*/, nPcmArg = args.size();
        iPcmArg < nPcmArg; ++iPcmArg) {
      ESourceFileKind sfk = GetSourceFileKind(args[iPcmArg].c_str());
      if (sfk == kSFKHeader || sfk == kSFKSource) {
         headers.push_back(args[iPcmArg]);
      } else if (sfk == kSFKNotC && args[iPcmArg][0] == '-') {
         switch (args[iPcmArg][1]) {
         case 'I':
            if (args[iPcmArg] != "-I." &&  args[iPcmArg] != "-Iinclude") {
               compI.push_back(args[iPcmArg].c_str() + 2);
            }
            break;
         case 'D':
            if (args[iPcmArg] != "-DTRUE=1" && args[iPcmArg] != "-DFALSE=0"
                && args[iPcmArg] != "-DG__NOCINTDLL") {
               // keep -DROOT_Math_VectorUtil_Cint -DG__VECTOR_HAS_CLASS_ITERATOR?
               compD.push_back(args[iPcmArg].c_str() + 2);
            }
            break;
         case 'U': compU.push_back(args[iPcmArg].c_str() + 2); break;
         }
      }
   }

   // Dictionary initialization code for loading the module
   (*dictSrcOut) << "namespace {\n"
      "  static struct DictInit {\n"
      "    DictInit() {\n"
      "      static const char* headers[] = {\n";

   {
      for (size_t iH = 0, eH = headers.size(); iH < eH; ++iH) {
         (*dictSrcOut) << "             \"" << headers[iH] << "\"," << std::endl;
      }
   }
   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* includePaths[] = {\n";
   for (std::vector<const char*>::const_iterator
           iI = compI.begin(), iE = compI.end(); iI != iE; ++iI) {
      (*dictSrcOut) << "             \"" << *iI << "\"," << std::endl;
   }
   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* macroDefines[] = {\n";
   for (std::vector<const char*>::const_iterator
           iD = compD.begin(), iDE = compD.end(); iD != iDE; ++iD) {
      (*dictSrcOut) << "             \"" << *iD << "\"," << std::endl;
   }
   (*dictSrcOut) << 
      "      0 };\n"
      "      static const char* macroUndefines[] = {\n";
   for (std::vector<const char*>::const_iterator
           iU = compU.begin(), iUE = compU.end(); iU != iUE; ++iU) {
      (*dictSrcOut) << "             \"" << *iU << "\"," << std::endl;
   }
   (*dictSrcOut) << 
      "      0 };\n"
      "      TCintWithCling__RegisterModule(\"" << dictname << "\",\n"
      "         headers, includePaths, macroDefines, macroUndefines);\n"
      "    }\n"
      "  } __TheInitializer;\n"
      "}" << std::endl;

   clang::CompilerInstance* CI = gInterp->getCI();

// Note: need to resolve _where_ to create the pcm
   std::string dictDir = "lib/";
#ifdef WIN32
   struct _stati64 finfo;
   
   if (_stati64(dictDir.c_str(), &finfo) < 0 ||
       !(finfo.st_mode & S_IFDIR)) {
      dictDir = "./";
   }
#else
   struct stat finfo;
   if (stat(dictDir.c_str(), &finfo) < 0 ||
       !S_ISDIR(finfo.st_mode)) {
      dictDir = "./";
   }
   
#endif
   
   CI->getPreprocessor().getHeaderSearchInfo().setModuleCachePath(dictDir.c_str());
   std::string moduleFile = dictDir + ROOT::TMetaUtils::GetModuleFileName(dictname.c_str());
   clang::Module* module = 0;
   {
      std::vector<const char*> headersCStr;
      for (std::vector<std::string>::const_iterator
              iH = headers.begin(), eH = headers.end();
           iH != eH; ++iH) {
         headersCStr.push_back(iH->c_str());
      }
      headersCStr.push_back(0);
      module = ROOT::TMetaUtils::declareModuleMap(CI, moduleFile.c_str(), &headersCStr[0]);
   }

   // From PCHGenerator and friends:
   llvm::SmallVector<char, 128> Buffer;
   llvm::BitstreamWriter Stream(Buffer);
   clang::ASTWriter Writer(Stream);
   llvm::raw_ostream *OS
      = CI->createOutputFile(moduleFile, /*Binary=*/true,
                             /*RemoveFileOnSignal=*/false, /*InFile*/"",
                             /*Extension=*/"", /*useTemporary=*/false,
                             /*CreateMissingDirectories*/false);
   // Emit the PCH file
   CI->getFrontendOpts().RelocatablePCH = true;
   Writer.WriteAST(CI->getSema(), 0, moduleFile, module, "/DUMMY_ROOTSYS/include/" /*SysRoot*/);

   // Write the generated bitstream to "Out".
   OS->write((char *)&Buffer.front(), Buffer.size());

   // Make sure it hits disk now.
   OS->flush();
   delete OS;

   // Free up some memory, in case the process is kept alive.
   Buffer.clear();

   return 0;
}


// cross-compiling for iOS and iOS simulator (assumes host is Intel Mac OS X)
#if defined(R__IOSSIM) || defined(R__IOS)
#ifdef __x86_64__
#undef __x86_64__
#endif
#ifdef __i386__
#undef __i386__
#endif
#ifdef R__IOSSIM
#define __i386__ 1
#endif
#ifdef R__IOS
#define __arm__ 1
#endif
#endif

//______________________________________________________________________________
int main(int argc, char **argv)
{
   if (argc < 2) {
      fprintf(stderr,
              "Usage: %s [-v][-v0-4] [-cint|-reflex|-gccxml] [-l] [-f] [out.cxx] [-c] file1.h[+][-][!] file2.h[+][-][!]...[LinkDef.h]\n",
              argv[0]);
      fprintf(stderr, "For more extensive help type: %s -h\n", argv[0]);
      return 1;
   }

   char dictname[1024];
   int i, j, ic, ifl, force;
   int icc = 0;
   bool requestAllSymbols = false; // Would be set to true is we decide to support an option like --deep.

   std::string currentDirectory;
   R__GetCurrentDirectory(currentDirectory);

   ic = 1;
   if (!strcmp(argv[ic], "-v")) {
      gErrorIgnoreLevel = kInfo; // The default is kError
      ic++;
   } else if (!strcmp(argv[ic], "-v0")) {
      gErrorIgnoreLevel = kFatal; // Explicitly remove all messages
      ic++;
   } else if (!strcmp(argv[ic], "-v1")) {
      gErrorIgnoreLevel = kError; // Only error message (default)
      ic++;
   } else if (!strcmp(argv[ic], "-v2")) {
      gErrorIgnoreLevel = kWarning; // error and warning message
      ic++;
   } else if (!strcmp(argv[ic], "-v3")) {
      gErrorIgnoreLevel = kNote; // error, warning and note
      ic++;
   } else if (!strcmp(argv[ic], "-v4")) {
      gErrorIgnoreLevel = kInfo; // Display all information (same as -v)
      ic++;
   }
   if (ic < argc) {
      if (!strcmp(argv[ic], "-cint")) {
         // Flag is ignored, should warn of deprecation.
         ic++;
      } else if (!strcmp(argv[ic], "-reflex")) {
         // Flag is ignored, should warn of deprecation.
         ic++;
      } else if (!strcmp(argv[ic], "-gccxml")) {
         // Flag is ignored, should warn of deprecation.
         ic++;
      }
   }

   const char* libprefix = "--lib-list-prefix=";

   ifl = 0;
   while (ic < argc && strncmp(argv[ic], "-",1)==0
          && strcmp(argv[ic], "-f")!=0 ) {
      if (!strcmp(argv[ic], "-l")) {

         ic++;
      } else if (!strncmp(argv[ic],libprefix,strlen(libprefix))) {

         gLiblistPrefix = argv[ic]+strlen(libprefix);

         string filein = gLiblistPrefix + ".in";
         FILE *fp;
         if ((fp = fopen(filein.c_str(), "r")) == 0) {
            Error(0, "%s: The input list file %s does not exist\n", argv[0], filein.c_str());
            return 1;
         }
         fclose(fp);

         ic++;
      } else {
         break;
      }
   }

   if (ic < argc && !strcmp(argv[ic], "-f")) {
      force = 1;
      ic++;
   } else if (argc > 1 && (!strcmp(argv[1], "-?") || !strcmp(argv[1], "-h"))) {
      fprintf(stderr, "%s\n", help);
      return 1;
   } else if (ic < argc && !strncmp(argv[ic], "-",1)) {
      fprintf(stderr,"Usage: %s [-v][-v0-4] [-reflex] [-l] [-f] [out.cxx] [-c] file1.h[+][-][!] file2.h[+][-][!]...[LinkDef.h]\n",
              argv[0]);
      fprintf(stderr,"Only one verbose flag is authorized (one of -v, -v0, -v1, -v2, -v3, -v4)\n"
              "and must be before the -f flags\n");
      fprintf(stderr,"For more extensive help type: %s -h\n", argv[0]);
      return 1;
   } else {
      force = 0;
   }

#if defined(R__WIN32) && !defined(R__WINGCC)
   // cygwin's make is presenting us some cygwin paths even though
   // we are windows native. Convert them as good as we can.
   for (int iic = ic; iic < argc; ++iic) {
      std::string iiarg(argv[iic]);
      if (FromCygToNativePath(iiarg)) {
         size_t len = iiarg.length();
         // yes, we leak.
         char* argviic = new char[len + 1];
         strlcpy(argviic, iiarg.c_str(), len + 1);
         argv[iic] = argviic;
      }
   }
#endif

   string header("");
   if (ic < argc && (strstr(argv[ic],".C")  || strstr(argv[ic],".cpp") ||
       strstr(argv[ic],".cp") || strstr(argv[ic],".cxx") ||
       strstr(argv[ic],".cc") || strstr(argv[ic],".c++"))) {
      FILE *fp;
      if ((fp = fopen(argv[ic], "r")) != 0) {
         fclose(fp);
         if (!force) {
            Error(0, "%s: output file %s already exists\n", argv[0], argv[ic]);
            return 1;
         }
      }
      //string header( argv[ic] );
      header = argv[ic];
      int loc = strrchr(argv[ic],'.') - argv[ic];
      header[loc+1] = 'h';
      header[loc+2] = '\0';
      if ((fp = fopen(header.c_str(), "r")) != 0) {
         fclose(fp);
         if (!force) {
            Error(0, "%s: output file %s already exists\n", argv[0], header.c_str());
            return 1;
         } else {
            for (int k = ic+1; k < argc; ++k) {
               if (*argv[k] != '-' && *argv[k] != '+') {
                  if (strcmp(header.c_str(),argv[k])==0) {
                     Error(0, "%s: output file %s would overwrite one of the input files!\n", argv[0], header.c_str());
                     return 1;
                  }
                  if (strcmp(argv[ic],argv[k])==0) {
                     Error(0, "%s: output file %s would overwrite one of the input files!\n", argv[0],argv[ic]);
                     return 1;
                  }
               }
            }
         }
      }

      dictsrc=argv[ic];
      fp = fopen(argv[ic], "w");
      if (fp) fclose(fp);    // make sure file is created and empty
      ifl = ic;
      ic++;

      // remove possible pathname to get the dictionary name
      if (strlen(argv[ifl]) > (sizeof(dictname)-1)) {
         Error(0, "rootcint: dictionary name too long (more than %d characters): %s\n",
               sizeof(dictname)-1,argv[ifl]);
         CleanupOnExit(1);
         return 1;
      }
      strncpy(dictname, argv[ifl], sizeof(dictname)-1);
      char *p = 0;
      // find the right part of then name.
      for (p = dictname + strlen(dictname)-1;p!=dictname;--p) {
         if (*p =='/' ||  *p =='\\') {
            *p = 0;
            break;
         }
      }
      if (!p)
         p = dictname;
      else if (p != dictname) {
         p++;
         memmove(dictname, p, strlen(p)+1);
      }
   } else if (!strcmp(argv[1], "-?") || !strcmp(argv[1], "-h")) {
      fprintf(stderr, "%s\n", help);
      return 1;
   } else {
      ic = 1;
      if (force) ic = 2;
      ifl = 0;
   }
   
   int argcc, iv, il;
   std::vector<std::string> path;
   char *argvv[500];

   std::vector<std::string> clingArgs;
   clingArgs.push_back(argv[0]);
   clingArgs.push_back("-I.");
   clingArgs.push_back("-DROOT_Math_VectorUtil_Cint"); // ignore that little problem maker
   
   if (! R__IsPointer<std::vector<int>::iterator>::kVal) {
      // Tell cling (for parsing pragma) that std::vector's iterator is a class
      clingArgs.push_back("-DG__VECTOR_HAS_CLASS_ITERATOR");
   }

#if !defined(ROOTBUILD) && defined(ROOTINCDIR)
   SetRootSys();
#endif
   path.push_back(std::string("-I") + TMetaUtils::GetROOTIncludeDir(ROOTBUILDVAL));

   argvv[0] = argv[0];
   argcc = 1;

   if (ic < argc && !strcmp(argv[ic], "-c")) {
      icc++;
      if (ifl) {
         char *s;
         ic++;
         argvv[argcc++] = (char *)"-q0";
         argvv[argcc++] = (char *)"-n";
         int ncha = strlen(argv[ifl])+1;
         argvv[argcc] = (char *)calloc(ncha, 1);
         strlcpy(argvv[argcc], argv[ifl],ncha); argcc++;
         argvv[argcc++] = (char *)"-N";
         s = strrchr(dictname,'.');
         argvv[argcc] = (char *)calloc(strlen(dictname), 1);
         strncpy(argvv[argcc], dictname, s-dictname); argcc++;

         while (ic < argc && (*argv[ic] == '-' || *argv[ic] == '+')) {
            if (strcmp("+P", argv[ic]) == 0 ||
                strcmp("+V", argv[ic]) == 0 ||
                strcmp("+STUB", argv[ic]) == 0) {
               // break when we see positional options
               break;
            }
            if (strcmp("-pipe", argv[ic])!=0 && strcmp("-pthread", argv[ic])!=0) {
               // filter out undesirable options
               if (strcmp("-fPIC", argv[ic]) && strcmp("-fpic", argv[ic])
                   && strcmp("-p", argv[ic])) {
                  clingArgs.push_back(argv[ic]);
               }
               argvv[argcc++] = argv[ic++];
            } else {
               ic++;
            }
         }

         for (i = 0; i < (int)path.size(); i++) {
            argvv[argcc++] = (char*)path[i].c_str();
            clingArgs.push_back(path[i].c_str());
         }

#ifdef __hpux
         argvv[argcc++] = (char *)"-I/usr/include/X11R5";
#endif
         switch (gErrorIgnoreLevel) {
         case kInfo:     argvv[argcc++] = (char *)"-J4"; break;
         case kNote:     argvv[argcc++] = (char *)"-J3"; break;
         case kWarning:  argvv[argcc++] = (char *)"-J2"; break;
         case kError:    argvv[argcc++] = (char *)"-J1"; break;
         case kSysError:
         case kFatal:    argvv[argcc++] = (char *)"-J0"; break;
         default:        argvv[argcc++] = (char *)"-J1"; break;
         }

         // If the compiler's preprocessor is not used
         // we still need to declare the compiler specific flags
         // so that the header file are properly parsed.
#ifdef __KCC
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__KCC=%ld", (long)__KCC); argcc++;
#endif
#ifdef __INTEL_COMPILER
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__INTEL_COMPILER=%ld", (long)__INTEL_COMPILER); argcc++;
#endif
#ifdef __xlC__
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__xlC__=%ld", (long)__xlC__); argcc++;
#endif
#ifdef __GNUC__
         argvv[argcc] = (char *)calloc(64, 1);
         // coverity[secure_coding] - sufficient space
         snprintf(argvv[argcc],64, "-D__GNUC__=%ld", (long)__GNUC__); argcc++;
#endif
#ifdef __GNUC_MINOR__
         argvv[argcc] = (char *)calloc(64, 1);
         // coverity[secure_coding] - sufficient space
         snprintf(argvv[argcc],64, "-D__GNUC_MINOR__=%ld", (long)__GNUC_MINOR__); argcc++;
#endif
#ifdef __HP_aCC
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64 "-D__HP_aCC=%ld", (long)__HP_aCC); argcc++;
#endif
#ifdef __sun
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__sun=%ld", (long)__sun); argcc++;
#endif
#ifdef __SUNPRO_CC
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__SUNPRO_CC=%ld", (long)__SUNPRO_CC); argcc++;
#endif
#ifdef __ia64__
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__ia64__=%ld", (long)__ia64__); argcc++;
#endif
#ifdef __x86_64__
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__x86_64__=%ld", (long)__x86_64__); argcc++;
#endif
#ifdef __i386__
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__i386__=%ld", (long)__i386__); argcc++;
#endif
#ifdef __arm__
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D__arm__=%ld", (long)__arm__); argcc++;
#endif
#ifdef R__B64
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-DR__B64"); argcc++;
#endif
#ifdef _WIN32
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D_WIN32=%ld",(long)_WIN32); argcc++;
#endif
#ifdef WIN32
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-DWIN32=%ld",(long)WIN32); argcc++;
#endif
#ifdef _MSC_VER
         argvv[argcc] = (char *)calloc(64, 1);
         snprintf(argvv[argcc],64, "-D_MSC_VER=%ld",(long)_MSC_VER); argcc++;
#endif

#ifdef ROOTBUILD
         argvv[argcc++] = (char *)"-DG__NOCINTDLL";
         clingArgs.push_back(argvv[argcc - 1]);
#endif
         argvv[argcc++] = (char *)"-DTRUE=1";
         clingArgs.push_back(argvv[argcc - 1]);
         argvv[argcc++] = (char *)"-DFALSE=0";
         clingArgs.push_back(argvv[argcc - 1]);
         argvv[argcc++] = (char *)"-Dexternalref=extern";
         argvv[argcc++] = (char *)"-DSYSV";
         argvv[argcc++] = (char *)"-D__MAKECINT__";
         // NO! clang needs to see the truth.
         // clingArgs.push_back(argvv[argcc - 1]);
         argvv[argcc++] = (char *)"-V";        // include info on private members
         argvv[argcc++] = (char *)"-c-10";
         argvv[argcc++] = (char *)"+V";        // turn on class comment mode
      } else {
         Error(0, "%s: option -c can only be used when an output file has been specified\n", argv[0]);
         return 1;
      }
   }
   iv = 0;
   il = 0;

   std::vector<std::string> pcmArgs;
   for (size_t parg = 0, n = clingArgs.size(); parg < n; ++parg) {
      if (clingArgs[parg] != "-c")
         pcmArgs.push_back(clingArgs[parg]);
   }

   // cling-only arguments
   clingArgs.push_back("-fsyntax-only");
   std::string interpInclude
      = TMetaUtils::GetInterpreterExtraIncludePath(ROOTBUILDVAL);
   clingArgs.push_back(interpInclude);

   std::vector<const char*> clingArgsC;
   for (size_t iclingArgs = 0, nclingArgs = clingArgs.size();
        iclingArgs < nclingArgs; ++iclingArgs) {
      clingArgsC.push_back(clingArgs[iclingArgs].c_str());
   }

   gResourceDir = TMetaUtils::GetLLVMResourceDir(ROOTBUILDVAL);
   cling::Interpreter interp(clingArgsC.size(), &clingArgsC[0],
                             gResourceDir.c_str());
   interp.declare("namespace std {} using namespace std;");
#ifdef ROOTBUILD
   interp.declare("#include \"include/Rtypes.h\"");
   interp.declare("#include \"include/TClingRuntime.h\"");
   interp.declare("#include \"include/TObject.h\"");
#else
# ifndef ROOTINCDIR
   interp.declare("#include \"Rtypes.h\"");
   interp.declare("#include \"TClingRuntime.h\"");
   interp.declare("#include \"TObject.h\"");
# else
   interp.declare("#include \"" ROOTINCDIR "/Rtypes.h\"");
   interp.declare("#include \"" ROOTINCDIR "/TClingRuntime.h\"");
   interp.declare("#include \"" ROOTINCDIR "/TObject.h\"");
# endif
#endif
   gInterp = &interp;

   

   // For the list of 'opaque' typedef to also include string, we have to include it now.
   interp.declare("#include <string>");
  
   // We are now ready (enough is loaded) to init the list of opaque typedefs.
   ROOT::TMetaUtils::TNormalizedCtxt normCtxt(interp.getLookupHelper());
   TClassEdit::Init(interp,normCtxt);

   // flags used only for the pragma parser:
   clingArgs.push_back("-D__CINT__");
   clingArgs.push_back("-D__MAKECINT__");
   char platformDefines[64] = {0};
#ifdef __INTEL_COMPILER
   snprintf(platformDefines, 64, "-DG__INTEL_COMPILER=%ld", (long)__INTEL_COMPILER);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __xlC__
   snprintf(platformDefines, 64, "-DG__xlC=%ld", (long)__xlC__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __GNUC__
   snprintf(platformDefines, 64, "-DG__GNUC=%ld", (long)__GNUC__);
   snprintf(platformDefines, 64, "-DG__GNUC_VER=%ld", (long)__GNUC__*1000 + __GNUC_MINOR__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __GNUC_MINOR__
   snprintf(platformDefines, 64, "-DG__GNUC_MINOR=%ld", (long)__GNUC_MINOR__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __HP_aCC
   snprintf(platformDefines, 64, "-DG__HP_aCC=%ld", (long)__HP_aCC);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __sun
   snprintf(platformDefines, 64, "-DG__sun=%ld", (long)__sun);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __SUNPRO_CC
   snprintf(platformDefines, 64, "-DG__SUNPRO_CC=%ld", (long)__SUNPRO_CC);
   clingArgs.push_back(platformDefines);
#endif
#ifdef _STLPORT_VERSION
   // stlport version, used on e.g. SUN
   snprintf(platformDefines, 64, "-DG__STLPORT_VERSION=%ld", (long)_STLPORT_VERSION);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __ia64__
   snprintf(platformDefines, 64, "-DG__ia64=%ld", (long)__ia64__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __x86_64__
   snprintf(platformDefines, 64, "-DG__x86_64=%ld", (long)__x86_64__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __i386__
   snprintf(platformDefines, 64, "-DG__i386=%ld", (long)__i386__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef __arm__
   snprintf(platformDefines, 64, "-DG__arm=%ld", (long)__arm__);
   clingArgs.push_back(platformDefines);
#endif
#ifdef _WIN32
   snprintf(platformDefines, 64, "-DG__WIN32=%ld",(long)_WIN32);
   clingArgs.push_back(platformDefines);
#else
# ifdef WIN32
   snprintf(platformDefines, 64, "-DG__WIN32=%ld",(long)WIN32);
   clingArgs.push_back(platformDefines);
# endif
#endif
#ifdef _MSC_VER
   snprintf(platformDefines, 64, "-DG__MSC_VER=%ld",(long)_MSC_VER);
   clingArgs.push_back(platformDefines);
   snprintf(platformDefines, 64, "-DG__VISUAL=%ld",(long)_MSC_VER);
   clingArgs.push_back(platformDefines);
#endif

   std::string interpPragmaSource;
   std::string includeForSource;
   string esc_arg;
   for (i = ic; i < argc; i++) {
      if (!iv && *argv[i] != '-' && *argv[i] != '+') {
         if (!icc) {
            for (j = 0; j < (int)path.size(); j++) {
               argvv[argcc++] = (char*)path[j].c_str();
            }
            argvv[argcc++] = (char *)"+V";
         }
         iv = i;
      }
      if (R__IsSelectionFile(argv[i])) {
         il = i;
         if (i != argc-1) {
            Error(0, "%s: %s must be last file on command line\n", argv[0], argv[i]);
            return 1;
         }
      }
      if (!strcmp(argv[i], "-c")) {
         Error(0, "%s: option -c must come directly after the output file\n", argv[0]);
         return 1;
      }
      if (strcmp("-pipe", argv[ic])!=0) {
         // filter out undesirable options
         string argkeep;
            // coverity[tainted_data] The OS should already limit the argument size, so we are safe here
         StrcpyArg(argkeep, argv[i]);
         int ncha = argkeep.length()+1;
         // coverity[tainted_data] The OS should already limit the argument size, so we are safe here
         argvv[argcc++] = (char*)calloc(ncha,1);
         // coverity[tainted_data] The OS should already limit the argument size, so we are safe here
         strlcpy(argvv[argcc-1],argkeep.c_str(),ncha);
         
         if (*argv[i] != '-' && *argv[i] != '+') {
            // Looks like a file
            if (cling::Interpreter::kSuccess 
                == interp.declare(std::string("#include \"") + argv[i] + "\"")) {
               interpPragmaSource += std::string("#include \"") + argv[i] + "\"\n";
               std::string header( R__GetRelocatableHeaderName( argv[i], currentDirectory ) );
               if (!R__IsSelectionFile(argv[i])) 
                  includeForSource += std::string("#include \"") + header + "\"\n";
               pcmArgs.push_back(header);
            } else {
               Error(0, "%s: compilation failure\n", argv[0]);
               CleanupOnExit(1);
               return 1;
            }
            
            // remove header files from CINT view
            free(argvv[argcc-1]);
            argvv[--argcc] = 0;
         }
      }
   }

   if (!iv) {
      Error(0, "%s: no input files specified\n", argv[0]);
      CleanupOnExit(1);
      return 1;
   }

   if (!il) {
      // Generate autolinkdef
      GenerateLinkdef(&argc, argv, iv, interpPragmaSource);
   }

   // make name of dict include file "aapDict.cxx" -> "aapDict.h"
   std::string dictheader( argv[ifl] );
   size_t pos = dictheader.rfind('.');
   dictheader.erase(pos);
   dictheader.append(".h");
   
   std::string inclf(dictname);
   pos = inclf.rfind('.');
   inclf.erase(pos);
   inclf.append(".h");
   
   // Check if code goes to stdout or rootcling file
   std::ofstream fileout;
   std::ofstream headerout;
   if (ifl) {
      fileout.open(argv[ifl]);
      dictSrcOut = &fileout;
      if (!(*dictSrcOut)) {
         Error(0, "rootcint: failed to open %s in main\n",
               argv[ifl]);
         CleanupOnExit(1);
         return 1;
      }
      headerout.open(dictheader.c_str());
      dictHdrOut = &headerout;
      if (!(*dictHdrOut)) {
         Error(0, "rootcint: failed to open %s in main\n",
               dictheader.c_str());
         CleanupOnExit(1);
         return 1;
      }
   } else {
      dictSrcOut = &std::cout;
      dictHdrOut = &std::cout;
   }
   
   string main_dictname(argv[ifl]);
   size_t dh = main_dictname.rfind('.');
   if (dh != std::string::npos) {
      main_dictname.erase(dh);
   }
   // Need to replace all the characters not allowed in a symbol ...
   std::string main_dictname_copy(main_dictname);
   TMetaUtils::GetCppName(main_dictname, main_dictname_copy.c_str());

   time_t t = time(0);
   (*dictSrcOut) << "//"  << std::endl
                 << "// File generated by " << argv[0] << " at " << ctime(&t) << std::endl
                 << "// Do NOT change. Changes will be lost next time file is generated" << std::endl
                 << "//" << std::endl << std::endl

                 << "#define R__DICTIONARY_FILENAME " << main_dictname << std::endl
                 << "#include \"" << inclf << "\"\n"
                 << std::endl;
#ifndef R__SOLARIS
   (*dictSrcOut) << "// Since CINT ignores the std namespace, we need to do so in this file." << std::endl
                 << "namespace std {} using namespace std;" << std::endl << std::endl;
#endif

   //---------------------------------------------------------------------------
   // Parse the linkdef or selection.xml file.
   //---------------------------------------------------------------------------

   string linkdefFilename;
   if (!il) {
      linkdefFilename = "in memory";
   } else {
      bool found = Which(interp, argv[il], linkdefFilename);
      if (!found) {
         Error(0, "%s: cannot open linkdef file %s\n", argv[0], argv[il]);
         CleanupOnExit(1);
         return 1;
      }
   }   

   SelectionRules selectionRules;
   std::string extraIncludes;

   if (requestAllSymbols) {
      selectionRules.SetDeep(true);
   } else if (!il) {
      // There is no linkdef file, we added the 'default' #pragma to 
      // interpPragmaSource.

      LinkdefReader ldefr;
      ldefr.SetIOCtorTypeCallback(AddConstructorType);
      clingArgs.push_back("-Ietc/cling/cint"); // For multiset and multimap
 
      if (!ldefr.Parse(selectionRules, interpPragmaSource, clingArgs,
                       gResourceDir.c_str())) {
         Error(0,"Parsing #pragma failed %s",linkdefFilename.c_str());
      }
      else {
         Info(0,"#pragma successfully parsed.\n");
      }

      ldefr.LoadIncludes(interp,extraIncludes);

   } else if (R__IsSelectionXml(linkdefFilename.c_str())) {

      selectionRules.SetSelectionFileType(SelectionRules::kSelectionXMLFile);

      std::ifstream file(linkdefFilename.c_str());
      if(file.is_open()){
         Info(0,"Selection XML file\n");

         XMLReader xmlr;
         if (!xmlr.Parse(file, selectionRules)) {
            Error(0,"Parsing XML file %s",linkdefFilename.c_str());
         }
         else {
            Info(0,"XML file successfully parsed\n");
         }            
         file.close();
      }
      else {
         Error(0,"XML file %s couldn't be opened!\n",linkdefFilename.c_str());
      }

   } else if (R__IsLinkdefFile(linkdefFilename.c_str())) {

      std::ifstream file(linkdefFilename.c_str());
      if(file.is_open()) {
         Info(0,"Using linkdef file: %s\n",linkdefFilename.c_str());
         file.close();
      }
      else {
         Error(0,"Linkdef file %s couldn't be opened!\n",linkdefFilename.c_str());
      }

      selectionRules.SetSelectionFileType(SelectionRules::kLinkdefFile);

      LinkdefReader ldefr;
      ldefr.SetIOCtorTypeCallback(AddConstructorType);
      clingArgs.push_back("-Ietc/cling/cint"); // For multiset and multimap 

      if (!ldefr.Parse(selectionRules, interpPragmaSource, clingArgs,
                       gResourceDir.c_str())) {
         Error(0,"Parsing Linkdef file %s",linkdefFilename.c_str());
      }
      else {
         Info(0,"Linkdef file successfully parsed.\n");
      }

      ldefr.LoadIncludes(interp,extraIncludes);
   } else {

      Error(0,"Unrecognized selection file: %s",linkdefFilename.c_str());

   }

   //---------------------------------------------------------------------------
   // Write schema evolution reelated headers and declarations
   //---------------------------------------------------------------------------
   if( !gReadRules.empty() || !gReadRawRules.empty() ) {
      (*dictSrcOut) << "#include \"TBuffer.h\"" << std::endl;
      (*dictSrcOut) << "#include \"TVirtualObject.h\"" << std::endl;
      (*dictSrcOut) << "#include <vector>" << std::endl;
      (*dictSrcOut) << "#include \"TSchemaHelper.h\"" << std::endl << std::endl;

      std::list<std::string>           includes;
      std::list<std::string>::iterator it;
      GetRuleIncludes( includes );
      for( it = includes.begin(); it != includes.end(); ++it )
         (*dictSrcOut) << "#include <" << *it << ">" << std::endl;
      (*dictSrcOut) << std::endl;
   }

   //---------------------------------------------------------------------------
   // Write all the necessary #include
   //---------------------------------------------------------------------------
   (*dictSrcOut) << "// Header files passed as explicit arguments\n";
   (*dictSrcOut) << includeForSource;
   (*dictSrcOut) << "\n// Header files passed via #pragma extra_include\n";
   (*dictSrcOut) << extraIncludes << endl;

   selectionRules.SearchNames(interp);

   clang::CompilerInstance* CI = interp.getCI();
   
   RScanner scan(selectionRules,interp,normCtxt);
   // If needed initialize the autoloading hook
   if (gLiblistPrefix.length()) {
      LoadLibraryMap();
      scan.SetRecordDeclCallback(RecordDeclCallback);
   }
   scan.Scan(CI->getASTContext());

   bool has_input_error = false;

// SELECTION LOOP
   // Check for error in the class layout before doing anything else.
   RScanner::ClassColl_t::const_iterator iter = scan.fSelectedClasses.begin();
   RScanner::ClassColl_t::const_iterator end = scan.fSelectedClasses.end();
   for( ; iter != end; ++iter) 
   {
      if (ClassInfo__HasMethod(*iter,"Streamer")) {
         if (iter->RequestNoInputOperator()) {
            int version = GetClassVersion(*iter);
            if (version!=0) {
               // Only Check for input operator is the object is I/O has
               // been requested.
               has_input_error |= CheckInputOperator(*iter);
            }
         }
      }
      has_input_error |= !CheckClassDef(*iter);
   }

   if (has_input_error) {
      // Be a little bit makefile friendly and remove the dictionary in case of error.
      // We could add an option -k to keep the file even in case of error.
      CleanupOnExit(1);
      exit(1);
   }

   //
   // We will loop over all the classes several times.
   // In order we will call
   //
   //     WriteClassInit (code to create the TGenericClassInfo)
   //     check for constructor and operator input
   //     WriteClassFunctions (declared in ClassDef)
   //     WriteClassCode (Streamer,ShowMembers,Auxiliary functions)
   //
   
   // The order of addition to the list of constructor type
   // is significant.  The list is sorted by with the highest
   // priority first.
   AddConstructorType("TRootIOCtor");
   AddConstructorType("");
   
   //
   // Loop over all classes and create Streamer() & Showmembers() methods
   //
   
   // SELECTION LOOP
   RScanner::NamespaceColl_t::const_iterator ns_iter = scan.fSelectedNamespaces.begin();
   RScanner::NamespaceColl_t::const_iterator ns_end = scan.fSelectedNamespaces.end();
   for( ; ns_iter != ns_end; ++ns_iter) {
      WriteNamespaceInit(*ns_iter);         
   }
   
   iter = scan.fSelectedClasses.begin();
   end = scan.fSelectedClasses.end();
   for( ; iter != end; ++iter) 
   {
      if (!iter->GetRecordDecl()->isCompleteDefinition()) {
         Error(0,"A dictionary has been requested for %s but there is no declaration!\n",R__GetQualifiedName(* iter->GetRecordDecl()).c_str());
         continue;
      }
      if (iter->RequestOnlyTClass()) {
         // fprintf(stderr,"rootcling: Skipping class %s\n",R__GetQualifiedName(* iter->GetRecordDecl()).c_str());
         // For now delay those for later.
         continue;
      }
      
      if (clang::CXXRecordDecl* CXXRD = llvm::dyn_cast<clang::CXXRecordDecl>(const_cast<clang::RecordDecl*>(iter->GetRecordDecl())))
            R__AnnotateDecl(*CXXRD);
      const clang::CXXRecordDecl* CRD = llvm::dyn_cast<clang::CXXRecordDecl>(iter->GetRecordDecl());
      if (CRD) {
         Info(0,"Generating code for class %s\n", iter->GetNormalizedName() );
         std::string qualname( CRD->getQualifiedNameAsString() );
         if (IsStdClass(*CRD) && 0 != TClassEdit::STLKind(CRD->getName().str().c_str() /* unqualified name without template arguement */) ) {
            // coverity[fun_call_w_exception] - that's just fine.
            RStl::Instance().GenerateTClassFor( iter->GetNormalizedName(), CRD, interp, normCtxt);
         } else {
            WriteClassInit(*iter, interp, normCtxt);
         }               
      }
   }

   //
   // Write all TBuffer &operator>>(...), Class_Name(), Dictionary(), etc.
   // first to allow template specialisation to occur before template
   // instantiation (STK)
   //
   // SELECTION LOOP
   iter = scan.fSelectedClasses.begin();
   end = scan.fSelectedClasses.end();
   for( ; iter != end; ++iter) 
   {
      if (!iter->GetRecordDecl()->isCompleteDefinition()) {
         continue;
      }                       
      if (iter->RequestOnlyTClass()) {
         // For now delay those for later.
         continue;
      }
      const clang::CXXRecordDecl* cxxdecl = llvm::dyn_cast<clang::CXXRecordDecl>(iter->GetRecordDecl());
      if (cxxdecl && ClassInfo__HasMethod(*iter,"Class_Name")) {
         WriteClassFunctions(cxxdecl);
      }
   }
   
   // LINKDEF SELECTION LOOP
   // Loop to get the shadow class for the class marker 'RequestOnlyTClass' (but not the
   // STL class which is done via RStl::Instance().WriteClassInit(0);
   // and the ClassInit
   iter = scan.fSelectedClasses.begin();
   end = scan.fSelectedClasses.end();
   for( ; iter != end; ++iter) 
   {
      if (!iter->GetRecordDecl()->isCompleteDefinition()) {
         continue;
      }
      if (!iter->RequestOnlyTClass()) {
         continue;
      }
      if (!IsSTLContainer(*iter)) {
         WriteClassInit(*iter, interp, normCtxt);
      }
   }
   // Loop to write all the ClassCode
   iter = scan.fSelectedClasses.begin();
   end = scan.fSelectedClasses.end();
   for( ; iter != end; ++iter) 
   {
      if (!iter->GetRecordDecl()->isCompleteDefinition()) {
         continue;
      }
      WriteClassCode(*iter, interp, normCtxt);
   }
   
   // coverity[fun_call_w_exception] - that's just fine.
   RStl::Instance().WriteClassInit(0, interp, normCtxt);
   
   // Now we have done all our looping and thus all the possible 
   // annotation, let's write the pcms.
   if (strstr(dictname,"rootcint_") != dictname) {
      // Modules only for "regular" dictionaries, not for cintdlls
      // pcmArgs does not need any of the 'extra' include (entered via
      // #pragma) as those are needed only for compilation.
      // However CINT was essentially treating them the same as any other
      // so we may have to put them here too ... maybe.
      GenerateModule(dictname, pcmArgs, currentDirectory);
   }

   // Now that CINT is not longer there to write the header file,
   // write one and include in there a few things for backward 
   // compatibility.
   (*dictHdrOut) << "/********************************************************************\n";
   
   (*dictHdrOut) << "* " << dictheader << "\n";
   (*dictHdrOut) << "* CAUTION: DON'T CHANGE THIS FILE. THIS FILE IS AUTOMATICALLY GENERATED\n";
   (*dictHdrOut) << "*          FROM HEADER FILES LISTED IN 'DictInit::headers'.\n";
   (*dictHdrOut) << "*          CHANGE THOSE HEADER FILES AND REGENERATE THIS FILE.\n";
   (*dictHdrOut) << "********************************************************************/\n";
   (*dictHdrOut) << "#include <stddef.h>\n";
   (*dictHdrOut) << "#include <stdio.h>\n";
   (*dictHdrOut) << "#include <stdlib.h>\n";
   (*dictHdrOut) << "#include <math.h>\n";
   (*dictHdrOut) << "#include <string.h>\n";
   (*dictHdrOut) << "#define G__DICTIONARY\n";
   (*dictHdrOut) << "#include \"RConfig.h\"\n"
                 << "#include \"TClass.h\"\n"
                 << "#include \"TCintWithCling.h\"\n"
                 << "#include \"TBuffer.h\"\n"
                 << "#include \"TMemberInspector.h\"\n"
                 << "#include \"TError.h\"\n\n"
                 << "#ifndef G__ROOT\n"
                 << "#define G__ROOT\n"
                 << "#endif\n\n"
                 << "#include \"RtypesImp.h\"\n"
                 << "#include \"TIsAProxy.h\"\n"
                 << "#include \"TFileMergeInfo.h\"\n";
   (*dictSrcOut) << std::endl;
   if (gNeedCollectionProxy) {
      (*dictHdrOut) << "#include <algorithm>\n";
      (*dictHdrOut) << "\n#include \"TCollectionProxyInfo.h\"";
   }
   (*dictHdrOut) << "\n";
   
   if (gLiblistPrefix.length()) {
      string liblist_filename = gLiblistPrefix + ".out";

      ofstream outputfile( liblist_filename.c_str(), ios::out );
      if (!outputfile) {
         Error(0,"%s: Unable to open output lib file %s\n",
               argv[0], liblist_filename.c_str());
      } else {
         const size_t endStr = gLibsNeeded.find_last_not_of(" \t");
         outputfile << gLibsNeeded.substr(0, endStr+1) << endl;
         // Add explicit delimiter
         outputfile << "# Now the list of classes\n";
// SELECTION LOOP
         iter = scan.fSelectedClasses.begin();
         end = scan.fSelectedClasses.end();
         for( ; iter != end; ++iter) 
         {
            // Shouldn't it be GetLong64_Name( cl_input.GetNormalizedName() )
            // or maybe we should be normalizing to turn directly all long long into Long64_t
            outputfile << iter->GetNormalizedName() << endl;            
         }
      }
   }

   CleanupOnExit(0);
   return 0;
}
