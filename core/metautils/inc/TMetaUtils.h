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

#include "RConversionRuleParser.h"

// #include "llvm/Attr.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <stdlib.h>

namespace clang {
   class ASTContext;
   class CompilerInstance;
   class CXXBaseSpecifier;
   class CXXRecordDecl;
   class Decl;
   class FieldDecl;
   class FunctionDecl;
   class Module;
   class NamedDecl;
   class QualType;
   class RecordDecl;
   class SourceLocation;
   class Type;
   class TypedefNameDecl;
   class Attr;
}

namespace cling {
   class Interpreter;
   class LookupHelper;
}

#include "cling/Utils/AST.h"

// For TClassEdit::ESTLType and for TClassEdit::TInterpreterLookupHelper
#include "TClassEdit.h"

#ifndef ROOT_Varargs
#include "Varargs.h"
#endif

namespace ROOT {
   namespace TMetaUtils {

      // Convention used to separate name/value of properties in the ast annotations
      static const std::string PropertyNameValSeparator("@@@");

      int extractAttrString(clang::Attr* attribute, std::string& attrString);
      int extractPropertyNameValFromString(const std::string attributeStr,std::string& attrName, std::string& attrValue);
      int extractPropertyNameVal(clang::Attr* attribute, std::string& attrName, std::string& attrValue);
      bool IsInt(const std::string& s);

      class TNormalizedCtxt {
         typedef llvm::SmallSet<const clang::Type*, 4> TypesCont_t;
         typedef cling::utils::Transform::Config Config_t;
      private:
         Config_t    fConfig;
         TypesCont_t fTypeWithAlternative;
      public:
         TNormalizedCtxt(const cling::LookupHelper &lh);

         const Config_t    &GetConfig() const { return fConfig; }
         const TypesCont_t &GetTypeToSkip() const { return fConfig.m_toSkip; }
         const TypesCont_t &GetTypeWithAlternative() const { return fTypeWithAlternative; }
      };

      class TClingLookupHelper : public TClassEdit::TInterpreterLookupHelper {
      private:
         cling::Interpreter *fInterpreter;
         TNormalizedCtxt    *fNormalizedCtxt;
      public:
         TClingLookupHelper(cling::Interpreter &interpreter, ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
         virtual ~TClingLookupHelper() { /* we're not owner */ }
         virtual void GetPartiallyDesugaredName(std::string &nameLong);
         virtual bool IsAlreadyPartiallyDesugaredName(const std::string &nondef, const std::string &nameLong);
         virtual bool IsDeclaredScope(const std::string &base);
         virtual bool GetPartiallyDesugaredNameWithScopeHandling(const std::string &tname, std::string &result);
      };

      // Add default template parameters.
      clang::QualType AddDefaultParameters(clang::QualType instanceType, const cling::Interpreter &interpret, const TNormalizedCtxt &normCtxt);

      // Get the array index information for a data member.
      enum DataMemberInfo__ValidArrayIndex_error_code { VALID, NOT_INT, NOT_DEF, IS_PRIVATE, UNKNOWN };
      const char* DataMemberInfo__ValidArrayIndex(const clang::FieldDecl &m, int *errnum = 0, const char **errstr = 0);

      // Return the ROOT include directory
      std::string GetROOTIncludeDir(bool rootbuild);


      // These are the methods which were moved from rootcling to TMetaUtils
      class AnnotatedRecordDecl {
      private:
         long fRuleIndex;
         const clang::RecordDecl* fDecl;
         std::string fRequestedName;
         std::string fNormalizedName;
         bool fRequestStreamerInfo;
         bool fRequestNoStreamer;
         bool fRequestNoInputOperator;
         bool fRequestOnlyTClass;
         int  fRequestedVersionNumber;

      public:
         enum ERootFlag {
            kNoStreamer      = 0x01,
            kNoInputOperator = 0x02,
            kUseByteCount    = 0x04,
            kStreamerInfo    = 0x04,
            kHasVersion      = 0x08
         };

         AnnotatedRecordDecl(long index, const clang::RecordDecl *decl, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
         AnnotatedRecordDecl(long index, const clang::RecordDecl *decl, const char *requestName, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
         AnnotatedRecordDecl(long index, const clang::Type *requestedType, const clang::RecordDecl *decl, const char *requestedName, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
        ~AnnotatedRecordDecl() {
            // Nothing to do we do not own the pointer;
         }


         long GetRuleIndex() const { return fRuleIndex; }

         const char *GetRequestedName() const { return fRequestedName.c_str(); }
         const char *GetNormalizedName() const { return fNormalizedName.c_str(); }
         bool HasClassVersion() const { return fRequestedVersionNumber >=0 ; }
         bool RequestStreamerInfo() const {
            // Equivalent to CINT's cl.RootFlag() & G__USEBYTECOUNT
            return fRequestStreamerInfo;
         }
         bool RequestNoInputOperator() const { return fRequestNoInputOperator; }
         bool RequestNoStreamer() const { return fRequestNoStreamer; }
         bool RequestOnlyTClass() const { return fRequestOnlyTClass; }
         int  RequestedVersionNumber() const { return fRequestedVersionNumber; }
         int  RootFlag() const {
            // Return the request (streamerInfo, has_version, etc.) combined in a single
            // int.  See RScanner::AnnotatedRecordDecl::ERootFlag.
            int result = 0;
            if (fRequestNoStreamer) result = kNoStreamer;
            if (fRequestNoInputOperator) result |= kNoInputOperator;
            if (fRequestStreamerInfo) result |= kStreamerInfo;
            if (fRequestedVersionNumber > -1) result |= kHasVersion;
            return result;
         }
         const clang::RecordDecl* GetRecordDecl() const { return fDecl; }

         operator clang::RecordDecl const *() const {
            return fDecl;
         }

         bool operator<(const AnnotatedRecordDecl& right) const
         {
            return fRuleIndex < right.fRuleIndex;
         }

         struct CompareByName {
            bool operator() (const AnnotatedRecordDecl& right, const AnnotatedRecordDecl& left)
            {
               return left.fNormalizedName < right.fNormalizedName;
            }
         };
      };

      class RConstructorType
      {
         private:
            std::string           fArgTypeName;
            const clang::CXXRecordDecl *fArgType;

         public:
            RConstructorType(const char *type_of_arg, const cling::Interpreter&);

            const char *GetName();
            const clang::CXXRecordDecl *GetType();
      };

      bool CheckConstructor(const clang::CXXRecordDecl*, ROOT::TMetaUtils::RConstructorType&);
      bool ClassInfo__HasMethod(const clang::RecordDecl *cl, char const*);
      void CreateNameTypeMap(clang::CXXRecordDecl const&, std::map<std::string, ROOT::TSchemaType, std::less<std::string>, std::allocator<std::pair<std::string const, ROOT::TSchemaType> > >&);

      int ElementStreamer(std::ostream& finalString, const clang::NamedDecl &forcontext, const clang::QualType &qti, const char *R__t,int rwmode, const cling::Interpreter &gInterp, const char *tcl=0);
      bool R__IsBase(const clang::CXXRecordDecl *cl, const clang::CXXRecordDecl *base, const clang::CXXRecordDecl *context = 0);
      bool R__IsBase(const clang::FieldDecl &m, const char* basename, const cling::Interpreter &gInterp);

      bool HasCustomOperatorNewArrayPlacement(clang::RecordDecl const&, const cling::Interpreter &interp);
      bool HasCustomOperatorNewPlacement(char const*, clang::RecordDecl const&, const cling::Interpreter&);
      bool HasCustomOperatorNewPlacement(clang::RecordDecl const&, const cling::Interpreter&);
      bool HasDirectoryAutoAdd(clang::CXXRecordDecl const*, const cling::Interpreter&);
      bool HasIOConstructor(clang::CXXRecordDecl const*, std::string*, const cling::Interpreter&);
      bool HasNewMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);
      bool HasOldMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);
      bool hasOpaqueTypedef(clang::QualType instanceType, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      bool hasOpaqueTypedef(const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);

      bool HasResetAfterMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);
      bool NeedDestructor(clang::CXXRecordDecl const*);
      bool NeedTemplateKeyword(clang::CXXRecordDecl const*);
      bool R__CheckPublicFuncWithProto(clang::CXXRecordDecl const*, char const*, char const*, const cling::Interpreter&);

      long R__GetLineNumber(clang::Decl const*);

      bool R__GetNameWithinNamespace(std::string&, std::string&, std::string&, clang::CXXRecordDecl const*);

      void R__GetQualifiedName(std::string &qual_name, const clang::QualType &type, const clang::NamedDecl &forcontext);
      void R__GetQualifiedName(std::string &qual_name, const clang::NamedDecl &cl);
      void R__GetQualifiedName(std::string &qual_name, const ROOT::TMetaUtils::AnnotatedRecordDecl &annotated);
      std::string R__GetQualifiedName(const clang::QualType &type, const clang::NamedDecl &forcontext);
      std::string R__GetQualifiedName(const clang::Type &type, const clang::NamedDecl &forcontext);
      std::string R__GetQualifiedName(const clang::NamedDecl &cl);
      std::string R__GetQualifiedName(const clang::CXXBaseSpecifier &base);
      std::string R__GetQualifiedName(const ROOT::TMetaUtils::AnnotatedRecordDecl &annotated);

      int WriteNamespaceHeader(std::ostream&, const clang::RecordDecl *);
      void WritePointersSTL(const AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);

      // void WriteAutoStreamer(const AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      // void WriteStreamer(const AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);

      void WritePointersSTL(const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      int GetClassVersion(const clang::RecordDecl *cl);
      int IsSTLContainer(const ROOT::TMetaUtils::AnnotatedRecordDecl &annotated);
      TClassEdit::ESTLType IsSTLContainer(const clang::FieldDecl &m);
      int IsSTLContainer(const clang::CXXBaseSpecifier &base);
      const char *ShortTypeName(const char *typeDesc);
      std::string ShortTypeName(const clang::FieldDecl &m);
      bool IsStreamableObject(const clang::FieldDecl &m);
      clang::RecordDecl *R__GetUnderlyingRecordDecl(clang::QualType type);

      std::string R__TrueName(const clang::FieldDecl &m);

      const clang::CXXRecordDecl *R__ScopeSearch(const char *name, const cling::Interpreter &gInterp, const clang::Type** resultType = 0);
      void AddConstructorType(const char *arg, const cling::Interpreter &interp);
      void WriteAuxFunctions(std::ostream& finalString, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      //void WriteShowMembers(const AnnotatedRecordDecl &cl, bool outside = false);
      void WriteShowMembers(std::ostream& finalString, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt, bool outside = false);
      bool NeedExternalShowMember(const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl,  const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);

      const clang::FunctionDecl *R__GetFuncWithProto(const clang::Decl* cinfo, const char *method, const char *proto, const cling::Interpreter &gInterp);

      typedef void (*CallWriteStreamer_t)(const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt, bool isAutoStreamer);

      void WriteClassCode(CallWriteStreamer_t WriteStreamerFunc, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt, std::ostream& finalString);
      void WriteEverything(CallWriteStreamer_t WriteStreamerFunc, std::ostream& finalString, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      void WriteClassInit(std::ostream& finalString, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt, bool& needCollectionProxy);

      bool HasCustomStreamerMemberFunction(const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl* clxx, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt);
      void WriteBodyShowMembers(std::ostream& finalString, const ROOT::TMetaUtils::AnnotatedRecordDecl &cl, const clang::CXXRecordDecl *decl, const cling::Interpreter &interp, const ROOT::TMetaUtils::TNormalizedCtxt &normCtxt, bool outside);
      const int kInfo     =      0;
      const int kNote     =    500;
      const int kWarning  =   1000;
      const int kError    =   2000;
      const int kSysError =   3000;
      const int kFatal    =   4000;
      const int kMaxLen   =   1024;

      extern int gErrorIgnoreLevel;

      void LevelPrint(bool prefix, int level, const char *location, const char *fmt, va_list ap);
      void Error(const char *location, const char *va_(fmt), ...);
      void SysError(const char *location, const char *va_(fmt), ...);
      void Info(const char *location, const char *va_(fmt), ...);
      void Warning(const char *location, const char *va_(fmt), ...);
      void Fatal(const char *location, const char *va_(fmt), ...);


      // Return the header file to be included to declare the Decl
      llvm::StringRef GetFileName(const clang::Decl *decl);

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

      // Return the type with all parts fully qualified (most typedefs),
      // including template arguments.
      clang::QualType GetFullyQualifiedType(const clang::QualType &type, const cling::Interpreter &interpreter);

      // Return the type with all parts fully qualified (most typedefs),
      // including template arguments, appended to name.
      void GetFullyQualifiedTypeName(std::string &name, const clang::QualType &type, const cling::Interpreter &interpreter);

      // Return the type name normalized for ROOT,
      // keeping only the ROOT opaque typedef (Double32_t, etc.) and
      // adding default template argument for all types except the STL collections
      // where we remove the default template argument if any.
      void GetNormalizedName(std::string &norm_name, const clang::QualType &type, const cling::Interpreter &interpreter, const TNormalizedCtxt &normCtxt);

      // Returns the comment (// striped away), annotating declaration in a meaningful
      // for ROOT IO way.
      // Takes optional out parameter clang::SourceLocation returning the source
      // location of the comment.
      //
      // CXXMethodDecls, FieldDecls and TagDecls are annotated.
      // CXXMethodDecls declarations and FieldDecls are annotated as follows:
      // Eg. void f(); // comment1
      //     int member; // comment2
      // Inline definitions of CXXMethodDecls - after the closing ) and before {. Eg:
      // void f() // comment3
      // {...}
      // TagDecls are annotated in the end of the ClassDef macro. Eg.
      // class MyClass {
      // ...
      // ClassDef(MyClass, 1) // comment4
      //
      llvm::StringRef GetComment(const clang::Decl &decl, clang::SourceLocation *loc = 0);

      // Return the class comment:
      // class MyClass {
      // ...
      // ClassDef(MyClass, 1) // class comment
      //
      llvm::StringRef GetClassComment(const clang::CXXRecordDecl &decl, clang::SourceLocation *loc, const cling::Interpreter &interpreter);

      // Return the base/underlying type of a chain of array or pointers type.
      const clang::Type *GetUnderlyingType(clang::QualType type);

      // Scans the redeclaration chain for a definition of the redeclarable which
      // is annotated.
      //
      // returns 0 if no annotation was found.
      //
      template<typename T>
      const T* GetAnnotatedRedeclarable(const T* Redecl) {
         if (!Redecl)
            return 0;

         Redecl = Redecl->getMostRecentDecl();
         while (Redecl && !(Redecl->hasAttrs() && Redecl->isThisDeclarationADefinition()))
            Redecl = Redecl->getPreviousDecl();

         return Redecl;
      }

      // Overload the template for typedefs, because they don't contain
      // isThisDeclarationADefinition method. (Use inline to avoid violating ODR)
      const clang::TypedefNameDecl* GetAnnotatedRedeclarable(const clang::TypedefNameDecl* TND);

      // Return true if the decl is part of the std namespace.
      bool IsStdClass(const clang::RecordDecl &cl);

      // Return which kind of STL container the decl is, if any.
      TClassEdit::ESTLType IsSTLCont(const clang::RecordDecl &cl);

      // Check if 'input' or any of its template parameter was substituted when
      // instantiating the class template instance and replace it with the
      // partially sugared type we have from 'instance'.
      clang::QualType ReSubstTemplateArg(clang::QualType input, const clang::Type *instance);

      // Kind of stl container
      TClassEdit::ESTLType STLKind(const llvm::StringRef type);

   } // namespace TMetaUtils


} // namespace ROOT

#endif // ROOT_TMetaUtils
