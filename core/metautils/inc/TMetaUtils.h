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
   class TagDecl;
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

// Forward Declarations --------------------------------------------------------
class AnnotatedRecordDecl;
      
// Constants, typedefs and Enums -----------------------------------------------

// Convention used to separate name/value of properties in the ast annotations
static const std::string PropertyNameValSeparator("@@@");

extern int gErrorIgnoreLevel;

// Get the array index information for a data member.
enum DataMemberInfo__ValidArrayIndex_error_code { VALID, NOT_INT, NOT_DEF, IS_PRIVATE, UNKNOWN };

typedef void (*CallWriteStreamer_t)(const AnnotatedRecordDecl &cl,
                                    const cling::Interpreter &interp,
                                    const TNormalizedCtxt &normCtxt,
                                    bool isAutoStreamer);

const int kInfo     =      0;
const int kNote     =    500;
const int kWarning  =   1000;
const int kError    =   2000;
const int kSysError =   3000;
const int kFatal    =   4000;
const int kMaxLen   =   1024;

// Classes ---------------------------------------------------------------------      

//______________________________________________________________________________
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

//______________________________________________________________________________
class TClingLookupHelper : public TClassEdit::TInterpreterLookupHelper {
private:
   cling::Interpreter *fInterpreter;
   TNormalizedCtxt    *fNormalizedCtxt;
public:
   TClingLookupHelper(cling::Interpreter &interpreter, TNormalizedCtxt &normCtxt);
   virtual ~TClingLookupHelper() { /* we're not owner */ }
   virtual void GetPartiallyDesugaredName(std::string &nameLong);
   virtual bool IsAlreadyPartiallyDesugaredName(const std::string &nondef, const std::string &nameLong);
   virtual bool IsDeclaredScope(const std::string &base);
   virtual bool GetPartiallyDesugaredNameWithScopeHandling(const std::string &tname, std::string &result);
};

//______________________________________________________________________________
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

   AnnotatedRecordDecl(long index, const clang::RecordDecl *decl, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const TNormalizedCtxt &normCtxt);
   AnnotatedRecordDecl(long index, const clang::RecordDecl *decl, const char *requestName, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const TNormalizedCtxt &normCtxt);
   AnnotatedRecordDecl(long index, const clang::Type *requestedType, const clang::RecordDecl *decl, const char *requestedName, bool rStreamerInfo, bool rNoStreamer, bool rRequestNoInputOperator, bool rRequestOnlyTClass, int rRequestedVersionNumber, const cling::Interpreter &interpret, const TNormalizedCtxt &normCtxt);
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

//______________________________________________________________________________
class RConstructorType {
private:
   std::string           fArgTypeName;
   const clang::CXXRecordDecl *fArgType;

public:
   RConstructorType(const char *type_of_arg, const cling::Interpreter&);

   const char *GetName();
   const clang::CXXRecordDecl *GetType();
};
      
// Functions -------------------------------------------------------------------

//______________________________________________________________________________
int extractAttrString(clang::Attr* attribute, std::string& attrString);

//______________________________________________________________________________
int extractPropertyNameValFromString(const std::string attributeStr,std::string& attrName, std::string& attrValue);

//______________________________________________________________________________
int extractPropertyNameVal(clang::Attr* attribute, std::string& attrName, std::string& attrValue);

//______________________________________________________________________________
bool IsInt(const std::string& s);

//______________________________________________________________________________
// Add default template parameters.
clang::QualType AddDefaultParameters(clang::QualType instanceType,
                                     const cling::Interpreter &interpret,
                                     const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
const char* DataMemberInfo__ValidArrayIndex(const clang::FieldDecl &m, int *errnum = 0, const char **errstr = 0);

//______________________________________________________________________________
// Return the ROOT include directory
std::string GetROOTIncludeDir(bool rootbuild);

//______________________________________________________________________________
bool CheckConstructor(const clang::CXXRecordDecl*, RConstructorType&);

//______________________________________________________________________________
bool ClassInfo__HasMethod(const clang::RecordDecl *cl, char const*);

//______________________________________________________________________________
void CreateNameTypeMap(clang::CXXRecordDecl const&, std::map<std::string, ROOT::TSchemaType, std::less<std::string>, std::allocator<std::pair<std::string const, ROOT::TSchemaType> > >&);

//______________________________________________________________________________
int ElementStreamer(std::ostream& finalString,
                    const clang::NamedDecl &forcontext,
                    const clang::QualType &qti,
                    const char *t,
                    int rwmode,
                    const cling::Interpreter &gInterp,
                    const char *tcl=0);

//______________________________________________________________________________
bool IsBase(const clang::CXXRecordDecl *cl, const clang::CXXRecordDecl *base, const clang::CXXRecordDecl *context = 0);

//______________________________________________________________________________
bool IsBase(const clang::FieldDecl &m, const char* basename, const cling::Interpreter &gInterp);

//______________________________________________________________________________
bool HasCustomOperatorNewArrayPlacement(clang::RecordDecl const&, const cling::Interpreter &interp);

//______________________________________________________________________________
bool HasCustomOperatorNewPlacement(char const*, clang::RecordDecl const&, const cling::Interpreter&);

//______________________________________________________________________________
bool HasCustomOperatorNewPlacement(clang::RecordDecl const&, const cling::Interpreter&);

//______________________________________________________________________________
bool HasDirectoryAutoAdd(clang::CXXRecordDecl const*, const cling::Interpreter&);

//______________________________________________________________________________
bool HasIOConstructor(clang::CXXRecordDecl const*, std::string*, const cling::Interpreter&);

//______________________________________________________________________________
bool HasNewMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);

//______________________________________________________________________________
bool HasOldMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);

//______________________________________________________________________________
bool hasOpaqueTypedef(clang::QualType instanceType, const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
bool hasOpaqueTypedef(const AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
bool HasResetAfterMerge(clang::CXXRecordDecl const*, const cling::Interpreter&);

//______________________________________________________________________________
bool NeedDestructor(clang::CXXRecordDecl const*);

//______________________________________________________________________________
bool NeedTemplateKeyword(clang::CXXRecordDecl const*);

//______________________________________________________________________________
bool CheckPublicFuncWithProto(clang::CXXRecordDecl const*, char const*, char const*, const cling::Interpreter&);

//______________________________________________________________________________
long GetLineNumber(clang::Decl const*);

//______________________________________________________________________________
bool GetNameWithinNamespace(std::string&, std::string&, std::string&, clang::CXXRecordDecl const*);

//______________________________________________________________________________
void GetQualifiedName(std::string &qual_name, const clang::QualType &type, const clang::NamedDecl &forcontext);

//----
std::string GetQualifiedName(const clang::QualType &type, const clang::NamedDecl &forcontext);

//______________________________________________________________________________
void GetQualifiedName(std::string &qual_name, const clang::Type &type, const clang::NamedDecl &forcontext);

//----
std::string GetQualifiedName(const clang::Type &type, const clang::NamedDecl &forcontext);

//______________________________________________________________________________
void GetQualifiedName(std::string &qual_name, const clang::NamespaceDecl &nsd);

//----
std::string GetQualifiedName(const clang::NamespaceDecl &nsd);

//______________________________________________________________________________
void GetQualifiedName(std::string &qual_name, const AnnotatedRecordDecl &annotated);

//----
std::string GetQualifiedName(const AnnotatedRecordDecl &annotated);

//______________________________________________________________________________
void GetQualifiedName(std::string &qual_name, const clang::RecordDecl &recordDecl);

//----
std::string GetQualifiedName(const clang::RecordDecl &recordDecl);

//______________________________________________________________________________
int WriteNamespaceHeader(std::ostream&, const clang::RecordDecl *);

//______________________________________________________________________________
void WritePointersSTL(const AnnotatedRecordDecl &cl, const cling::Interpreter &interp, const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
int GetClassVersion(const clang::RecordDecl *cl);

//______________________________________________________________________________
int IsSTLContainer(const AnnotatedRecordDecl &annotated);

//______________________________________________________________________________
TClassEdit::ESTLType IsSTLContainer(const clang::FieldDecl &m);

//______________________________________________________________________________
int IsSTLContainer(const clang::CXXBaseSpecifier &base);

//______________________________________________________________________________
const char *ShortTypeName(const char *typeDesc);

//______________________________________________________________________________
std::string ShortTypeName(const clang::FieldDecl &m);

//______________________________________________________________________________
bool IsStreamableObject(const clang::FieldDecl &m);

//______________________________________________________________________________
clang::RecordDecl *GetUnderlyingRecordDecl(clang::QualType type);

//______________________________________________________________________________
std::string TrueName(const clang::FieldDecl &m);

//______________________________________________________________________________
const clang::CXXRecordDecl *ScopeSearch(const char *name, const cling::Interpreter &gInterp, const clang::Type** resultType = 0);

//______________________________________________________________________________
void AddConstructorType(const char *arg, const cling::Interpreter &interp);

//______________________________________________________________________________
void WriteAuxFunctions(std::ostream& finalString,
                       const AnnotatedRecordDecl &cl,
                       const clang::CXXRecordDecl *decl,
                       const cling::Interpreter &interp,
                       const TNormalizedCtxt &normCtxt);


//______________________________________________________________________________
void WriteShowMembers(std::ostream& finalString,
                      const AnnotatedRecordDecl &cl,
                      const clang::CXXRecordDecl *decl,
                      const cling::Interpreter &interp,
                      const TNormalizedCtxt &normCtxt,
                      bool outside = false);

//______________________________________________________________________________
bool NeedExternalShowMember(const AnnotatedRecordDecl &cl,
                            const clang::CXXRecordDecl *decl,
                            const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
const clang::FunctionDecl *GetFuncWithProto(const clang::Decl* cinfo,
                                               const char *method,
                                               const char *proto,
                                               const cling::Interpreter &gInterp);

//______________________________________________________________________________
void WriteClassCode(CallWriteStreamer_t WriteStreamerFunc,
                    const AnnotatedRecordDecl &cl,
                    const cling::Interpreter &interp,
                    const TNormalizedCtxt &normCtxt,
                    std::ostream& finalString,
                    bool isGenreflex);

//______________________________________________________________________________
void WriteClassInit(std::ostream& finalString,
                    const AnnotatedRecordDecl &cl,
                    const clang::CXXRecordDecl *decl,
                    const cling::Interpreter &interp,
                    const TNormalizedCtxt &normCtxt,
                    bool& needCollectionProxy);

//______________________________________________________________________________
bool HasCustomStreamerMemberFunction(const AnnotatedRecordDecl &cl,
                                     const clang::CXXRecordDecl* clxx,
                                     const cling::Interpreter &interp,
                                     const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
void WriteBodyShowMembers(std::ostream& finalString,
                          const AnnotatedRecordDecl &cl,
                          const clang::CXXRecordDecl *decl,
                          const TNormalizedCtxt &normCtxt,
                          bool outside);

//______________________________________________________________________________
// Return the header file to be included to declare the Decl
llvm::StringRef GetFileName(const clang::Decl *decl,
                            const cling::Interpreter& interp);

//______________________________________________________________________________
// Return the dictionary file name for a module
std::string GetModuleFileName(const char* moduleName);

//______________________________________________________________________________
// Declare a virtual module.map to clang. Returns Module on success.
clang::Module* declareModuleMap(clang::CompilerInstance* CI,
                                 const char* moduleFileName,
                                 const char* headers[]);

//______________________________________________________________________________
// Return the -I needed to find RuntimeUniverse.h
std::string GetInterpreterExtraIncludePath(bool rootbuild);

//______________________________________________________________________________
// Return the LLVM / clang resource directory
std::string GetLLVMResourceDir(bool rootbuild);

//______________________________________________________________________________
// Return the ROOT include directory
std::string GetROOTIncludeDir(bool rootbuild);

//______________________________________________________________________________
// Return (in the argument 'output') a mangled version of the C++ symbol/type (pass as 'input')
// that can be used in C++ as a variable name.
void GetCppName(std::string &output, const char *input);

//______________________________________________________________________________
// Return the type with all parts fully qualified (most typedefs),
// including template arguments.
clang::QualType GetFullyQualifiedType(const clang::QualType &type, const cling::Interpreter &interpreter);

//______________________________________________________________________________
// Return the type with all parts fully qualified (most typedefs),
// including template arguments, without the interpreter
clang::QualType GetFullyQualifiedType(const clang::QualType &type, const clang::ASTContext &);

//______________________________________________________________________________
// Return the type with all parts fully qualified (most typedefs),
// including template arguments, appended to name.
void GetFullyQualifiedTypeName(std::string &name, const clang::QualType &type, const cling::Interpreter &interpreter);

//______________________________________________________________________________
// Return the type with all parts fully qualified (most typedefs),
// including template arguments, appended to name, without using the interpreter
void GetFullyQualifiedTypeName(std::string &name, const clang::QualType &type, const clang::ASTContext &);

//______________________________________________________________________________
// Return the type name normalized for ROOT,
// keeping only the ROOT opaque typedef (Double32_t, etc.) and
// adding default template argument for all types except the STL collections
// where we remove the default template argument if any.
void GetNormalizedName(std::string &norm_name, const clang::QualType &type, const cling::Interpreter &interpreter, const TNormalizedCtxt &normCtxt);

//______________________________________________________________________________
// Returns comment in a meaningful way
llvm::StringRef GetComment(const clang::Decl &decl, clang::SourceLocation *loc = 0);

//______________________________________________________________________________
// Returns the comment of the ClassDef macro
llvm::StringRef GetClassComment(const clang::CXXRecordDecl &decl, clang::SourceLocation *loc, const cling::Interpreter &interpreter);

//______________________________________________________________________________
// Return the base/underlying type of a chain of array or pointers type.
const clang::Type *GetUnderlyingType(clang::QualType type);

//______________________________________________________________________________
// Scans the redeclaration chain for an annotation.
//
// returns 0 if no annotation was found.
//
template<typename T>
const T* GetAnnotatedRedeclarable(const T* Redecl) {
   if (!Redecl)
      return 0;

   Redecl = Redecl->getMostRecentDecl();
   while (Redecl && !Redecl->hasAttrs())
      Redecl = Redecl->getPreviousDecl();

   return Redecl;
}

//______________________________________________________________________________
// Overload the template for typedefs, because they don't contain
// isThisDeclarationADefinition method. (Use inline to avoid violating ODR)
const clang::TypedefNameDecl* GetAnnotatedRedeclarable(const clang::TypedefNameDecl* TND);

//______________________________________________________________________________
// Overload the template for tags, because we only check definitions.
const clang::TagDecl* GetAnnotatedRedeclarable(const clang::TagDecl* TND);

//______________________________________________________________________________
// Return true if the decl is part of the std namespace.
bool IsStdClass(const clang::RecordDecl &cl);

//______________________________________________________________________________
// Return which kind of STL container the decl is, if any.
TClassEdit::ESTLType IsSTLCont(const clang::RecordDecl &cl);

//______________________________________________________________________________
// Check if 'input' or any of its template parameter was substituted when
// instantiating the class template instance and replace it with the
// partially sugared type we have from 'instance'.
clang::QualType ReSubstTemplateArg(clang::QualType input, const clang::Type *instance);

//______________________________________________________________________________
// Kind of stl container
TClassEdit::ESTLType STLKind(const llvm::StringRef type);

//______________________________________________________________________________
// Set the toolchain and the include paths for the relocatability
void SetPathsForRelocatability(std::vector<std::string>& clingArgs);

// Functions for the printouts -------------------------------------------------

//______________________________________________________________________________
void LevelPrint(bool prefix, int level, const char *location, const char *fmt, va_list ap);

//______________________________________________________________________________
void Error(const char *location, const char *va_(fmt), ...);

//______________________________________________________________________________
void SysError(const char *location, const char *va_(fmt), ...);

//______________________________________________________________________________
void Info(const char *location, const char *va_(fmt), ...);

//______________________________________________________________________________
void Warning(const char *location, const char *va_(fmt), ...);

//______________________________________________________________________________
void Fatal(const char *location, const char *va_(fmt), ...);


   } // namespace TMetaUtils


} // namespace ROOT

#endif // ROOT_TMetaUtils
