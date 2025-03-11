//--------------------------------------------------------------------*- C++ -*-
// CLING - the C++ LLVM-based InterpreterG :)
// author:  Vassil Vassilev <vvasilev@cern.ch>
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

#include "clang/Interpreter/CppInterOp.h"

#include "Compatibility.h"

#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclAccessPair.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclarationName.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/NestedNameSpecifier.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/DiagnosticSema.h"
#include "clang/Basic/Linkage.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Sema.h"
#if CLANG_VERSION_MAJOR >= 19
#include "clang/Sema/Redeclaration.h"
#endif
#include "clang/Sema/TemplateDeduction.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_os_ostream.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>

// Stream redirect.
#ifdef _WIN32
#include <io.h>
#ifndef STDOUT_FILENO
#define STDOUT_FILENO 1
// For exec().
#include <stdio.h>
#define popen(x, y) (_popen(x, y))
#define pclose (_pclose)
#endif
#else
#include <dlfcn.h>
#include <unistd.h>
#endif // WIN32

#include <stack>

namespace Cpp {

  using namespace clang;
  using namespace llvm;
  using namespace std;

  // Flag to indicate ownership when an external interpreter instance is used.
  static bool OwningSInterpreter = true;
  static compat::Interpreter* sInterpreter = nullptr;
  // Valgrind complains about __cxa_pure_virtual called when deleting
  // llvm::SectionMemoryManager::~SectionMemoryManager as part of the dtor chain
  // of the Interpreter.
  // This might fix the issue https://reviews.llvm.org/D107087
  // FIXME: For now we just leak the Interpreter.
  struct InterpDeleter {
    ~InterpDeleter() = default;
  } Deleter;

  static compat::Interpreter& getInterp() {
    assert(sInterpreter &&
           "Interpreter instance must be set before calling this!");
    return *sInterpreter;
  }
  static clang::Sema& getSema() { return getInterp().getCI()->getSema(); }
  static clang::ASTContext& getASTContext() { return getSema().getASTContext(); }

#define DEBUG_TYPE "jitcall"
  bool JitCall::AreArgumentsValid(void* result, ArgList args,
                                  void* self) const {
    bool Valid = true;
    if (Cpp::IsConstructor(m_FD)) {
      assert(result && "Must pass the location of the created object!");
      Valid &= (bool)result;
    }
    if (Cpp::GetFunctionRequiredArgs(m_FD) > args.m_ArgSize) {
      assert(0 && "Must pass at least the minimal number of args!");
      Valid = false;
    }
    if (args.m_ArgSize) {
      assert(args.m_Args != nullptr && "Must pass an argument list!");
      Valid &= (bool)args.m_Args;
    }
    if (!Cpp::IsConstructor(m_FD) && !Cpp::IsDestructor(m_FD) &&
        Cpp::IsMethod(m_FD) && !Cpp::IsStaticMethod(m_FD)) {
      assert(self && "Must pass the pointer to object");
      Valid &= (bool)self;
    }
    const auto* FD = cast<FunctionDecl>((const Decl*)m_FD);
    if (!FD->getReturnType()->isVoidType() && !result) {
      assert(0 && "We are discarding the return type of the function!");
      Valid = false;
    }
    assert(m_Kind != kDestructorCall && "Wrong overload!");
    Valid &= m_Kind != kDestructorCall;
    return Valid;
  }

  void JitCall::ReportInvokeStart(void* result, ArgList args, void* self) const{
    std::string Name;
    llvm::raw_string_ostream OS(Name);
    auto FD = (const FunctionDecl*) m_FD;
    FD->getNameForDiagnostic(OS, FD->getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
    LLVM_DEBUG(dbgs() << "Run '" << Name
               << "', compiled at: " << (void*) m_GenericCall
               << " with result at: " << result
               << " , args at: " << args.m_Args
               << " , arg count: " << args.m_ArgSize
               << " , self at: " << self << "\n";
               );
  }

  void JitCall::ReportInvokeStart(void* object, unsigned long nary,
                                  int withFree) const {
    std::string Name;
    llvm::raw_string_ostream OS(Name);
    auto FD = (const FunctionDecl*) m_FD;
    FD->getNameForDiagnostic(OS, FD->getASTContext().getPrintingPolicy(),
                             /*Qualified=*/true);
    LLVM_DEBUG(dbgs() << "Finish '" << Name
               << "', compiled at: " << (void*) m_DestructorCall);
  }

#undef DEBUG_TYPE

  std::string GetVersion() {
    const char* const VERSION = CPPINTEROP_VERSION;
    std::string fullVersion = "CppInterOp version";
    fullVersion += VERSION;
    fullVersion += "\n (based on "
#ifdef CPPINTEROP_USE_CLING
                   "cling ";
#else
                   "clang-repl";
#endif // CPPINTEROP_USE_CLING
    return fullVersion + "[" + clang::getClangFullVersion() + "])\n";
  }

  std::string Demangle(const std::string& mangled_name) {
#if CLANG_VERSION_MAJOR > 16
#ifdef _WIN32
    std::string demangle = microsoftDemangle(mangled_name, nullptr, nullptr);
#else
    std::string demangle = itaniumDemangle(mangled_name);
#endif
#else
#ifdef _WIN32
    std::string demangle = microsoftDemangle(mangled_name.c_str(), nullptr,
                                             nullptr, nullptr, nullptr);
#else
    std::string demangle =
        itaniumDemangle(mangled_name.c_str(), nullptr, nullptr, nullptr);
#endif
#endif
    return demangle;
  }

  void EnableDebugOutput(bool value/* =true*/) {
    llvm::DebugFlag = value;
  }

  bool IsDebugOutputEnabled() {
    return llvm::DebugFlag;
  }

  bool IsAggregate(TCppScope_t scope) {
    Decl *D = static_cast<Decl*>(scope);

    // Aggregates are only arrays or tag decls.
    if (ValueDecl *ValD = dyn_cast<ValueDecl>(D))
      if (ValD->getType()->isArrayType())
        return true;

    // struct, class, union
    if (CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(D))
      return CXXRD->isAggregate();

    return false;
  }

  bool IsNamespace(TCppScope_t scope) {
    Decl *D = static_cast<Decl*>(scope);
    return isa<NamespaceDecl>(D);
  }

  bool IsClass(TCppScope_t scope) {
    Decl *D = static_cast<Decl*>(scope);
    return isa<CXXRecordDecl>(D);
  }

  bool IsFunction(TCppScope_t scope) {
    Decl* D = static_cast<Decl*>(scope);
    return isa<FunctionDecl>(D);
  }

  bool IsFunctionPointerType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->isFunctionPointerType();
  }

  bool IsClassPolymorphic(TCppScope_t klass) {
    Decl* D = static_cast<Decl*>(klass);
    if (auto* CXXRD = llvm::dyn_cast<CXXRecordDecl>(D))
      if (auto* CXXRDD = CXXRD->getDefinition())
        return CXXRDD->isPolymorphic();
    return false;
  }

  static SourceLocation GetValidSLoc(Sema& semaRef) {
    auto& SM = semaRef.getSourceManager();
    return SM.getLocForStartOfFile(SM.getMainFileID());
  }

  // See TClingClassInfo::IsLoaded
  bool IsComplete(TCppScope_t scope) {
    if (!scope)
      return false;

    Decl *D = static_cast<Decl*>(scope);

    if (isa<ClassTemplateSpecializationDecl>(D)) {
      QualType QT = QualType::getFromOpaquePtr(GetTypeFromScope(scope));
      clang::Sema &S = getSema();
      SourceLocation fakeLoc = GetValidSLoc(S);
#ifdef CPPINTEROP_USE_CLING
      cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif // CPPINTEROP_USE_CLING
      return S.isCompleteType(fakeLoc, QT);
    }

    if (auto *CXXRD = dyn_cast<CXXRecordDecl>(D))
      return CXXRD->hasDefinition();
    else if (auto *TD = dyn_cast<TagDecl>(D))
      return TD->getDefinition();

    // Everything else is considered complete.
    return true;
  }

  size_t SizeOf(TCppScope_t scope) {
    assert (scope);
    if (!IsComplete(scope))
      return 0;

    if (auto *RD = dyn_cast<RecordDecl>(static_cast<Decl*>(scope))) {
      ASTContext &Context = RD->getASTContext();
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(RD);
      return Layout.getSize().getQuantity();
    }

    return 0;
  }

  bool IsBuiltin(TCppType_t type) {
    QualType Ty = QualType::getFromOpaquePtr(type);
    if (Ty->isBuiltinType() || Ty->isAnyComplexType())
      return true;
    // FIXME: Figure out how to avoid the string comparison.
    return llvm::StringRef(Ty.getAsString()).contains("complex");
  }

  bool IsTemplate(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    return llvm::isa_and_nonnull<clang::TemplateDecl>(D);
  }

  bool IsTemplateSpecialization(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    return llvm::isa_and_nonnull<clang::ClassTemplateSpecializationDecl>(D);
  }

  bool IsTypedefed(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    return llvm::isa_and_nonnull<clang::TypedefNameDecl>(D);
  }

  bool IsAbstract(TCppType_t klass) {
    auto *D = (clang::Decl *)klass;
    if (auto *CXXRD = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(D))
      return CXXRD->isAbstract();

    return false;
  }

  bool IsEnumScope(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    return llvm::isa_and_nonnull<clang::EnumDecl>(D);
  }

  bool IsEnumConstant(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    return llvm::isa_and_nonnull<clang::EnumConstantDecl>(D);
  }

  bool IsEnumType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->isEnumeralType();
  }

  static bool isSmartPointer(const RecordType* RT) {
    auto IsUseCountPresent = [](const RecordDecl *Record) {
      ASTContext &C = Record->getASTContext();
      return !Record->lookup(&C.Idents.get("use_count")).empty();
    };
    auto IsOverloadedOperatorPresent = [](const RecordDecl *Record,
                                          OverloadedOperatorKind Op) {
      ASTContext &C = Record->getASTContext();
      DeclContextLookupResult Result =
          Record->lookup(C.DeclarationNames.getCXXOperatorName(Op));
      return !Result.empty();
    };

    const RecordDecl *Record = RT->getDecl();
    if (IsUseCountPresent(Record))
      return true;

    bool foundStarOperator = IsOverloadedOperatorPresent(Record, OO_Star);
    bool foundArrowOperator = IsOverloadedOperatorPresent(Record, OO_Arrow);
    if (foundStarOperator && foundArrowOperator)
      return true;

    const CXXRecordDecl *CXXRecord = dyn_cast<CXXRecordDecl>(Record);
    if (!CXXRecord)
      return false;

    auto FindOverloadedOperators = [&](const CXXRecordDecl *Base) {
      // If we find use_count, we are done.
      if (IsUseCountPresent(Base))
        return false; // success.
      if (!foundStarOperator)
        foundStarOperator = IsOverloadedOperatorPresent(Base, OO_Star);
      if (!foundArrowOperator)
        foundArrowOperator = IsOverloadedOperatorPresent(Base, OO_Arrow);
      if (foundStarOperator && foundArrowOperator)
        return false; // success.
      return true;
    };

    return !CXXRecord->forallBases(FindOverloadedOperators);
  }

  bool IsSmartPtrType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    if (const RecordType *RT = QT->getAs<RecordType>()) {
      // Add quick checks for the std smart prts to cover most of the cases.
      std::string typeString = GetTypeAsString(type);
      llvm::StringRef tsRef(typeString);
      if (tsRef.starts_with("std::unique_ptr") ||
          tsRef.starts_with("std::shared_ptr") ||
          tsRef.starts_with("std::weak_ptr"))
        return true;
      return isSmartPointer(RT);
    }
    return false;
  }

  TCppType_t GetIntegerTypeFromEnumScope(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    if (auto *ED = llvm::dyn_cast_or_null<clang::EnumDecl>(D)) {
      return ED->getIntegerType().getAsOpaquePtr();
    }

    return 0;
  }

  TCppType_t GetIntegerTypeFromEnumType(TCppType_t enum_type) {
    if (!enum_type)
      return nullptr;

    QualType QT = QualType::getFromOpaquePtr(enum_type);
    if (auto *ET = QT->getAs<EnumType>())
      return ET->getDecl()->getIntegerType().getAsOpaquePtr();

    return nullptr;
  }

  std::vector<TCppScope_t> GetEnumConstants(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;

    if (auto *ED = llvm::dyn_cast_or_null<clang::EnumDecl>(D)) {
      std::vector<TCppScope_t> enum_constants;
      for (auto *ECD : ED->enumerators()) {
        enum_constants.push_back((TCppScope_t) ECD);
      }

      return enum_constants;
    }

    return {};
  }

  TCppType_t GetEnumConstantType(TCppScope_t handle) {
    if (!handle)
      return nullptr;

    auto *D = (clang::Decl *)handle;
    if (auto *ECD = llvm::dyn_cast<clang::EnumConstantDecl>(D))
      return ECD->getType().getAsOpaquePtr();

    return 0;
  }

  TCppIndex_t GetEnumConstantValue(TCppScope_t handle) {
    auto *D = (clang::Decl *)handle;
    if (auto *ECD = llvm::dyn_cast_or_null<clang::EnumConstantDecl>(D)) {
      const llvm::APSInt& Val = ECD->getInitVal();
      return Val.getExtValue();
    }
    return 0;
  }

  size_t GetSizeOfType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    if (const TagType *TT = QT->getAs<TagType>())
      return SizeOf(TT->getDecl());

    // FIXME: Can we get the size of a non-tag type?
    auto TI = getSema().getASTContext().getTypeInfo(QT);
    size_t TypeSize = TI.Width;
    return TypeSize/8;
  }

  bool IsVariable(TCppScope_t scope) {
    auto *D = (clang::Decl *)scope;
    return llvm::isa_and_nonnull<clang::VarDecl>(D);
  }

  std::string GetName(TCppType_t klass) {
    auto *D = (clang::NamedDecl *) klass;

    if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
      return "";
    }

    if (auto *ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
      return ND->getNameAsString();
    }

    return "<unnamed>";
  }

  std::string GetCompleteName(TCppType_t klass)
  {
    auto &C = getSema().getASTContext();
    auto *D = (Decl *) klass;

    if (auto *ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
      if (auto *TD = llvm::dyn_cast<TagDecl>(ND)) {
        std::string type_name;
        QualType QT = C.getTagDeclType(TD);
        PrintingPolicy Policy = C.getPrintingPolicy();
        Policy.SuppressUnwrittenScope = true;
        Policy.SuppressScope = true;
        Policy.AnonymousTagLocations = false;
        QT.getAsStringInternal(type_name, Policy);

        return type_name;
      }

      return ND->getNameAsString();
    }

    if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
      return "";
    }

    return "<unnamed>";
  }

  std::string GetQualifiedName(TCppType_t klass)
  {
    auto *D = (Decl *) klass;
    if (auto *ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
      return ND->getQualifiedNameAsString();
    }

    if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
      return "";
    }

    return "<unnamed>";
  }

  //FIXME: Figure out how to merge with GetCompleteName.
  std::string GetQualifiedCompleteName(TCppType_t klass)
  {
    auto &C = getSema().getASTContext();
    auto *D = (Decl *) klass;

    if (auto *ND = llvm::dyn_cast_or_null<NamedDecl>(D)) {
      if (auto *TD = llvm::dyn_cast<TagDecl>(ND)) {
        std::string type_name;
        QualType QT = C.getTagDeclType(TD);
        QT.getAsStringInternal(type_name, C.getPrintingPolicy());

        return type_name;
      }

      return ND->getQualifiedNameAsString();
    }

    if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
      return "";
    }

    return "<unnamed>";
  }

  std::vector<TCppScope_t> GetUsingNamespaces(TCppScope_t scope) {
    auto *D = (clang::Decl *) scope;

    if (auto *DC = llvm::dyn_cast_or_null<clang::DeclContext>(D)) {
      std::vector<TCppScope_t> namespaces;
      for (auto UD : DC->using_directives()) {
        namespaces.push_back((TCppScope_t) UD->getNominatedNamespace());
      }
      return namespaces;
    }

    return {};
  }

  TCppScope_t GetGlobalScope()
  {
    return getSema().getASTContext().getTranslationUnitDecl()->getFirstDecl();
  }

  static Decl *GetScopeFromType(QualType QT) {
    if (auto* Type = QT.getCanonicalType().getTypePtrOrNull()) {
      Type = Type->getPointeeOrArrayElementType();
      Type = Type->getUnqualifiedDesugaredType();
      if (auto *ET = llvm::dyn_cast<EnumType>(Type))
        return ET->getDecl();
      if (auto* FnType = llvm::dyn_cast<FunctionProtoType>(Type))
        Type = const_cast<clang::Type*>(FnType->getReturnType().getTypePtr());
      return Type->getAsCXXRecordDecl();
    }
    return 0;
  }

  TCppScope_t GetScopeFromType(TCppType_t type)
  {
    QualType QT = QualType::getFromOpaquePtr(type);
    return (TCppScope_t) GetScopeFromType(QT);
  }

  static clang::Decl* GetUnderlyingScope(clang::Decl * D) {
    if (auto *TND = dyn_cast_or_null<TypedefNameDecl>(D)) {
      if (auto* Scope = GetScopeFromType(TND->getUnderlyingType()))
        D = Scope;
    } else if (auto* USS = dyn_cast_or_null<UsingShadowDecl>(D)) {
      if (auto* Scope = USS->getTargetDecl())
        D = Scope;
    }

    return D;
  }

  TCppScope_t GetUnderlyingScope(TCppScope_t scope) {
    if (!scope)
      return 0;
    return GetUnderlyingScope((clang::Decl *) scope);
  }

  TCppScope_t GetScope(const std::string &name, TCppScope_t parent)
  {
    // FIXME: GetScope should be replaced by a general purpose lookup
    // and filter function. The function should be like GetNamed but
    // also take in a filter parameter which determines which results
    // to pass back
    if (name == "")
        return GetGlobalScope();

    auto *ND = (NamedDecl*)GetNamed(name, parent);

    if (!ND || ND == (NamedDecl *) -1)
      return 0;

    if (llvm::isa<NamespaceDecl>(ND)     ||
        llvm::isa<RecordDecl>(ND)        ||
        llvm::isa<ClassTemplateDecl>(ND) ||
        llvm::isa<TypedefNameDecl>(ND))
      return (TCppScope_t)(ND->getCanonicalDecl());

    return 0;
  }

  TCppScope_t GetScopeFromCompleteName(const std::string &name)
  {
    std::string delim = "::";
    size_t start = 0;
    size_t end = name.find(delim);
    TCppScope_t curr_scope = 0;
    while (end != std::string::npos)
    {
      curr_scope = GetScope(name.substr(start, end - start), curr_scope);
      start = end + delim.length();
      end = name.find(delim, start);
    }
    return GetScope(name.substr(start, end), curr_scope);
  }

  TCppScope_t GetNamed(const std::string &name,
                       TCppScope_t parent /*= nullptr*/)
  {
    clang::DeclContext *Within = 0;
    if (parent) {
      auto *D = (clang::Decl *)parent;
      D = GetUnderlyingScope(D);
      Within = llvm::dyn_cast<clang::DeclContext>(D);
    }

    auto *ND = Cpp_utils::Lookup::Named(&getSema(), name, Within);
    if (ND && ND != (clang::NamedDecl*) -1) {
      return (TCppScope_t)(ND->getCanonicalDecl());
    }

    return 0;
  }

  TCppScope_t GetParentScope(TCppScope_t scope)
  {
    auto *D = (clang::Decl *) scope;

    if (llvm::isa_and_nonnull<TranslationUnitDecl>(D)) {
      return 0;
    }
    auto *ParentDC = D->getDeclContext();

    if (!ParentDC)
      return 0;

    auto* P = clang::Decl::castFromDeclContext(ParentDC)->getCanonicalDecl();

    if (auto* TU = llvm::dyn_cast_or_null<TranslationUnitDecl>(P))
      return (TCppScope_t)TU->getFirstDecl();

    return (TCppScope_t)P;
  }

  TCppIndex_t GetNumBases(TCppScope_t klass)
  {
    auto *D = (Decl *) klass;

    if (auto *CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
      if (CXXRD->hasDefinition())
        return CXXRD->getNumBases();
    }

    return 0;
  }

  TCppScope_t GetBaseClass(TCppScope_t klass, TCppIndex_t ibase)
  {
    auto *D = (Decl *) klass;
    auto *CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D);
    if (!CXXRD || CXXRD->getNumBases() <= ibase) return 0;

    auto type = (CXXRD->bases_begin() + ibase)->getType();
    if (auto RT = type->getAs<RecordType>())
      return (TCppScope_t)RT->getDecl();

    return 0;
  }

  // FIXME: Consider dropping this interface as it seems the same as
  // IsTypeDerivedFrom.
  bool IsSubclass(TCppScope_t derived, TCppScope_t base)
  {
    if (derived == base)
      return true;

    if (!derived || !base)
      return false;

    auto *derived_D = (clang::Decl *) derived;
    auto *base_D = (clang::Decl *) base;

    if (!isa<CXXRecordDecl>(derived_D) || !isa<CXXRecordDecl>(base_D))
      return false;

    auto Derived = cast<CXXRecordDecl>(derived_D);
    auto Base = cast<CXXRecordDecl>(base_D);
    return IsTypeDerivedFrom(GetTypeFromScope(Derived),
                             GetTypeFromScope(Base));
  }

  // Copied from VTableBuilder.cpp
  // This is an internal helper function for the CppInterOp library (as evident
  // by the 'static' declaration), while the similar GetBaseClassOffset()
  // function below is exposed to library users.
  static unsigned ComputeBaseOffset(const ASTContext &Context,
                                    const CXXRecordDecl *DerivedRD,
                                    const CXXBasePath &Path) {
    CharUnits NonVirtualOffset = CharUnits::Zero();

    unsigned NonVirtualStart = 0;
    const CXXRecordDecl *VirtualBase = nullptr;

    // First, look for the virtual base class.
    for (int I = Path.size(), E = 0; I != E; --I) {
      const CXXBasePathElement &Element = Path[I - 1];

      if (Element.Base->isVirtual()) {
        NonVirtualStart = I;
        QualType VBaseType = Element.Base->getType();
        VirtualBase = VBaseType->getAsCXXRecordDecl();
        break;
      }
    }

    // Now compute the non-virtual offset.
    for (unsigned I = NonVirtualStart, E = Path.size(); I != E; ++I) {
      const CXXBasePathElement &Element = Path[I];

      // Check the base class offset.
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(Element.Class);

      const CXXRecordDecl *Base = Element.Base->getType()->getAsCXXRecordDecl();

      NonVirtualOffset += Layout.getBaseClassOffset(Base);
    }

    // FIXME: This should probably use CharUnits or something. Maybe we should
    // even change the base offsets in ASTRecordLayout to be specified in
    // CharUnits.
    //return BaseOffset(DerivedRD, VirtuaBose, aBlnVirtualOffset);
    if (VirtualBase) {
      const ASTRecordLayout &Layout = Context.getASTRecordLayout(DerivedRD);
      CharUnits VirtualOffset = Layout.getVBaseClassOffset(VirtualBase);
      return (NonVirtualOffset + VirtualOffset).getQuantity();
    }
    return NonVirtualOffset.getQuantity();

  }

  int64_t GetBaseClassOffset(TCppScope_t derived, TCppScope_t base) {
    if (base == derived)
      return 0;

    assert(derived || base);

    auto *DD = (Decl *) derived;
    auto *BD = (Decl *) base;
    if (!isa<CXXRecordDecl>(DD) || !isa<CXXRecordDecl>(BD))
      return -1;
    CXXRecordDecl *DCXXRD = cast<CXXRecordDecl>(DD);
    CXXRecordDecl *BCXXRD = cast<CXXRecordDecl>(BD);
    CXXBasePaths Paths(/*FindAmbiguities=*/false, /*RecordPaths=*/true,
                       /*DetectVirtual=*/false);
    DCXXRD->isDerivedFrom(BCXXRD, Paths);

    // FIXME: We might want to cache these requests as they seem expensive.
    return ComputeBaseOffset(getSema().getASTContext(), DCXXRD, Paths.front());
  }

  template <typename DeclType>
  static void GetClassDecls(TCppScope_t klass,
                            std::vector<TCppFunction_t>& methods) {
    if (!klass)
      return;

    auto* D = (clang::Decl*)klass;

    if (auto* TD = dyn_cast<TypedefNameDecl>(D))
      D = GetScopeFromType(TD->getUnderlyingType());

    if (!D || !isa<CXXRecordDecl>(D))
      return;

    auto* CXXRD = dyn_cast<CXXRecordDecl>(D);
#ifdef CPPINTEROP_USE_CLING
    cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif // CPPINTEROP_USE_CLING
    getSema().ForceDeclarationOfImplicitMembers(CXXRD);
    for (Decl* DI : CXXRD->decls()) {
      if (auto* MD = dyn_cast<DeclType>(DI))
        methods.push_back(MD);
      else if (auto* USD = dyn_cast<UsingShadowDecl>(DI))
        if (auto* MD = dyn_cast<DeclType>(USD->getTargetDecl()))
          methods.push_back(MD);
    }
  }

  void GetClassMethods(TCppScope_t klass,
                       std::vector<TCppFunction_t>& methods) {
    GetClassDecls<CXXMethodDecl>(klass, methods);
  }

  void GetFunctionTemplatedDecls(TCppScope_t klass,
                                 std::vector<TCppFunction_t>& methods) {
    GetClassDecls<FunctionTemplateDecl>(klass, methods);
  }

  bool HasDefaultConstructor(TCppScope_t scope) {
    auto *D = (clang::Decl *) scope;

    if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D))
      return CXXRD->hasDefaultConstructor();

    return false;
  }

  TCppFunction_t GetDefaultConstructor(TCppScope_t scope) {
    if (!HasDefaultConstructor(scope))
      return nullptr;

    auto *CXXRD = (clang::CXXRecordDecl*)scope;
    return getSema().LookupDefaultConstructor(CXXRD);
  }

  TCppFunction_t GetDestructor(TCppScope_t scope) {
    auto *D = (clang::Decl *) scope;

    if (auto *CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
      getSema().ForceDeclarationOfImplicitMembers(CXXRD);
      return CXXRD->getDestructor();
    }

    return 0;
  }

  void DumpScope(TCppScope_t scope)
  {
    auto *D = (clang::Decl *) scope;
    D->dump();
  }

  std::vector<TCppFunction_t> GetFunctionsUsingName(
        TCppScope_t scope, const std::string& name)
  {
    auto *D = (Decl *) scope;

    if (!scope || name.empty())
      return {};

    D = GetUnderlyingScope(D);

    std::vector<TCppFunction_t> funcs;
    llvm::StringRef Name(name);
    auto &S = getSema();
    DeclarationName DName = &getASTContext().Idents.get(name);
    clang::LookupResult R(S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                          For_Visible_Redeclaration);

    Cpp_utils::Lookup::Named(&S, R, Decl::castToDeclContext(D));

    if (R.empty())
      return funcs;

    R.resolveKind();

    for (auto *Found : R)
      if (llvm::isa<FunctionDecl>(Found))
        funcs.push_back(Found);

    return funcs;
  }

  TCppType_t GetFunctionReturnType(TCppFunction_t func)
  {
    auto *D = (clang::Decl *) func;
    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D)) {
      QualType Type = FD->getReturnType();
      if (Type->isUndeducedAutoType() && IsTemplatedFunction(FD) &&
          !FD->isDefined()) {
#ifdef CPPINTEROP_USE_CLING
        cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif
        getSema().InstantiateFunctionDefinition(SourceLocation(), FD, true,
                                                true);
        Type = FD->getReturnType();
      }
      return Type.getAsOpaquePtr();
    }

    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
      return (FD->getTemplatedDecl())->getReturnType().getAsOpaquePtr();

    return 0;
  }

  TCppIndex_t GetFunctionNumArgs(TCppFunction_t func)
  {
    auto *D = (clang::Decl *) func;
    if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D))
      return FD->getNumParams();

    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
      return (FD->getTemplatedDecl())->getNumParams();

    return 0;
  }

  TCppIndex_t GetFunctionRequiredArgs(TCppConstFunction_t func)
  {
    const auto* D = static_cast<const clang::Decl*>(func);
    if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D))
      return FD->getMinRequiredArguments();

    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
      return (FD->getTemplatedDecl())->getMinRequiredArguments();

    return 0;
  }

  TCppType_t GetFunctionArgType(TCppFunction_t func, TCppIndex_t iarg)
  {
    auto *D = (clang::Decl *) func;

    if (auto *FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D)) {
        if (iarg < FD->getNumParams()) {
            auto *PVD = FD->getParamDecl(iarg);
            return PVD->getOriginalType().getAsOpaquePtr();
        }
    }

    return 0;
  }

  std::string GetFunctionSignature(TCppFunction_t func) {
    if (!func)
      return "<unknown>";

    auto *D = (clang::Decl *) func;
    if (auto *FD = llvm::dyn_cast<FunctionDecl>(D)) {
      std::string Signature;
      raw_string_ostream SS(Signature);
      PrintingPolicy Policy = getASTContext().getPrintingPolicy();
      // Skip printing the body
      Policy.TerseOutput = true;
      Policy.FullyQualifiedName = true;
      Policy.SuppressDefaultTemplateArgs = false;
      FD->print(SS, Policy);
      SS.flush();
      return Signature;
    }

    return "<unknown>";
  }

  // Internal functions that are not needed outside the library are
  // encompassed in an anonymous namespace as follows.
  namespace {
    bool IsTemplatedFunction(Decl *D) {
      if (llvm::isa_and_nonnull<FunctionTemplateDecl>(D))
        return true;

      if (auto *FD = llvm::dyn_cast_or_null<FunctionDecl>(D)) {
        auto TK = FD->getTemplatedKind();
        return TK == FunctionDecl::TemplatedKind::
                     TK_FunctionTemplateSpecialization
              || TK == FunctionDecl::TemplatedKind::
                       TK_DependentFunctionTemplateSpecialization
              || TK == FunctionDecl::TemplatedKind::TK_FunctionTemplate;
      }

      return false;
    }
  }

  bool IsFunctionDeleted(TCppConstFunction_t function) {
    const auto* FD =
        cast<const FunctionDecl>(static_cast<const clang::Decl*>(function));
    return FD->isDeleted();
  }

  bool IsTemplatedFunction(TCppFunction_t func)
  {
    auto *D = (Decl *) func;
    return IsTemplatedFunction(D);
  }

  bool ExistsFunctionTemplate(const std::string& name,
          TCppScope_t parent)
  {
    DeclContext *Within = 0;
    if (parent) {
      auto* D = (Decl*)parent;
      Within = llvm::dyn_cast<DeclContext>(D);
    }

    auto *ND = Cpp_utils::Lookup::Named(&getSema(), name, Within);

    if ((intptr_t) ND == (intptr_t) 0)
      return false;

    if ((intptr_t) ND != (intptr_t) -1)
      return IsTemplatedFunction(ND);

    // FIXME: Cycle through the Decls and check if there is a templated function
    return true;
  }

  void GetClassTemplatedMethods(const std::string& name, TCppScope_t parent,
                                std::vector<TCppFunction_t>& funcs) {

    auto* D = (Decl*)parent;

    if (!parent || name.empty())
      return;

    D = GetUnderlyingScope(D);

    llvm::StringRef Name(name);
    auto& S = getSema();
    DeclarationName DName = &getASTContext().Idents.get(name);
    clang::LookupResult R(S, DName, SourceLocation(), Sema::LookupOrdinaryName,
                          For_Visible_Redeclaration);

    Cpp_utils::Lookup::Named(&S, R, Decl::castToDeclContext(D));

    if (R.empty())
      return;

    R.resolveKind();

    for (auto* Found : R)
      if (llvm::isa<FunctionTemplateDecl>(Found))
        funcs.push_back(Found);
  }

  // Adapted from inner workings of Sema::BuildCallExpr
  TCppFunction_t
  BestOverloadFunctionMatch(const std::vector<TCppFunction_t>& candidates,
                            const std::vector<TemplateArgInfo>& explicit_types,
                            const std::vector<TemplateArgInfo>& arg_types) {
    auto& S = getSema();
    auto& C = S.getASTContext();

#ifdef CPPINTEROP_USE_CLING
    cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif

    // The overload resolution interfaces in Sema require a list of expressions.
    // However, unlike handwritten C++, we do not always have a expression.
    // Here we synthesize a placeholder expression to be able to use
    // Sema::AddOverloadCandidate. Made up expressions are fine because the
    // interface uses the list size and the expression types.
    struct WrapperExpr : public OpaqueValueExpr {
      WrapperExpr() : OpaqueValueExpr(clang::Stmt::EmptyShell()) {}
    };
    auto* Exprs = new WrapperExpr[arg_types.size()];
    llvm::SmallVector<Expr*> Args;
    Args.reserve(arg_types.size());
    size_t idx = 0;
    for (auto i : arg_types) {
      QualType Type = QualType::getFromOpaquePtr(i.m_Type);
      ExprValueKind ExprKind = ExprValueKind::VK_PRValue;
      if (Type->isReferenceType())
        ExprKind = ExprValueKind::VK_LValue;

      new (&Exprs[idx]) OpaqueValueExpr(SourceLocation::getFromRawEncoding(1),
                                        Type.getNonReferenceType(), ExprKind);
      Args.push_back(&Exprs[idx]);
      ++idx;
    }

    // Create a list of template arguments.
    llvm::SmallVector<TemplateArgument> TemplateArgs;
    TemplateArgs.reserve(explicit_types.size());
    for (auto explicit_type : explicit_types) {
      QualType ArgTy = QualType::getFromOpaquePtr(explicit_type.m_Type);
      if (explicit_type.m_IntegralValue) {
        // We have a non-type template parameter. Create an integral value from
        // the string representation.
        auto Res = llvm::APSInt(explicit_type.m_IntegralValue);
        Res = Res.extOrTrunc(C.getIntWidth(ArgTy));
        TemplateArgs.push_back(TemplateArgument(C, Res, ArgTy));
      } else {
        TemplateArgs.push_back(ArgTy);
      }
    }

    TemplateArgumentListInfo ExplicitTemplateArgs{};
    for (auto TA : TemplateArgs)
      ExplicitTemplateArgs.addArgument(
          S.getTrivialTemplateArgumentLoc(TA, QualType(), SourceLocation()));

    OverloadCandidateSet Overloads(
        SourceLocation(), OverloadCandidateSet::CandidateSetKind::CSK_Normal);

    for (void* i : candidates) {
      Decl* D = static_cast<Decl*>(i);
      if (auto* FD = dyn_cast<FunctionDecl>(D)) {
        S.AddOverloadCandidate(FD, DeclAccessPair::make(FD, FD->getAccess()),
                               Args, Overloads);
      } else if (auto* FTD = dyn_cast<FunctionTemplateDecl>(D)) {
        // AddTemplateOverloadCandidate is causing a memory leak
        // It is a known bug at clang
        // call stack: AddTemplateOverloadCandidate -> MakeDeductionFailureInfo
        // source:
        // https://github.com/llvm/llvm-project/blob/release/19.x/clang/lib/Sema/SemaOverload.cpp#L731-L756
        S.AddTemplateOverloadCandidate(
            FTD, DeclAccessPair::make(FTD, FTD->getAccess()),
            &ExplicitTemplateArgs, Args, Overloads);
      }
    }

    OverloadCandidateSet::iterator Best;
    Overloads.BestViableFunction(S, SourceLocation(), Best);

    FunctionDecl* Result = Best != Overloads.end() ? Best->Function : nullptr;
    delete[] Exprs;
    return Result;
  }

  // Gets the AccessSpecifier of the function and checks if it is equal to
  // the provided AccessSpecifier.
  bool CheckMethodAccess(TCppFunction_t method, AccessSpecifier AS)
  {
    auto *D = (Decl *) method;
    if (auto *CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
      return CXXMD->getAccess() == AS;
    }

    return false;
  }

  bool IsMethod(TCppConstFunction_t method)
  {
    return dyn_cast_or_null<CXXMethodDecl>(
        static_cast<const clang::Decl*>(method));
  }

  bool IsPublicMethod(TCppFunction_t method)
  {
    return CheckMethodAccess(method, AccessSpecifier::AS_public);
  }

  bool IsProtectedMethod(TCppFunction_t method) {
    return CheckMethodAccess(method, AccessSpecifier::AS_protected);
  }

  bool IsPrivateMethod(TCppFunction_t method)
  {
    return CheckMethodAccess(method, AccessSpecifier::AS_private);
  }

  bool IsConstructor(TCppConstFunction_t method)
  {
    const auto* D = static_cast<const Decl*>(method);
    return llvm::isa_and_nonnull<CXXConstructorDecl>(D);
  }

  bool IsDestructor(TCppConstFunction_t method)
  {
    const auto* D = static_cast<const Decl*>(method);
    return llvm::isa_and_nonnull<CXXDestructorDecl>(D);
  }

  bool IsStaticMethod(TCppConstFunction_t method) {
    const auto* D = static_cast<const Decl*>(method);
    if (auto *CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
      return CXXMD->isStatic();
    }

    return false;
  }

  TCppFuncAddr_t GetFunctionAddress(const char* mangled_name) {
    auto& I = getInterp();
    auto FDAorErr = compat::getSymbolAddress(I, mangled_name);
    if (llvm::Error Err = FDAorErr.takeError())
      llvm::consumeError(std::move(Err)); // nullptr if missing
    else
      return llvm::jitTargetAddressToPointer<void*>(*FDAorErr);

    return nullptr;
  }

  TCppFuncAddr_t GetFunctionAddress(TCppFunction_t method)
  {
    auto *D = (Decl *) method;

    const auto get_mangled_name = [](FunctionDecl* FD) {
      auto MangleCtxt = getASTContext().createMangleContext();

      if (!MangleCtxt->shouldMangleDeclName(FD)) {
        return FD->getNameInfo().getName().getAsString();
      }

      std::string mangled_name;
      llvm::raw_string_ostream ostream(mangled_name);

      MangleCtxt->mangleName(FD, ostream);

      ostream.flush();
      delete MangleCtxt;

      return mangled_name;
    };

    if (auto* FD = llvm::dyn_cast_or_null<FunctionDecl>(D))
      return GetFunctionAddress(get_mangled_name(FD).c_str());

    return 0;
  }

  bool IsVirtualMethod(TCppFunction_t method) {
    auto *D = (Decl *) method;
    if (auto *CXXMD = llvm::dyn_cast_or_null<CXXMethodDecl>(D)) {
      return CXXMD->isVirtual();
    }

    return false;
  }

  void GetDatamembers(TCppScope_t scope,
                      std::vector<TCppScope_t>& datamembers) {
    auto *D = (Decl *) scope;

    if (auto* CXXRD = llvm::dyn_cast_or_null<CXXRecordDecl>(D)) {
      getSema().ForceDeclarationOfImplicitMembers(CXXRD);

      llvm::SmallVector<RecordDecl::decl_iterator, 2> stack_begin;
      llvm::SmallVector<RecordDecl::decl_iterator, 2> stack_end;
      stack_begin.push_back(CXXRD->decls_begin());
      stack_end.push_back(CXXRD->decls_end());
      while (!stack_begin.empty()) {
        if (stack_begin.back() == stack_end.back()) {
          stack_begin.pop_back();
          stack_end.pop_back();
          continue;
        }
        Decl* D = *(stack_begin.back());
        if (auto* FD = llvm::dyn_cast<FieldDecl>(D)) {
          if (FD->isAnonymousStructOrUnion()) {
            if (const auto* RT = FD->getType()->getAs<RecordType>()) {
              if (auto* CXXRD = llvm::dyn_cast<CXXRecordDecl>(RT->getDecl())) {
                stack_begin.back()++;
                stack_begin.push_back(CXXRD->decls_begin());
                stack_end.push_back(CXXRD->decls_end());
                continue;
              }
            }
          }
          datamembers.push_back((TCppScope_t)D);

        } else if (auto* USD = llvm::dyn_cast<UsingShadowDecl>(D)) {
          if (llvm::isa<FieldDecl>(USD->getTargetDecl()))
            datamembers.push_back(USD);
        }
        stack_begin.back()++;
      }
    }
  }

  void GetStaticDatamembers(TCppScope_t scope,
                            std::vector<TCppScope_t>& datamembers) {
    GetClassDecls<VarDecl>(scope, datamembers);
  }

  void GetEnumConstantDatamembers(TCppScope_t scope,
                                  std::vector<TCppScope_t>& datamembers,
                                  bool include_enum_class) {
    std::vector<TCppScope_t> EDs;
    GetClassDecls<EnumDecl>(scope, EDs);
    for (TCppScope_t i : EDs) {
      auto* ED = static_cast<EnumDecl*>(i);

      bool is_class_tagged = ED->isScopedUsingClassTag();
      if (is_class_tagged && !include_enum_class)
        continue;

      std::copy(ED->enumerator_begin(), ED->enumerator_end(),
                std::back_inserter(datamembers));
    }
  }

  TCppScope_t LookupDatamember(const std::string& name, TCppScope_t parent) {
    clang::DeclContext *Within = 0;
    if (parent) {
      auto *D = (clang::Decl *)parent;
      Within = llvm::dyn_cast<clang::DeclContext>(D);
    }

    auto *ND = Cpp_utils::Lookup::Named(&getSema(), name, Within);
    if (ND && ND != (clang::NamedDecl*) -1) {
      if (llvm::isa_and_nonnull<clang::FieldDecl>(ND)) {
        return (TCppScope_t)ND;
      }
    }

    return 0;
  }

  TCppType_t GetVariableType(TCppScope_t var) {
    auto* D = static_cast<Decl*>(var);

    if (auto DD = llvm::dyn_cast_or_null<DeclaratorDecl>(D)) {
      QualType QT = DD->getType();

      // Check if the type is a typedef type
      if (QT->isTypedefNameType()) {
        return QT.getAsOpaquePtr();
      }

      // Else, return the canonical type
      QT = QT.getCanonicalType();
      return QT.getAsOpaquePtr();
    }

    if (auto* ECD = llvm::dyn_cast_or_null<EnumConstantDecl>(D))
      return ECD->getType().getAsOpaquePtr();

    return 0;
  }

  intptr_t GetVariableOffset(compat::Interpreter& I, Decl* D,
                             CXXRecordDecl* BaseCXXRD) {
    if (!D)
      return 0;

    auto& C = I.getSema().getASTContext();

    if (auto* FD = llvm::dyn_cast<FieldDecl>(D)) {
      clang::RecordDecl* FieldParentRecordDecl = FD->getParent();
      intptr_t offset =
          C.toCharUnitsFromBits(C.getFieldOffset(FD)).getQuantity();
      while (FieldParentRecordDecl->isAnonymousStructOrUnion()) {
        clang::RecordDecl* anon = FieldParentRecordDecl;
        FieldParentRecordDecl = llvm::dyn_cast<RecordDecl>(anon->getParent());
        for (auto F = FieldParentRecordDecl->field_begin();
             F != FieldParentRecordDecl->field_end(); ++F) {
          const auto* RT = F->getType()->getAs<RecordType>();
          if (!RT)
            continue;
          if (anon == RT->getDecl()) {
            FD = *F;
            break;
          }
        }
        offset += C.toCharUnitsFromBits(C.getFieldOffset(FD)).getQuantity();
      }
      if (BaseCXXRD && BaseCXXRD != FieldParentRecordDecl) {
        // FieldDecl FD belongs to some class C, but the base class BaseCXXRD is
        // not C. That means BaseCXXRD derives from C. Offset needs to be
        // calculated for Derived class

        // Depth first Search is performed to the class that declears FD from
        // the base class
        std::vector<CXXRecordDecl*> stack;
        std::map<CXXRecordDecl*, CXXRecordDecl*> direction;
        stack.push_back(BaseCXXRD);
        while (!stack.empty()) {
          CXXRecordDecl* RD = stack.back();
          stack.pop_back();
          size_t num_bases = GetNumBases(RD);
          bool flag = false;
          for (size_t i = 0; i < num_bases; i++) {
            auto* CRD = static_cast<CXXRecordDecl*>(GetBaseClass(RD, i));
            direction[CRD] = RD;
            if (CRD == FieldParentRecordDecl) {
              flag = true;
              break;
            }
            stack.push_back(CRD);
          }
          if (flag)
            break;
        }
        if (auto* RD = llvm::dyn_cast<CXXRecordDecl>(FieldParentRecordDecl)) {
          // add in the offsets for the (multi level) base classes
          while (BaseCXXRD != RD) {
            CXXRecordDecl* Parent = direction.at(RD);
            offset += C.getASTRecordLayout(Parent)
                          .getBaseClassOffset(RD)
                          .getQuantity();
            RD = Parent;
          }
        } else {
          assert(false && "Unreachable");
        }
      }
      return offset;
    }

    if (auto *VD = llvm::dyn_cast<VarDecl>(D)) {
      auto GD = GlobalDecl(VD);
      std::string mangledName;
      compat::maybeMangleDeclName(GD, mangledName);
      void* address = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(
          mangledName.c_str());

      if (!address)
        address = I.getAddressOfGlobal(GD);
      if (!address) {
        if (!VD->hasInit()) {
#ifdef CPPINTEROP_USE_CLING
          cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif // CPPINTEROP_USE_CLING
          getSema().InstantiateVariableDefinition(SourceLocation(), VD);
        }
        if (VD->hasInit() &&
            (VD->isConstexpr() || VD->getType().isConstQualified())) {
          if (const APValue* val = VD->evaluateValue()) {
            if (VD->getType()->isIntegralType(C)) {
              return (intptr_t)val->getInt().getRawData();
            }
          }
        }
      }
      if (!address) {
        auto Linkage = C.GetGVALinkageForVariable(VD);
        // The decl was deferred by CodeGen. Force its emission.
        // FIXME: In ASTContext::DeclMustBeEmitted we should check if the
        // Decl::isUsed is set or we should be able to access CodeGen's
        // addCompilerUsedGlobal.
        if (isDiscardableGVALinkage(Linkage))
          VD->addAttr(UsedAttr::CreateImplicit(C));
#ifdef CPPINTEROP_USE_CLING
        cling::Interpreter::PushTransactionRAII RAII(&I);
        I.getCI()->getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
#else // CLANG_REPL
        I.getCI()->getASTConsumer().HandleTopLevelDecl(DeclGroupRef(VD));
        // Take the newest llvm::Module produced by CodeGen and send it to JIT.
        auto GeneratedPTU = I.Parse("");
        if (!GeneratedPTU)
          llvm::logAllUnhandledErrors(GeneratedPTU.takeError(), llvm::errs(),
                                 "[GetVariableOffset] Failed to generate PTU:");

        // From cling's BackendPasses.cpp
        // FIXME: We need to upstream this code in IncrementalExecutor::addModule
        for (auto &GV : GeneratedPTU->TheModule->globals()) {
          llvm::GlobalValue::LinkageTypes LT = GV.getLinkage();
          if (GV.isDeclaration() || !GV.hasName() ||
              GV.getName().starts_with(".str") ||
              !GV.isDiscardableIfUnused(LT) ||
              LT != llvm::GlobalValue::InternalLinkage)
            continue; //nothing to do
          GV.setLinkage(llvm::GlobalValue::WeakAnyLinkage);
        }
        if (auto Err = I.Execute(*GeneratedPTU))
          llvm::logAllUnhandledErrors(std::move(Err), llvm::errs(),
                                  "[GetVariableOffset] Failed to execute PTU:");
#endif
      }
      auto VDAorErr = compat::getSymbolAddress(I, StringRef(mangledName));
      if (!VDAorErr) {
        llvm::logAllUnhandledErrors(VDAorErr.takeError(), llvm::errs(),
                                    "Failed to GetVariableOffset:");
        return 0;
      }
      return (intptr_t)jitTargetAddressToPointer<void*>(VDAorErr.get());
    }

    return 0;
  }

  intptr_t GetVariableOffset(TCppScope_t var, TCppScope_t parent) {
    auto* D = static_cast<Decl*>(var);
    auto* RD =
        llvm::dyn_cast_or_null<CXXRecordDecl>(static_cast<Decl*>(parent));
    return GetVariableOffset(getInterp(), D, RD);
  }

  // Check if the Access Specifier of the variable matches the provided value.
  bool CheckVariableAccess(TCppScope_t var, AccessSpecifier AS)
  {
    auto *D = (Decl *) var;
    return D->getAccess() == AS;
  }

  bool IsPublicVariable(TCppScope_t var)
  {
    return CheckVariableAccess(var, AccessSpecifier::AS_public);
  }

  bool IsProtectedVariable(TCppScope_t var)
  {
    return CheckVariableAccess(var, AccessSpecifier::AS_protected);
  }

  bool IsPrivateVariable(TCppScope_t var)
  {
    return CheckVariableAccess(var, AccessSpecifier::AS_private);
  }

  bool IsStaticVariable(TCppScope_t var)
  {
    auto *D = (Decl *) var;
    if (llvm::isa_and_nonnull<VarDecl>(D)) {
      return true;
    }

    return false;
  }

  bool IsConstVariable(TCppScope_t var)
  {
    auto *D = (clang::Decl *) var;

    if (auto *VD = llvm::dyn_cast_or_null<ValueDecl>(D)) {
      return VD->getType().isConstQualified();
    }

    return false;
  }

  bool IsRecordType(TCppType_t type)
  {
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->isRecordType();
  }

  bool IsPODType(TCppType_t type)
  {
    QualType QT = QualType::getFromOpaquePtr(type);

    if (QT.isNull())
      return false;

    return QT.isPODType(getASTContext());
  }

  bool IsPointerType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->isPointerType();
  }

  TCppType_t GetPointeeType(TCppType_t type) {
    if (!IsPointerType(type))
      return nullptr;
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->getPointeeType().getAsOpaquePtr();
  }

  bool IsReferenceType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT->isReferenceType();
  }

  TCppType_t GetNonReferenceType(TCppType_t type) {
    if (!IsReferenceType(type))
      return nullptr;
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT.getNonReferenceType().getAsOpaquePtr();
  }

  TCppType_t GetUnderlyingType(TCppType_t type)
  {
    QualType QT = QualType::getFromOpaquePtr(type);
    QT = QT->getCanonicalTypeUnqualified();

    // Recursively remove array dimensions
    while (QT->isArrayType())
      QT = QualType(QT->getArrayElementTypeNoTypeQual(), 0);

    // Recursively reduce pointer depth till we are left with a pointerless
    // type.
    for (auto PT = QT->getPointeeType(); !PT.isNull(); PT = QT->getPointeeType()){
      QT = PT;
    }
    QT = QT->getCanonicalTypeUnqualified();
    return QT.getAsOpaquePtr();
  }

  std::string GetTypeAsString(TCppType_t var)
  {
      QualType QT = QualType::getFromOpaquePtr(var);
      // FIXME: Get the default printing policy from the ASTContext.
      PrintingPolicy Policy((LangOptions()));
      Policy.Bool = true; // Print bool instead of _Bool.
      Policy.SuppressTagKeyword = true; // Do not print `class std::string`.
      return compat::FixTypeName(QT.getAsString(Policy));
  }

  TCppType_t GetCanonicalType(TCppType_t type)
  {
    if (!type)
      return 0;
    QualType QT = QualType::getFromOpaquePtr(type);
    return QT.getCanonicalType().getAsOpaquePtr();
  }

  // Internal functions that are not needed outside the library are
  // encompassed in an anonymous namespace as follows. This function converts
  // from a string to the actual type. It is used in the GetType() function.
  namespace {
    static QualType findBuiltinType(llvm::StringRef typeName, ASTContext &Context)
    {
      bool issigned = false;
      bool isunsigned = false;
      if (typeName.starts_with("signed ")) {
        issigned = true;
        typeName = StringRef(typeName.data()+7, typeName.size()-7);
      }
      if (!issigned && typeName.starts_with("unsigned ")) {
        isunsigned = true;
        typeName = StringRef(typeName.data()+9, typeName.size()-9);
      }
      if (typeName == "char") {
        if (isunsigned) return Context.UnsignedCharTy;
        return Context.SignedCharTy;
      }
      if (typeName == "short") {
        if (isunsigned) return Context.UnsignedShortTy;
        return Context.ShortTy;
      }
      if (typeName == "int") {
        if (isunsigned) return Context.UnsignedIntTy;
        return Context.IntTy;
      }
      if (typeName == "long") {
        if (isunsigned) return Context.UnsignedLongTy;
        return Context.LongTy;
      }
      if (typeName == "long long") {
        if (isunsigned)
        return Context.UnsignedLongLongTy;
        return Context.LongLongTy;
      }
      if (!issigned && !isunsigned) {
        if (typeName == "bool")
          return Context.BoolTy;
        if (typeName == "float")
          return Context.FloatTy;
        if (typeName == "double")
          return Context.DoubleTy;
        if (typeName == "long double")
          return Context.LongDoubleTy;

        if (typeName == "wchar_t")
          return Context.WCharTy;
        if (typeName == "char16_t")
          return Context.Char16Ty;
        if (typeName == "char32_t")
          return Context.Char32Ty;
      }
      /* Missing
     CanQualType WideCharTy; // Same as WCharTy in C++, integer type in C99.
     CanQualType WIntTy;   // [C99 7.24.1], integer type unchanged by default promotions.
       */
      return QualType();
    }
  }

  TCppType_t GetType(const std::string &name) {
    QualType builtin = findBuiltinType(name, getASTContext());
    if (!builtin.isNull())
      return builtin.getAsOpaquePtr();

    auto *D = (Decl *) GetNamed(name, /* Within= */ 0);
    if (auto *TD = llvm::dyn_cast_or_null<TypeDecl>(D)) {
      return QualType(TD->getTypeForDecl(), 0).getAsOpaquePtr();
    }

    return (TCppType_t)0;
  }

  TCppType_t GetComplexType(TCppType_t type) {
    QualType QT = QualType::getFromOpaquePtr(type);

    return getASTContext().getComplexType(QT).getAsOpaquePtr();
  }

  TCppType_t GetTypeFromScope(TCppScope_t klass) {
    if (!klass)
      return 0;

    auto *D = (Decl *) klass;
    ASTContext &C = getASTContext();

    if (ValueDecl *VD = dyn_cast<ValueDecl>(D))
      return VD->getType().getAsOpaquePtr();

    return C.getTypeDeclType(cast<TypeDecl>(D)).getAsOpaquePtr();
  }

  // Internal functions that are not needed outside the library are
  // encompassed in an anonymous namespace as follows.
  namespace {
    static unsigned long long gWrapperSerial = 0LL;

    enum EReferenceType { kNotReference, kLValueReference, kRValueReference };

    // Start of JitCall Helper Functions

#define DEBUG_TYPE "jitcall"

    // FIXME: Use that routine throughout CallFunc's port in places such as
    // make_narg_call.
    static inline void indent(ostringstream &buf, int indent_level) {
      static const std::string kIndentString("   ");
      for (int i = 0; i < indent_level; ++i)
        buf << kIndentString;
    }

    void *compile_wrapper(compat::Interpreter& I,
                          const std::string& wrapper_name,
                          const std::string& wrapper,
                          bool withAccessControl = true) {
      LLVM_DEBUG(dbgs() << "Compiling '" << wrapper_name << "'\n");
      return I.compileFunction(wrapper_name, wrapper, false /*ifUnique*/,
                                withAccessControl);
    }

    void get_type_as_string(QualType QT, std::string& type_name, ASTContext& C,
                            PrintingPolicy Policy) {
      //TODO: Implement cling desugaring from utils::AST
      //      cling::utils::Transform::GetPartiallyDesugaredType()
      if (!QT->isTypedefNameType() || QT->isBuiltinType())
        QT = QT.getDesugaredType(C);
#if CLANG_VERSION_MAJOR > 16
      Policy.SuppressElaboration = true;
#endif
      Policy.FullyQualifiedName = true;
      QT.getAsStringInternal(type_name, Policy);
    }

    void collect_type_info(const FunctionDecl* FD, QualType& QT,
                           std::ostringstream& typedefbuf,
                           std::ostringstream& callbuf, std::string& type_name,
                           EReferenceType& refType, bool& isPointer,
                           int indent_level, bool forArgument) {
      //
      //  Collect information about the type of a function parameter
      //  needed for building the wrapper function.
      //
      ASTContext& C = FD->getASTContext();
      PrintingPolicy Policy(C.getPrintingPolicy());
#if CLANG_VERSION_MAJOR > 16
      Policy.SuppressElaboration = true;
#endif
      refType = kNotReference;
      if (QT->isRecordType() && forArgument) {
        get_type_as_string(QT, type_name, C, Policy);
        return;
      }
      if (QT->isFunctionPointerType()) {
        std::string fp_typedef_name;
        {
          std::ostringstream nm;
          nm << "FP" << gWrapperSerial++;
          type_name = nm.str();
          raw_string_ostream OS(fp_typedef_name);
          QT.print(OS, Policy, type_name);
          OS.flush();
        }

        indent(typedefbuf, indent_level);

        typedefbuf << "typedef " << fp_typedef_name << ";\n";
        return;
      } else if (QT->isMemberPointerType()) {
        std::string mp_typedef_name;
        {
          std::ostringstream nm;
          nm << "MP" << gWrapperSerial++;
          type_name = nm.str();
          raw_string_ostream OS(mp_typedef_name);
          QT.print(OS, Policy, type_name);
          OS.flush();
        }

        indent(typedefbuf, indent_level);

        typedefbuf << "typedef " << mp_typedef_name << ";\n";
        return;
      } else if (QT->isPointerType()) {
        isPointer = true;
        QT = cast<clang::PointerType>(QT.getCanonicalType())->getPointeeType();
      } else if (QT->isReferenceType()) {
        if (QT->isRValueReferenceType())
          refType = kRValueReference;
        else
          refType = kLValueReference;
        QT = cast<ReferenceType>(QT.getCanonicalType())->getPointeeType();
      }
      // Fall through for the array type to deal with reference/pointer ro array
      // type.
      if (QT->isArrayType()) {
        std::string ar_typedef_name;
        {
          std::ostringstream ar;
          ar << "AR" << gWrapperSerial++;
          type_name = ar.str();
          raw_string_ostream OS(ar_typedef_name);
          QT.print(OS, Policy, type_name);
          OS.flush();
        }
        indent(typedefbuf, indent_level);
        typedefbuf << "typedef " << ar_typedef_name << ";\n";
        return;
      }
      get_type_as_string(QT, type_name, C, Policy);
    }

    void make_narg_ctor(const FunctionDecl* FD, const unsigned N,
                        std::ostringstream& typedefbuf,
                        std::ostringstream& callbuf,
                        const std::string& class_name, int indent_level) {
      // Make a code string that follows this pattern:
      //
      // ClassName(args...)
      //

      callbuf << class_name << "(";
      for (unsigned i = 0U; i < N; ++i) {
        const ParmVarDecl* PVD = FD->getParamDecl(i);
        QualType Ty = PVD->getType();
        QualType QT = Ty.getCanonicalType();
        std::string type_name;
        EReferenceType refType = kNotReference;
        bool isPointer = false;
        collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                          isPointer, indent_level, true);
        if (i) {
          callbuf << ',';
          if (i % 2) {
            callbuf << ' ';
          } else {
            callbuf << "\n";
            indent(callbuf, indent_level);
          }
        }
        if (refType != kNotReference) {
          callbuf << "(" << type_name.c_str()
                  << (refType == kLValueReference ? "&" : "&&") << ")*("
                  << type_name.c_str() << "*)args[" << i << "]";
        } else if (isPointer) {
          callbuf << "*(" << type_name.c_str() << "**)args[" << i << "]";
        } else {
          callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
        }
      }
      callbuf << ")";
    }

    const DeclContext* get_non_transparent_decl_context(const FunctionDecl* FD) {
      auto *DC = FD->getDeclContext();
      while (DC->isTransparentContext()) {
        DC = DC->getParent();
        assert(DC && "All transparent contexts should have a parent!");
      }
      return DC;
    }

    void make_narg_call(const FunctionDecl* FD, const std::string& return_type,
                        const unsigned N, std::ostringstream& typedefbuf,
                        std::ostringstream& callbuf,
                        const std::string& class_name, int indent_level) {
      //
      // Make a code string that follows this pattern:
      //
      // ((<class>*)obj)-><method>(*(<arg-i-type>*)args[i], ...)
      //

      // Sometimes it's necessary that we cast the function we want to call
      // first to its explicit function type before calling it. This is supposed
      // to prevent that we accidentally ending up in a function that is not
      // the one we're supposed to call here (e.g. because the C++ function
      // lookup decides to take another function that better fits). This method
      // has some problems, e.g. when we call a function with default arguments
      // and we don't provide all arguments, we would fail with this pattern.
      // Same applies with member methods which seem to cause parse failures
      // even when we supply the object parameter. Therefore we only use it in
      // cases where we know it works and set this variable to true when we do.
      bool ShouldCastFunction =
          !isa<CXXMethodDecl>(FD) && N == FD->getNumParams();
      if (ShouldCastFunction) {
        callbuf << "(";
        callbuf << "(";
        callbuf << return_type << " (&)";
        {
          callbuf << "(";
          for (unsigned i = 0U; i < N; ++i) {
            if (i) {
              callbuf << ',';
              if (i % 2) {
                callbuf << ' ';
              } else {
                callbuf << "\n";
                indent(callbuf, indent_level);
              }
            }
            const ParmVarDecl* PVD = FD->getParamDecl(i);
            QualType Ty = PVD->getType();
            QualType QT = Ty.getCanonicalType();
            std::string arg_type;
            ASTContext& C = FD->getASTContext();
            get_type_as_string(QT, arg_type, C, C.getPrintingPolicy());
            callbuf << arg_type;
          }
          if (FD->isVariadic())
            callbuf << ", ...";
          callbuf << ")";
        }

        callbuf << ")";
      }

      if (const CXXMethodDecl* MD = dyn_cast<CXXMethodDecl>(FD)) {
        // This is a class, struct, or union member.
        if (MD->isConst())
          callbuf << "((const " << class_name << "*)obj)->";
        else
          callbuf << "((" << class_name << "*)obj)->";
      } else if (const NamedDecl* ND =
                     dyn_cast<NamedDecl>(get_non_transparent_decl_context(FD))) {
        // This is a namespace member.
        (void)ND;
        callbuf << class_name << "::";
      }
      //   callbuf << fMethod->Name() << "(";
      {
        std::string name;
        {
          std::string complete_name;
          llvm::raw_string_ostream stream(complete_name);
          FD->getNameForDiagnostic(stream,
                                   FD->getASTContext().getPrintingPolicy(),
                                   /*Qualified=*/false);

          // insert space between template argument list and the function name
          // this is require if the function is `operator<`
          // `operator<<type1, type2, ...>` is invalid syntax
          // whereas `operator< <type1, type2, ...>` is valid
          std::string simple_name = FD->getNameAsString();
          size_t idx = complete_name.find(simple_name, 0) + simple_name.size();
          std::string name_without_template_args = complete_name.substr(0, idx);
          std::string template_args = complete_name.substr(idx);
          name = name_without_template_args +
                 (template_args.empty() ? "" : " " + template_args);
        }
        callbuf << name;
      }
      if (ShouldCastFunction)
        callbuf << ")";

      callbuf << "(";
      for (unsigned i = 0U; i < N; ++i) {
        const ParmVarDecl* PVD = FD->getParamDecl(i);
        QualType Ty = PVD->getType();
        QualType QT = Ty.getCanonicalType();
        std::string type_name;
        EReferenceType refType = kNotReference;
        bool isPointer = false;
        collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                          isPointer, indent_level, true);

        if (i) {
          callbuf << ',';
          if (i % 2) {
            callbuf << ' ';
          } else {
            callbuf << "\n";
            indent(callbuf, indent_level);
          }
        }

        if (refType != kNotReference) {
          callbuf << "(" << type_name.c_str()
                  << (refType == kLValueReference ? "&" : "&&") << ")*("
                  << type_name.c_str() << "*)args[" << i << "]";
        } else if (isPointer) {
          callbuf << "*(" << type_name.c_str() << "**)args[" << i << "]";
        } else {
          // pointer falls back to non-pointer case; the argument preserves
          // the "pointerness" (i.e. doesn't reference the value).
          callbuf << "*(" << type_name.c_str() << "*)args[" << i << "]";
        }
      }
      callbuf << ")";
    }

    void make_narg_ctor_with_return(const FunctionDecl* FD, const unsigned N,
                                    const std::string& class_name,
                                    std::ostringstream& buf, int indent_level) {
      // Make a code string that follows this pattern:
      //
      //  (*(ClassName**)ret) = (obj) ?
      //    new (*(ClassName**)ret) ClassName(args...) : new ClassName(args...);
      //
      {
        std::ostringstream typedefbuf;
        std::ostringstream callbuf;
        //
        //  Write the return value assignment part.
        //
        indent(callbuf, indent_level);
        callbuf << "(*(" << class_name << "**)ret) = ";
        callbuf << "(obj) ? new (*(" << class_name << "**)ret) ";
        make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level);

        callbuf << ": new ";
        //
        //  Write the actual expression.
        //
        make_narg_ctor(FD, N, typedefbuf, callbuf, class_name, indent_level);
        //
        //  End the new expression statement.
        //
        callbuf << ";\n";
        //
        //  Output the whole new expression and return statement.
        //
        buf << typedefbuf.str() << callbuf.str();
      }
    }

    void make_narg_call_with_return(compat::Interpreter& I,
                                    const FunctionDecl* FD, const unsigned N,
                                    const std::string& class_name,
                                    std::ostringstream& buf, int indent_level) {
      // Make a code string that follows this pattern:
      //
      // if (ret) {
      //    new (ret) (return_type) ((class_name*)obj)->func(args...);
      // }
      // else {
      //    (void)(((class_name*)obj)->func(args...));
      // }
      //
      if (const CXXConstructorDecl* CD = dyn_cast<CXXConstructorDecl>(FD)) {
        if (N <= 1 && llvm::isa<UsingShadowDecl>(FD)) {
          auto SpecMemKind = I.getCI()->getSema().getSpecialMember(CD);
          if ((N == 0 &&
               SpecMemKind == CXXSpecialMemberKindDefaultConstructor) ||
              (N == 1 &&
               (SpecMemKind == CXXSpecialMemberKindCopyConstructor ||
                SpecMemKind == CXXSpecialMemberKindMoveConstructor))) {
            // Using declarations cannot inject special members; do not call
            // them as such. This might happen by using `Base(Base&, int = 12)`,
            // which is fine to be called as `Derived d(someBase, 42)` but not
            // as copy constructor of `Derived`.
            return;
          }
        }
        make_narg_ctor_with_return(FD, N, class_name, buf, indent_level);
        return;
      }
      QualType QT = FD->getReturnType();
      if (QT->isVoidType()) {
        std::ostringstream typedefbuf;
        std::ostringstream callbuf;
        indent(callbuf, indent_level);
        make_narg_call(FD, "void", N, typedefbuf, callbuf, class_name,
                       indent_level);
        callbuf << ";\n";
        indent(callbuf, indent_level);
        callbuf << "return;\n";
        buf << typedefbuf.str() << callbuf.str();
      } else {
        indent(buf, indent_level);

        std::string type_name;
        EReferenceType refType = kNotReference;
        bool isPointer = false;

        std::ostringstream typedefbuf;
        std::ostringstream callbuf;

        collect_type_info(FD, QT, typedefbuf, callbuf, type_name, refType,
                          isPointer, indent_level, false);

        buf << typedefbuf.str();

        buf << "if (ret) {\n";
        ++indent_level;
        {
          //
          //  Write the placement part of the placement new.
          //
          indent(callbuf, indent_level);
          callbuf << "new (ret) ";
          //
          //  Write the type part of the placement new.
          //
          callbuf << "(" << type_name.c_str();
          if (refType != kNotReference) {
            callbuf << "*) (&";
            type_name += "&";
          } else if (isPointer) {
            callbuf << "*) (";
            type_name += "*";
          } else {
            callbuf << ") (";
          }
          //
          //  Write the actual function call.
          //
          make_narg_call(FD, type_name, N, typedefbuf, callbuf, class_name,
                         indent_level);
          //
          //  End the placement new.
          //
          callbuf << ");\n";
          indent(callbuf, indent_level);
          callbuf << "return;\n";
          //
          //  Output the whole placement new expression and return statement.
          //
          buf << typedefbuf.str() << callbuf.str();
        }
        --indent_level;
        indent(buf, indent_level);
        buf << "}\n";
        indent(buf, indent_level);
        buf << "else {\n";
        ++indent_level;
        {
          std::ostringstream typedefbuf;
          std::ostringstream callbuf;
          indent(callbuf, indent_level);
          callbuf << "(void)(";
          make_narg_call(FD, type_name, N, typedefbuf, callbuf, class_name,
                         indent_level);
          callbuf << ");\n";
          indent(callbuf, indent_level);
          callbuf << "return;\n";
          buf << typedefbuf.str() << callbuf.str();
        }
        --indent_level;
        indent(buf, indent_level);
        buf << "}\n";
      }
    }

    int get_wrapper_code(compat::Interpreter& I, const FunctionDecl* FD,
                         std::string& wrapper_name, std::string& wrapper) {
      assert(FD && "generate_wrapper called without a function decl!");
      ASTContext& Context = FD->getASTContext();
      PrintingPolicy Policy(Context.getPrintingPolicy());
      //
      //  Get the class or namespace name.
      //
      std::string class_name;
      const clang::DeclContext* DC = get_non_transparent_decl_context(FD);
      if (const TypeDecl* TD = dyn_cast<TypeDecl>(DC)) {
        // This is a class, struct, or union member.
        QualType QT(TD->getTypeForDecl(), 0);
        get_type_as_string(QT, class_name, Context, Policy);
      } else if (const NamedDecl* ND = dyn_cast<NamedDecl>(DC)) {
        // This is a namespace member.
        raw_string_ostream stream(class_name);
        ND->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);
        stream.flush();
      }
      //
      //  Check to make sure that we can
      //  instantiate and codegen this function.
      //
      bool needInstantiation = false;
      const FunctionDecl* Definition = 0;
      if (!FD->isDefined(Definition)) {
        FunctionDecl::TemplatedKind TK = FD->getTemplatedKind();
        switch (TK) {
          case FunctionDecl::TK_NonTemplate: {
            // Ordinary function, not a template specialization.
            // Note: This might be ok, the body might be defined
            //       in a library, and all we have seen is the
            //       header file.
            // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
            //      "Cannot make wrapper for a function which is "
            //      "declared but not defined!";
            // return 0;
          } break;
          case FunctionDecl::TK_FunctionTemplate: {
            // This decl is actually a function template,
            // not a function at all.
            llvm::errs() << "TClingCallFunc::make_wrapper"
                         << ":"
                         << "Cannot make wrapper for a function template!";
            return 0;
          } break;
          case FunctionDecl::TK_MemberSpecialization: {
            // This function is the result of instantiating an ordinary
            // member function of a class template, or of instantiating
            // an ordinary member function of a class member of a class
            // template, or of specializing a member function template
            // of a class template, or of specializing a member function
            // template of a class member of a class template.
            if (!FD->isTemplateInstantiation()) {
              // We are either TSK_Undeclared or
              // TSK_ExplicitSpecialization.
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
              //      "Cannot make wrapper for a function template "
              //      "explicit specialization which is declared "
              //      "but not defined!";
              // return 0;
              break;
            }
            const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
            if (!Pattern) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a member function "
                              "instantiation with no pattern!";
              return 0;
            }
            FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
            TemplateSpecializationKind PTSK =
                Pattern->getTemplateSpecializationKind();
            if (
                // The pattern is an ordinary member function.
                (PTK == FunctionDecl::TK_NonTemplate) ||
                // The pattern is an explicit specialization, and
                // so is not a template.
                ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                 ((PTSK == TSK_Undeclared) ||
                  (PTSK == TSK_ExplicitSpecialization)))) {
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              break;
            } else if (!Pattern->hasBody()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a member function "
                              "instantiation with no body!";
              return 0;
            }
            if (FD->isImplicitlyInstantiable()) {
              needInstantiation = true;
            }
          } break;
          case FunctionDecl::TK_FunctionTemplateSpecialization: {
            // This function is the result of instantiating a function
            // template or possibly an explicit specialization of a
            // function template.  Could be a namespace scope function or a
            // member function.
            if (!FD->isTemplateInstantiation()) {
              // We are either TSK_Undeclared or
              // TSK_ExplicitSpecialization.
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
              //      "Cannot make wrapper for a function template "
              //      "explicit specialization which is declared "
              //      "but not defined!";
              // return 0;
              break;
            }
            const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
            if (!Pattern) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a function template"
                              "instantiation with no pattern!";
              return 0;
            }
            FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
            TemplateSpecializationKind PTSK =
                Pattern->getTemplateSpecializationKind();
            if (
                // The pattern is an ordinary member function.
                (PTK == FunctionDecl::TK_NonTemplate) ||
                // The pattern is an explicit specialization, and
                // so is not a template.
                ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                 ((PTSK == TSK_Undeclared) ||
                  (PTSK == TSK_ExplicitSpecialization)))) {
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              break;
            }
            if (!Pattern->hasBody()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a function template"
                              "instantiation with no body!";
              return 0;
            }
            if (FD->isImplicitlyInstantiable()) {
              needInstantiation = true;
            }
          } break;
          case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
            // This function is the result of instantiating or
            // specializing a  member function of a class template,
            // or a member function of a class member of a class template,
            // or a member function template of a class template, or a
            // member function template of a class member of a class
            // template where at least some part of the function is
            // dependent on a template argument.
            if (!FD->isTemplateInstantiation()) {
              // We are either TSK_Undeclared or
              // TSK_ExplicitSpecialization.
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              // llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
              //      "Cannot make wrapper for a dependent function "
              //      "template explicit specialization which is declared "
              //      "but not defined!";
              // return 0;
              break;
            }
            const FunctionDecl* Pattern = FD->getTemplateInstantiationPattern();
            if (!Pattern) {
              llvm::errs()
                  << "TClingCallFunc::make_wrapper"
                  << ":"
                  << "Cannot make wrapper for a dependent function template"
                     "instantiation with no pattern!";
              return 0;
            }
            FunctionDecl::TemplatedKind PTK = Pattern->getTemplatedKind();
            TemplateSpecializationKind PTSK =
                Pattern->getTemplateSpecializationKind();
            if (
                // The pattern is an ordinary member function.
                (PTK == FunctionDecl::TK_NonTemplate) ||
                // The pattern is an explicit specialization, and
                // so is not a template.
                ((PTK != FunctionDecl::TK_FunctionTemplate) &&
                 ((PTSK == TSK_Undeclared) ||
                  (PTSK == TSK_ExplicitSpecialization)))) {
              // Note: This might be ok, the body might be defined
              //       in a library, and all we have seen is the
              //       header file.
              break;
            }
            if (!Pattern->hasBody()) {
              llvm::errs()
                  << "TClingCallFunc::make_wrapper"
                  << ":"
                  << "Cannot make wrapper for a dependent function template"
                     "instantiation with no body!";
              return 0;
            }
            if (FD->isImplicitlyInstantiable()) {
              needInstantiation = true;
            }
          } break;
          default: {
            // Will only happen if clang implementation changes.
            // Protect ourselves in case that happens.
            llvm::errs() << "TClingCallFunc::make_wrapper" << ":" <<
                           "Unhandled template kind!";
            return 0;
          } break;
        }
        // We do not set needInstantiation to true in these cases:
        //
        // isInvalidDecl()
        // TSK_Undeclared
        // TSK_ExplicitInstantiationDefinition
        // TSK_ExplicitSpecialization && !getClassScopeSpecializationPattern()
        // TSK_ExplicitInstantiationDeclaration &&
        //    getTemplateInstantiationPattern() &&
        //    PatternDecl->hasBody() &&
        //    !PatternDecl->isInlined()
        //
        // Set it true in these cases:
        //
        // TSK_ImplicitInstantiation
        // TSK_ExplicitInstantiationDeclaration && (!getPatternDecl() ||
        //    !PatternDecl->hasBody() || PatternDecl->isInlined())
        //
      }
      if (needInstantiation) {
        clang::FunctionDecl* FDmod = const_cast<clang::FunctionDecl*>(FD);
        clang::Sema& S = I.getCI()->getSema();
        // Could trigger deserialization of decls.
#ifdef CPPINTEROP_USE_CLING
        cling::Interpreter::PushTransactionRAII RAII(&I);
#endif
        S.InstantiateFunctionDefinition(SourceLocation(), FDmod,
                                        /*Recursive=*/true,
                                        /*DefinitionRequired=*/true);
        if (!FD->isDefined(Definition)) {
          llvm::errs() << "TClingCallFunc::make_wrapper"
                       << ":"
                       << "Failed to force template instantiation!";
          return 0;
        }
      }
      if (Definition) {
        FunctionDecl::TemplatedKind TK = Definition->getTemplatedKind();
        switch (TK) {
          case FunctionDecl::TK_NonTemplate: {
            // Ordinary function, not a template specialization.
            if (Definition->isDeleted()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a deleted function!";
              return 0;
            } else if (Definition->isLateTemplateParsed()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a late template parsed "
                              "function!";
              return 0;
            }
            // else if (Definition->isDefaulted()) {
            //   // Might not have a body, but we can still use it.
            //}
            // else {
            //   // Has a body.
            //}
          } break;
          case FunctionDecl::TK_FunctionTemplate: {
            // This decl is actually a function template,
            // not a function at all.
            llvm::errs() << "TClingCallFunc::make_wrapper"
                         << ":"
                         << "Cannot make wrapper for a function template!";
            return 0;
          } break;
          case FunctionDecl::TK_MemberSpecialization: {
            // This function is the result of instantiating an ordinary
            // member function of a class template or of a member class
            // of a class template.
            if (Definition->isDeleted()) {
              llvm::errs()
                  << "TClingCallFunc::make_wrapper"
                  << ":"
                  << "Cannot make wrapper for a deleted member function "
                     "of a specialization!";
              return 0;
            } else if (Definition->isLateTemplateParsed()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a late template parsed "
                              "member function of a specialization!";
              return 0;
            }
            // else if (Definition->isDefaulted()) {
            //   // Might not have a body, but we can still use it.
            //}
            // else {
            //   // Has a body.
            //}
          } break;
          case FunctionDecl::TK_FunctionTemplateSpecialization: {
            // This function is the result of instantiating a function
            // template or possibly an explicit specialization of a
            // function template.  Could be a namespace scope function or a
            // member function.
            if (Definition->isDeleted()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a deleted function "
                              "template specialization!";
              return 0;
            } else if (Definition->isLateTemplateParsed()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a late template parsed "
                              "function template specialization!";
              return 0;
            }
            // else if (Definition->isDefaulted()) {
            //   // Might not have a body, but we can still use it.
            //}
            // else {
            //   // Has a body.
            //}
          } break;
          case FunctionDecl::TK_DependentFunctionTemplateSpecialization: {
            // This function is the result of instantiating or
            // specializing a  member function of a class template,
            // or a member function of a class member of a class template,
            // or a member function template of a class template, or a
            // member function template of a class member of a class
            // template where at least some part of the function is
            // dependent on a template argument.
            if (Definition->isDeleted()) {
              llvm::errs()
                  << "TClingCallFunc::make_wrapper"
                  << ":"
                  << "Cannot make wrapper for a deleted dependent function "
                     "template specialization!";
              return 0;
            } else if (Definition->isLateTemplateParsed()) {
              llvm::errs() << "TClingCallFunc::make_wrapper"
                           << ":"
                           << "Cannot make wrapper for a late template parsed "
                              "dependent function template specialization!";
              return 0;
            }
            // else if (Definition->isDefaulted()) {
            //   // Might not have a body, but we can still use it.
            //}
            // else {
            //   // Has a body.
            //}
          } break;
          default: {
            // Will only happen if clang implementation changes.
            // Protect ourselves in case that happens.
            llvm::errs() << "TClingCallFunc::make_wrapper"
                         << ":"
                         << "Unhandled template kind!";
            return 0;
          } break;
        }
      }
      unsigned min_args = FD->getMinRequiredArguments();
      unsigned num_params = FD->getNumParams();
      //
      //  Make the wrapper name.
      //
      {
        std::ostringstream buf;
        buf << "__cf";
        // const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
        // std::string mn;
        // fInterp->maybeMangleDeclName(ND, mn);
        // buf << '_' << mn;
        buf << '_' << gWrapperSerial++;
        wrapper_name = buf.str();
      }
      //
      //  Write the wrapper code.
      // FIXME: this should be synthesized into the AST!
      //
      int indent_level = 0;
      std::ostringstream buf;
      buf << "#pragma clang diagnostic push\n"
             "#pragma clang diagnostic ignored \"-Wformat-security\"\n"
             "__attribute__((used)) "
             "__attribute__((annotate(\"__cling__ptrcheck(off)\")))\n"
             "extern \"C\" void ";
      buf << wrapper_name;
      buf << "(void* obj, int nargs, void** args, void* ret)\n"
             "{\n";
      ++indent_level;
      if (min_args == num_params) {
        // No parameters with defaults.
        make_narg_call_with_return(I, FD, num_params, class_name, buf,
                                   indent_level);
      } else {
        // We need one function call clause compiled for every
        // possible number of arguments per call.
        for (unsigned N = min_args; N <= num_params; ++N) {
          indent(buf, indent_level);
          buf << "if (nargs == " << N << ") {\n";
          ++indent_level;
          make_narg_call_with_return(I, FD, N, class_name, buf, indent_level);
          --indent_level;
          indent(buf, indent_level);
          buf << "}\n";
        }
      }
      --indent_level;
      buf << "}\n"
             "#pragma clang diagnostic pop";
      wrapper = buf.str();
      return 1;
    }

    JitCall::GenericCall make_wrapper(compat::Interpreter& I,
                                      const FunctionDecl* FD) {
      static std::map<const FunctionDecl*, void *> gWrapperStore;

      auto R = gWrapperStore.find(FD);
      if (R != gWrapperStore.end())
        return (JitCall::GenericCall) R->second;

      std::string wrapper_name;
      std::string wrapper_code;

      if (get_wrapper_code(I, FD, wrapper_name, wrapper_code) == 0)
        return 0;

      //
      //   Compile the wrapper code.
      //
      bool withAccessControl = true;
      // We should be able to call private default constructors.
      if (auto Ctor = dyn_cast<CXXConstructorDecl>(FD))
        withAccessControl = !Ctor->isDefaultConstructor();
      void *wrapper = compile_wrapper(I, wrapper_name, wrapper_code,
                                      withAccessControl);
      if (wrapper) {
        gWrapperStore.insert(std::make_pair(FD, wrapper));
      } else {
        llvm::errs() << "TClingCallFunc::make_wrapper"
                     << ":"
                     << "Failed to compile\n"
                     << "==== SOURCE BEGIN ====\n"
                     << wrapper_code << "\n"
                     << "==== SOURCE END ====\n";
      }
      LLVM_DEBUG(dbgs() << "Compiled '" << (wrapper ? "" : "un")
                 << "successfully:\n" << wrapper_code << "'\n");
      return (JitCall::GenericCall)wrapper;
    }

    // FIXME: Sink in the code duplication from get_wrapper_code.
    static std::string PrepareTorWrapper(const Decl* D,
                                         const char* wrapper_prefix,
                                         std::string& class_name) {
      ASTContext &Context = D->getASTContext();
      PrintingPolicy Policy(Context.getPrintingPolicy());
      Policy.SuppressTagKeyword = true;
      Policy.SuppressUnwrittenScope = true;
      //
      //  Get the class or namespace name.
      //
      if (const TypeDecl *TD = dyn_cast<TypeDecl>(D)) {
        // This is a class, struct, or union member.
        // Handle the typedefs to anonymous types.
        QualType QT;
        if (const TypedefDecl *Typedef = dyn_cast<const TypedefDecl>(TD))
          QT = Typedef->getTypeSourceInfo()->getType();
        else
          QT = {TD->getTypeForDecl(), 0};
        get_type_as_string(QT, class_name, Context, Policy);
      } else if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
        // This is a namespace member.
        raw_string_ostream stream(class_name);
        ND->getNameForDiagnostic(stream, Policy, /*Qualified=*/true);
        stream.flush();
      }

      //
      //  Make the wrapper name.
      //
      string wrapper_name;
      {
        ostringstream buf;
        buf << wrapper_prefix;
        //const NamedDecl* ND = dyn_cast<NamedDecl>(FD);
        //string mn;
        //fInterp->maybeMangleDeclName(ND, mn);
        //buf << '_dtor_' << mn;
        buf << '_' << gWrapperSerial++;
        wrapper_name = buf.str();
      }

      return wrapper_name;
    }

    static JitCall::DestructorCall make_dtor_wrapper(compat::Interpreter& interp,
                                                              const Decl *D) {
      // Make a code string that follows this pattern:
      //
      // void
      // unique_wrapper_ddd(void* obj, unsigned long nary, int withFree)
      // {
      //    if (withFree) {
      //       if (!nary) {
      //          delete (ClassName*) obj;
      //       }
      //       else {
      //          delete[] (ClassName*) obj;
      //       }
      //    }
      //    else {
      //       typedef ClassName DtorName;
      //       if (!nary) {
      //          ((ClassName*)obj)->~DtorName();
      //       }
      //       else {
      //          for (unsigned long i = nary - 1; i > -1; --i) {
      //             (((ClassName*)obj)+i)->~DtorName();
      //          }
      //       }
      //    }
      // }
      //
      //--

      static map<const Decl *, void *> gDtorWrapperStore;

      auto I = gDtorWrapperStore.find(D);
      if (I != gDtorWrapperStore.end())
        return (JitCall::DestructorCall) I->second;

      //
      //  Make the wrapper name.
      //
      std::string class_name;
      string wrapper_name = PrepareTorWrapper(D, "__dtor", class_name);
      //
      //  Write the wrapper code.
      //
      int indent_level = 0;
      ostringstream buf;
      buf << "__attribute__((used)) ";
      buf << "extern \"C\" void ";
      buf << wrapper_name;
      buf << "(void* obj, unsigned long nary, int withFree)\n";
      buf << "{\n";
      //    if (withFree) {
      //       if (!nary) {
      //          delete (ClassName*) obj;
      //       }
      //       else {
      //          delete[] (ClassName*) obj;
      //       }
      //    }
      ++indent_level;
      indent(buf, indent_level);
      buf << "if (withFree) {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "if (!nary) {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "delete (" << class_name << "*) obj;\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      indent(buf, indent_level);
      buf << "else {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "delete[] (" << class_name << "*) obj;\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      //    else {
      //       typedef ClassName Nm;
      //       if (!nary) {
      //          ((Nm*)obj)->~Nm();
      //       }
      //       else {
      //          for (unsigned long i = nary - 1; i > -1; --i) {
      //             (((Nm*)obj)+i)->~Nm();
      //          }
      //       }
      //    }
      indent(buf, indent_level);
      buf << "else {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "typedef " << class_name << " Nm;\n";
      buf << "if (!nary) {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "((Nm*)obj)->~Nm();\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      indent(buf, indent_level);
      buf << "else {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "do {\n";
      ++indent_level;
      indent(buf, indent_level);
      buf << "(((Nm*)obj)+(--nary))->~Nm();\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "} while (nary);\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      --indent_level;
      indent(buf, indent_level);
      buf << "}\n";
      // End wrapper.
      --indent_level;
      buf << "}\n";
      // Done.
      string wrapper(buf.str());
      //fprintf(stderr, "%s\n", wrapper.c_str());
      //
      //  Compile the wrapper code.
      //
      void *F = compile_wrapper(interp, wrapper_name, wrapper,
                                /*withAccessControl=*/false);
      if (F) {
        gDtorWrapperStore.insert(make_pair(D, F));
      } else {
        llvm::errs() << "make_dtor_wrapper"
                     << "Failed to compile\n"
                     << "==== SOURCE BEGIN ====\n"
                     << wrapper
                     << "\n  ==== SOURCE END ====";
      }
      LLVM_DEBUG(dbgs() << "Compiled '" << (F ? "" : "un")
                 << "successfully:\n" << wrapper << "'\n");
      return (JitCall::DestructorCall)F;
    }
#undef DEBUG_TYPE
    } // namespace
      // End of JitCall Helper Functions

    CPPINTEROP_API JitCall MakeFunctionCallable(TInterp_t I,
                                                TCppConstFunction_t func) {
      const auto* D = static_cast<const clang::Decl*>(func);
      if (!D)
        return {};

      auto* interp = static_cast<compat::Interpreter*>(I);

      // FIXME: Unify with make_wrapper.
      if (const auto* Dtor = dyn_cast<CXXDestructorDecl>(D)) {
        if (auto Wrapper = make_dtor_wrapper(*interp, Dtor->getParent()))
          return {JitCall::kDestructorCall, Wrapper, Dtor};
        // FIXME: else error we failed to compile the wrapper.
        return {};
      }

      if (auto Wrapper = make_wrapper(*interp, cast<FunctionDecl>(D))) {
        return {JitCall::kGenericCall, Wrapper, cast<FunctionDecl>(D)};
      }
      // FIXME: else error we failed to compile the wrapper.
      return {};
    }

    CPPINTEROP_API JitCall MakeFunctionCallable(TCppConstFunction_t func) {
      return MakeFunctionCallable(&getInterp(), func);
    }

  namespace {
  static std::string MakeResourcesPath() {
    StringRef Dir;
#ifdef LLVM_BINARY_DIR
    Dir = LLVM_BINARY_DIR;
#else
    // Dir is bin/ or lib/, depending on where BinaryPath is.
    void *MainAddr = (void *)(intptr_t)GetExecutablePath;
    std::string BinaryPath = GetExecutablePath(/*Argv0=*/nullptr, MainAddr);

    // build/tools/clang/unittests/Interpreter/Executable -> build/
    StringRef Dir = sys::path::parent_path(BinaryPath);

    Dir = sys::path::parent_path(Dir);
    Dir = sys::path::parent_path(Dir);
    Dir = sys::path::parent_path(Dir);
    Dir = sys::path::parent_path(Dir);
    //Dir = sys::path::parent_path(Dir);
#endif // LLVM_BINARY_DIR
    return compat::MakeResourceDir(Dir);
  }
  } // namespace

  TInterp_t CreateInterpreter(const std::vector<const char*>& Args /*={}*/,
                              const std::vector<const char*>& GpuArgs /*={}*/) {
    std::string MainExecutableName =
      sys::fs::getMainExecutable(nullptr, nullptr);
    std::string ResourceDir = MakeResourcesPath();
    std::vector<const char *> ClingArgv = {"-resource-dir", ResourceDir.c_str(),
                                           "-std=c++14"};
    ClingArgv.insert(ClingArgv.begin(), MainExecutableName.c_str());
#ifdef _WIN32
    // FIXME : Workaround Sema::PushDeclContext assert on windows
    ClingArgv.push_back("-fno-delayed-template-parsing");
#endif
    ClingArgv.insert(ClingArgv.end(), Args.begin(), Args.end());
    // To keep the Interpreter creation interface between cling and clang-repl
    // to some extent compatible we should put Args and GpuArgs together. On the
    // receiving end we should check for -xcuda to know.
    if (!GpuArgs.empty()) {
      llvm::StringRef Arg0 = GpuArgs[0];
      Arg0 = Arg0.trim().ltrim('-');
      if (Arg0 != "cuda") {
        llvm::errs() << "[CreateInterpreter]: Make sure --cuda is passed as the"
                     << " first argument of the GpuArgs\n";
        return nullptr;
      }
    }
    ClingArgv.insert(ClingArgv.end(), GpuArgs.begin(), GpuArgs.end());

    // Process externally passed arguments if present.
    std::vector<std::string> ExtraArgs;
    auto EnvOpt =
        llvm::sys::Process::GetEnv("CPPINTEROP_EXTRA_INTERPRETER_ARGS");
    if (EnvOpt) {
      StringRef Env(*EnvOpt);
      while (!Env.empty()) {
        StringRef Arg;
        std::tie(Arg, Env) = Env.split(' ');
        ExtraArgs.push_back(Arg.str());
      }
    }
    std::transform(ExtraArgs.begin(), ExtraArgs.end(),
                   std::back_inserter(ClingArgv),
                   [&](const std::string& str) { return str.c_str(); });

    auto I = new compat::Interpreter(ClingArgv.size(), &ClingArgv[0]);

    // Honor -mllvm.
    //
    // FIXME: Remove this, one day.
    // This should happen AFTER plugins have been loaded!
    const CompilerInstance* Clang = I->getCI();
    if (!Clang->getFrontendOpts().LLVMArgs.empty()) {
      unsigned NumArgs = Clang->getFrontendOpts().LLVMArgs.size();
      auto Args = std::make_unique<const char*[]>(NumArgs + 2);
      Args[0] = "clang (LLVM option parsing)";
      for (unsigned i = 0; i != NumArgs; ++i)
        Args[i + 1] = Clang->getFrontendOpts().LLVMArgs[i].c_str();
      Args[NumArgs + 1] = nullptr;
      llvm::cl::ParseCommandLineOptions(NumArgs + 1, Args.get());
    }
    // FIXME: Enable this assert once we figure out how to fix the multiple
    // calls to CreateInterpreter.
    //assert(!sInterpreter && "Interpreter already set.");
    sInterpreter = I;
    return I;
  }

  TInterp_t GetInterpreter() { return sInterpreter; }

  void UseExternalInterpreter(TInterp_t I) {
    assert(!sInterpreter && "sInterpreter already in use!");
    sInterpreter = static_cast<compat::Interpreter*>(I);
    OwningSInterpreter = false;
  }

  void AddSearchPath(const char *dir, bool isUser,
                     bool prepend) {
    getInterp().getDynamicLibraryManager()->addSearchPath(dir, isUser, prepend);
  }

  const char* GetResourceDir() {
    return getInterp().getCI()->getHeaderSearchOpts().ResourceDir.c_str();
  }

  ///\returns 0 on success.
  static bool exec(const char* cmd, std::vector<std::string>& outputs) {
#define DEBUG_TYPE "exec"

    std::array<char, 256> buffer;
    struct file_deleter {
      void operator()(FILE* fp) { pclose(fp); }
    };
    std::unique_ptr<FILE, file_deleter> pipe{popen(cmd, "r")};
    LLVM_DEBUG(dbgs() << "Executing command '" << cmd << "'\n");

    if (!pipe) {
      LLVM_DEBUG(dbgs() << "Execute failed!\n");
      perror("exec: ");
      return false;
    }

    LLVM_DEBUG(dbgs() << "Execute returned:\n");
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get())) {
      LLVM_DEBUG(dbgs() << buffer.data());
      llvm::StringRef trimmed = buffer.data();
      outputs.push_back(trimmed.trim().str());
    }

#undef DEBUG_TYPE

    return true;
  }

  std::string DetectResourceDir(const char* ClangBinaryName /* = clang */) {
    std::string cmd = std::string(ClangBinaryName) + " -print-resource-dir";
    std::vector<std::string> outs;
    exec(cmd.c_str(), outs);
    if (outs.empty() || outs.size() > 1)
      return "";

    std::string detected_resource_dir = outs.back();

    std::string version =
#if CLANG_VERSION_MAJOR < 16
        CLANG_VERSION_STRING;
#else
        CLANG_VERSION_MAJOR_STRING;
#endif
    // We need to check if the detected resource directory is compatible.
    if (llvm::sys::path::filename(detected_resource_dir) != version)
      return "";

    return detected_resource_dir;
  }

  void DetectSystemCompilerIncludePaths(std::vector<std::string>& Paths,
                                        const char* CompilerName /*= "c++"*/) {
    std::string cmd = "LC_ALL=C ";
    cmd += CompilerName;
    cmd += " -xc++ -E -v /dev/null 2>&1 | sed -n -e '/^.include/,${' -e '/^ "
           "\\/.*/p' -e '}'";
    std::vector<std::string> outs;
    exec(cmd.c_str(), Paths);
  }

  void AddIncludePath(const char *dir) {
    getInterp().AddIncludePath(dir);
  }

  void GetIncludePaths(std::vector<std::string>& IncludePaths, bool withSystem,
                       bool withFlags) {
    llvm::SmallVector<std::string> paths(1);
    getInterp().GetIncludePaths(paths, withSystem, withFlags);
    for (auto& i : paths)
      IncludePaths.push_back(i);
  }

  namespace {

  class clangSilent {
  public:
    clangSilent(clang::DiagnosticsEngine &diag) : fDiagEngine(diag) {
      fOldDiagValue = fDiagEngine.getSuppressAllDiagnostics();
      fDiagEngine.setSuppressAllDiagnostics(true);
    }

    ~clangSilent() { fDiagEngine.setSuppressAllDiagnostics(fOldDiagValue); }

  protected:
    clang::DiagnosticsEngine &fDiagEngine;
    bool fOldDiagValue;
  };
  } // namespace

  int Declare(const char* code, bool silent) {
    auto& I = getInterp();

    if (silent) {
      clangSilent diagSuppr(I.getSema().getDiagnostics());
      return I.declare(code);
    }

    return I.declare(code);
  }

  int Process(const char *code) {
    return getInterp().process(code);
  }

  intptr_t Evaluate(const char *code,
                    bool *HadError/*=nullptr*/) {
#ifdef CPPINTEROP_USE_CLING
    cling::Value V;
#else
    clang::Value V;
#endif // CPPINTEROP_USE_CLING

    if (HadError)
      *HadError = false;

    auto res = getInterp().evaluate(code, V);
    if (res != 0) { // 0 is success
      if (HadError)
        *HadError = true;
      // FIXME: Make this return llvm::Expected
      return ~0UL;
    }

    return compat::convertTo<intptr_t>(V);
  }

  std::string LookupLibrary(const char* lib_name) {
    return getInterp().getDynamicLibraryManager()->lookupLibrary(lib_name);
  }

  bool LoadLibrary(const char* lib_stem, bool lookup) {
    compat::Interpreter::CompilationResult res =
        getInterp().loadLibrary(lib_stem, lookup);

    return res == compat::Interpreter::kSuccess;
  }

  void UnloadLibrary(const char* lib_stem) {
    getInterp().getDynamicLibraryManager()->unloadLibrary(lib_stem);
  }

  std::string SearchLibrariesForSymbol(const char* mangled_name,
                                       bool search_system /*true*/) {
    auto* DLM = getInterp().getDynamicLibraryManager();
    return DLM->searchLibrariesForSymbol(mangled_name, search_system);
  }

  bool InsertOrReplaceJitSymbol(compat::Interpreter& I,
                                const char* linker_mangled_name,
                                uint64_t address) {
    // FIXME: This approach is problematic since we could replace a symbol
    // whose address was already taken by clients.
    //
    // A safer approach would be to define our symbol replacements early in the
    // bootstrap process like:
    // auto J = LLJITBuilder().create();
    // if (!J)
    //   return Err;
    //
    // if (Jupyter) {
    //   llvm::orc::SymbolMap Overrides;
    //   Overrides[J->mangleAndIntern("printf")] =
    //     { ExecutorAddr::fromPtr(&printf), JITSymbolFlags::Exported };
    //   Overrides[...] =
    //     { ... };
    //   if (auto Err =
    //   J->getProcessSymbolsJITDylib().define(absoluteSymbols(std::move(Overrides)))
    //     return Err;
    // }

    // FIXME: If we still want to do symbol replacement we should use the
    // ReplacementManager which is available in llvm 18.
    using namespace llvm;
    using namespace llvm::orc;

    auto Symbol = compat::getSymbolAddress(I, linker_mangled_name);
    llvm::orc::LLJIT& Jit = *compat::getExecutionEngine(I);
    llvm::orc::ExecutionSession& ES = Jit.getExecutionSession();
#if CLANG_VERSION_MAJOR < 17
    JITDylib& DyLib = Jit.getMainJITDylib();
#else
    JITDylib& DyLib = *Jit.getProcessSymbolsJITDylib().get();
#endif // CLANG_VERSION_MAJOR

    if (Error Err = Symbol.takeError()) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "[InsertOrReplaceJitSymbol] error: ");
#define DEBUG_TYPE "orc"
      LLVM_DEBUG(ES.dump(dbgs()));
#undef DEBUG_TYPE
      return true;
    }

    // Nothing to define, we are redefining the same function.
    if (*Symbol && *Symbol == address) {
      errs() << "[InsertOrReplaceJitSymbol] warning: redefining '"
             << linker_mangled_name << "' with the same address\n";
      return true;
    }

    // Let's inject it.
    llvm::orc::SymbolMap InjectedSymbols;
    auto& DL = compat::getExecutionEngine(I)->getDataLayout();
    char GlobalPrefix = DL.getGlobalPrefix();
    std::string tmp(linker_mangled_name);
    if (GlobalPrefix != '\0') {
      tmp = std::string(1, GlobalPrefix) + tmp;
    }
    auto Name = ES.intern(tmp);
    InjectedSymbols[Name] =
#if CLANG_VERSION_MAJOR < 17
        JITEvaluatedSymbol(address,
#else
        ExecutorSymbolDef(ExecutorAddr(address),
#endif // CLANG_VERSION_MAJOR < 17
                           JITSymbolFlags::Exported);

    // We want to replace a symbol with a custom provided one.
    if (Symbol && address)
      // The symbol be in the DyLib or in-process.
      if (auto Err = DyLib.remove({Name})) {
        logAllUnhandledErrors(std::move(Err), errs(),
                              "[InsertOrReplaceJitSymbol] error: ");
        return true;
      }

    if (Error Err = DyLib.define(absoluteSymbols(InjectedSymbols))) {
      logAllUnhandledErrors(std::move(Err), errs(),
                            "[InsertOrReplaceJitSymbol] error: ");
      return true;
    }

    return false;
  }

  bool InsertOrReplaceJitSymbol(const char* linker_mangled_name,
                                uint64_t address) {
    return InsertOrReplaceJitSymbol(getInterp(), linker_mangled_name, address);
  }

  std::string ObjToString(const char *type, void *obj) {
    return getInterp().toString(type, obj);
  }

  static Decl* InstantiateTemplate(TemplateDecl* TemplateD,
                                   TemplateArgumentListInfo& TLI, Sema& S) {
    // This is not right but we don't have a lot of options to choose from as a
    // template instantiation requires a valid source location.
    SourceLocation fakeLoc = GetValidSLoc(S);
    if (auto* FunctionTemplate = dyn_cast<FunctionTemplateDecl>(TemplateD)) {
      FunctionDecl* Specialization = nullptr;
      clang::sema::TemplateDeductionInfo Info(fakeLoc);
      Template_Deduction_Result Result = S.DeduceTemplateArguments(
          FunctionTemplate, &TLI, Specialization, Info,
          /*IsAddressOfFunction*/ true);
      if (Result != Template_Deduction_Result_Success) {
        // FIXME: Diagnose what happened.
        (void)Result;
      }
      return Specialization;
    }

    if (auto* VarTemplate = dyn_cast<VarTemplateDecl>(TemplateD)) {
      DeclResult R = S.CheckVarTemplateId(VarTemplate, fakeLoc, fakeLoc, TLI);
      if (R.isInvalid()) {
        // FIXME: Diagnose
      }
      return R.get();
    }

    // This will instantiate tape<T> type and return it.
    SourceLocation noLoc;
    QualType TT = S.CheckTemplateIdType(TemplateName(TemplateD), noLoc, TLI);

    // Perhaps we can extract this into a new interface.
    S.RequireCompleteType(fakeLoc, TT, diag::err_tentative_def_incomplete_type);
    return GetScopeFromType(TT);

    // ASTContext &C = S.getASTContext();
    // // Get clad namespace and its identifier clad::.
    // CXXScopeSpec CSS;
    // CSS.Extend(C, GetCladNamespace(), noLoc, noLoc);
    // NestedNameSpecifier* NS = CSS.getScopeRep();

    // // Create elaborated type with namespace specifier,
    // // i.e. class<T> -> clad::class<T>
    // return C.getElaboratedType(ETK_None, NS, TT);
  }

  Decl* InstantiateTemplate(TemplateDecl* TemplateD,
                            ArrayRef<TemplateArgument> TemplateArgs, Sema& S) {
    // Create a list of template arguments.
    TemplateArgumentListInfo TLI{};
    for (auto TA : TemplateArgs)
      TLI.addArgument(S.getTrivialTemplateArgumentLoc(TA,QualType(),
                                                      SourceLocation()));

    return InstantiateTemplate(TemplateD, TLI, S);
  }

  TCppScope_t InstantiateTemplate(compat::Interpreter& I, TCppScope_t tmpl,
                                  const TemplateArgInfo* template_args,
                                  size_t template_args_size) {
    auto& S = I.getSema();
    auto& C = S.getASTContext();

    llvm::SmallVector<TemplateArgument> TemplateArgs;
    TemplateArgs.reserve(template_args_size);
    for (size_t i = 0; i < template_args_size; ++i) {
      QualType ArgTy = QualType::getFromOpaquePtr(template_args[i].m_Type);
      if (template_args[i].m_IntegralValue) {
        // We have a non-type template parameter. Create an integral value from
        // the string representation.
        auto Res = llvm::APSInt(template_args[i].m_IntegralValue);
        Res = Res.extOrTrunc(C.getIntWidth(ArgTy));
        TemplateArgs.push_back(TemplateArgument(C, Res, ArgTy));
      } else {
        TemplateArgs.push_back(ArgTy);
      }
    }

    TemplateDecl* TmplD = static_cast<TemplateDecl*>(tmpl);

    // We will create a new decl, push a transaction.
#ifdef CPPINTEROP_USE_CLING
    cling::Interpreter::PushTransactionRAII RAII(&I);
#endif
    return InstantiateTemplate(TmplD, TemplateArgs, S);
  }

  TCppScope_t InstantiateTemplate(TCppScope_t tmpl,
                                  const TemplateArgInfo* template_args,
                                  size_t template_args_size) {
    return InstantiateTemplate(getInterp(), tmpl, template_args,
                               template_args_size);
  }

  void GetClassTemplateInstantiationArgs(TCppScope_t templ_instance,
                                         std::vector<TemplateArgInfo> &args) {
    auto* CTSD = static_cast<ClassTemplateSpecializationDecl*>(templ_instance);
    for(const auto& TA : CTSD->getTemplateInstantiationArgs().asArray()) {
      switch (TA.getKind()) {
      default:
        assert(0 && "Not yet supported!");
        break;
      case TemplateArgument::Pack:
        for (auto SubTA : TA.pack_elements())
          args.push_back({SubTA.getAsType().getAsOpaquePtr()});
        break;
      case TemplateArgument::Integral:
        // FIXME: Support this case where the problem is where we provide the
        // storage for the m_IntegralValue.
        //llvm::APSInt Val = TA.getAsIntegral();
        //args.push_back({TA.getIntegralType(), TA.getAsIntegral()})
        //break;
      case TemplateArgument::Type:
        args.push_back({TA.getAsType().getAsOpaquePtr()});
      }
    }
  }

  TCppFunction_t
  InstantiateTemplateFunctionFromString(const char* function_template) {
    // FIXME: Drop this interface and replace it with the proper overload
    // resolution handling and template instantiation selection.

    // Try to force template instantiation and overload resolution.
    static unsigned long long var_count = 0;
    std::string id = "__Cppyy_GetMethTmpl_" + std::to_string(var_count++);
    std::string instance = "auto " + id + " = " + function_template + ";\n";

    if (!Cpp::Declare(instance.c_str(), /*silent=*/false)) {
      VarDecl* VD = (VarDecl*)Cpp::GetNamed(id, 0);
      DeclRefExpr* DRE = (DeclRefExpr*)VD->getInit()->IgnoreImpCasts();
      return DRE->getDecl();
    }
    return nullptr;
  }

  void GetAllCppNames(TCppScope_t scope, std::set<std::string>& names) {
    auto *D = (clang::Decl *)scope;
    clang::DeclContext *DC;
    clang::DeclContext::decl_iterator decl;

    if (auto *TD = dyn_cast_or_null<TagDecl>(D)) {
      DC = clang::TagDecl::castToDeclContext(TD);
      decl = DC->decls_begin();
      decl++;
    } else if (auto *ND = dyn_cast_or_null<NamespaceDecl>(D)) {
      DC = clang::NamespaceDecl::castToDeclContext(ND);
      decl = DC->decls_begin();
    } else if (auto *TUD = dyn_cast_or_null<TranslationUnitDecl>(D)) {
      DC = clang::TranslationUnitDecl::castToDeclContext(TUD);
      decl = DC->decls_begin();
    } else {
      return;
    }

    for (/* decl set above */; decl != DC->decls_end(); decl++) {
      if (auto *ND = llvm::dyn_cast_or_null<NamedDecl>(*decl)) {
        names.insert(ND->getNameAsString());
      }
    }
  }

  void GetEnums(TCppScope_t scope, std::vector<std::string>& Result) {
    auto* D = static_cast<clang::Decl*>(scope);

    if (!llvm::isa_and_nonnull<clang::DeclContext>(D))
      return;

    auto* DC = llvm::dyn_cast<clang::DeclContext>(D);

    llvm::SmallVector<clang::DeclContext*, 4> DCs;
    DC->collectAllContexts(DCs);

    // FIXME: We should use a lookup based approach instead of brute force
    for (auto* DC : DCs) {
      for (auto decl = DC->decls_begin(); decl != DC->decls_end(); decl++) {
        if (auto* ND = llvm::dyn_cast_or_null<EnumDecl>(*decl)) {
          Result.push_back(ND->getNameAsString());
        }
      }
    }
  }

  // FIXME: On the CPyCppyy side the receiver is of type
  //        vector<long int> instead of vector<TCppIndex_t>
  std::vector<long int> GetDimensions(TCppType_t type)
  {
    QualType Qual = QualType::getFromOpaquePtr(type);
    if (Qual.isNull())
      return {};
    Qual = Qual.getCanonicalType();
    std::vector<long int> dims;
    if (Qual->isArrayType())
    {
      const clang::ArrayType *ArrayType = dyn_cast<clang::ArrayType>(Qual.getTypePtr());
      while (ArrayType)
      {
        if (const auto *CAT = dyn_cast_or_null<ConstantArrayType>(ArrayType)) {
          llvm::APSInt Size(CAT->getSize());
          long int ArraySize = Size.getLimitedValue();
          dims.push_back(ArraySize);
        } else /* VariableArrayType, DependentSizedArrayType, IncompleteArrayType */ {
          dims.push_back(DimensionValue::UNKNOWN_SIZE);
        }
        ArrayType = ArrayType->getElementType()->getAsArrayTypeUnsafe();
      }
      return dims;
    }
    return dims;
  }

  bool IsTypeDerivedFrom(TCppType_t derived, TCppType_t base)
  {
    auto &S = getSema();
    auto fakeLoc = GetValidSLoc(S);
    auto derivedType = clang::QualType::getFromOpaquePtr(derived);
    auto baseType = clang::QualType::getFromOpaquePtr(base);

#ifdef CPPINTEROP_USE_CLING
    cling::Interpreter::PushTransactionRAII RAII(&getInterp());
#endif
    return S.IsDerivedFrom(fakeLoc,derivedType,baseType);
  }

  std::string GetFunctionArgDefault(TCppFunction_t func,
                                    TCppIndex_t param_index) {
    auto *D = (clang::Decl *)func;
    clang::ParmVarDecl* PI = nullptr;

    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D))
      PI = FD->getParamDecl(param_index);

    else if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
      PI = (FD->getTemplatedDecl())->getParamDecl(param_index);

    if (PI->hasDefaultArg())
    {
      std::string Result;
      llvm::raw_string_ostream OS(Result);
      Expr *DefaultArgExpr = const_cast<Expr *>(PI->getDefaultArg());
      DefaultArgExpr->printPretty(OS, nullptr, PrintingPolicy(LangOptions()));

      // FIXME: Floats are printed in clang with the precision of their underlying representation
      // and not as written. This is a deficiency in the printing mechanism of clang which we require
      // extra work to mitigate. For example float PI = 3.14 is printed as 3.1400000000000001
      if (PI->getType()->isFloatingType())
      {
        if (!Result.empty() && Result.back() == '.')
          return Result;
        auto DefaultArgValue = std::stod(Result);
        std::ostringstream oss;
        oss << DefaultArgValue;
        Result = oss.str();
      }
      return Result;
    }
    return "";
  }

  bool IsConstMethod(TCppFunction_t method)
  {
    if (!method)
      return false;

    auto *D = (clang::Decl *)method;
    if (auto *func = dyn_cast<CXXMethodDecl>(D))
       return func->getMethodQualifiers().hasConst();

    return false;
  }

  std::string GetFunctionArgName(TCppFunction_t func, TCppIndex_t param_index)
  {
    auto *D = (clang::Decl *)func;
    clang::ParmVarDecl* PI = nullptr;

    if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D))
      PI = FD->getParamDecl(param_index);
    else if (auto* FD = llvm::dyn_cast_or_null<clang::FunctionTemplateDecl>(D))
      PI = (FD->getTemplatedDecl())->getParamDecl(param_index);

    return PI->getNameAsString();
  }

  OperatorArity GetOperatorArity(TCppFunction_t op) {
    Decl* D = static_cast<Decl*>(op);
    if (auto* FD = llvm::dyn_cast<FunctionDecl>(D)) {
      if (FD->isOverloadedOperator()) {
        switch (FD->getOverloadedOperator()) {
#define OVERLOADED_OPERATOR(Name, Spelling, Token, Unary, Binary,            \
                              MemberOnly)                                      \
    case OO_##Name:                                                            \
      if ((Unary) && (Binary))                                                 \
        return kBoth;                                                          \
      if (Unary)                                                               \
        return kUnary;                                                         \
      if (Binary)                                                              \
        return kBinary;                                                        \
      break;
#include "clang/Basic/OperatorKinds.def"
        default:
          break;
        }
      }
    }
    return (OperatorArity)~0U;
  }

  void GetOperator(TCppScope_t scope, Operator op,
                   std::vector<TCppFunction_t>& operators, OperatorArity kind) {
    Decl* D = static_cast<Decl*>(scope);
    if (auto* DC = llvm::dyn_cast_or_null<DeclContext>(D)) {
      ASTContext& C = getSema().getASTContext();
      DeclContextLookupResult Result =
          DC->lookup(C.DeclarationNames.getCXXOperatorName(
              (clang::OverloadedOperatorKind)op));

      for (auto* i : Result) {
        if (kind & GetOperatorArity(i))
          operators.push_back(i);
      }
    }
  }

  TCppObject_t Allocate(TCppScope_t scope) {
    return (TCppObject_t)::operator new(Cpp::SizeOf(scope));
  }

  void Deallocate(TCppScope_t scope, TCppObject_t address) {
    ::operator delete(address);
  }

  // FIXME: Add optional arguments to the operator new.
  TCppObject_t Construct(compat::Interpreter& interp, TCppScope_t scope,
                         void* arena /*=nullptr*/) {
    auto* Class = (Decl*) scope;
    // FIXME: Diagnose.
    if (!HasDefaultConstructor(Class))
      return nullptr;

    auto* const Ctor = GetDefaultConstructor(Class);
    if (JitCall JC = MakeFunctionCallable(&interp, Ctor)) {
      if (arena) {
        JC.Invoke(&arena, {}, (void*)~0); // Tell Invoke to use placement new.
        return arena;
      }

      void *obj = nullptr;
      JC.Invoke(&obj);
      return obj;
    }
    return nullptr;
  }

  TCppObject_t Construct(TCppScope_t scope, void* arena /*=nullptr*/) {
    return Construct(getInterp(), scope, arena);
  }

  void Destruct(compat::Interpreter& interp, TCppObject_t This, Decl* Class,
                bool withFree) {
    if (auto wrapper = make_dtor_wrapper(interp, Class)) {
      (*wrapper)(This, /*nary=*/0, withFree);
      return;
    }
    // FIXME: Diagnose.
  }

  void Destruct(TCppObject_t This, TCppScope_t scope, bool withFree /*=true*/) {
    auto* Class = static_cast<Decl*>(scope);
    Destruct(getInterp(), This, Class, withFree);
  }

  class StreamCaptureInfo {
    struct file_deleter {
      void operator()(FILE* fp) { pclose(fp); }
    };
    std::unique_ptr<FILE, file_deleter> m_TempFile;
    int m_FD = -1;
    int m_DupFD = -1;

  public:
#ifdef _MSC_VER
    StreamCaptureInfo(int FD)
        : m_TempFile{[]() {
            FILE* stream = nullptr;
            errno_t err;
            err = tmpfile_s(&stream);
            if (err)
              printf("Cannot create temporary file!\n");
            return stream;
          }()},
          m_FD(FD) {
#else
    StreamCaptureInfo(int FD) : m_TempFile{tmpfile()}, m_FD(FD) {
#endif
      if (!m_TempFile) {
        perror("StreamCaptureInfo: Unable to create temp file");
        return;
      }

      m_DupFD = dup(FD);

      // Flush now or can drop the buffer when dup2 is called with Fd later.
      // This seems only neccessary when piping stdout or stderr, but do it
      // for ttys to avoid over complicated code for minimal benefit.
      ::fflush(FD == STDOUT_FILENO ? stdout : stderr);
      if (dup2(fileno(m_TempFile.get()), FD) < 0)
        perror("StreamCaptureInfo:");
    }
    StreamCaptureInfo(const StreamCaptureInfo&) = delete;
    StreamCaptureInfo& operator=(const StreamCaptureInfo&) = delete;
    StreamCaptureInfo(StreamCaptureInfo&&) = delete;
    StreamCaptureInfo& operator=(StreamCaptureInfo&&) = delete;

    ~StreamCaptureInfo() {
      assert(m_DupFD == -1 && "Captured output not used?");
    }

    std::string GetCapturedString() {
      assert(m_DupFD != -1 && "Multiple calls to GetCapturedString");

      fflush(nullptr);
      if (dup2(m_DupFD, m_FD) < 0)
        perror("StreamCaptureInfo:");
      // Go to the end of the file.
      if (fseek(m_TempFile.get(), 0L, SEEK_END) != 0)
        perror("StreamCaptureInfo:");

      // Get the size of the file.
      long bufsize = ftell(m_TempFile.get());
      if (bufsize == -1)
        perror("StreamCaptureInfo:");

      // Allocate our buffer to that size.
      std::unique_ptr<char[]> content(new char[bufsize + 1]);

      // Go back to the start of the file.
      if (fseek(m_TempFile.get(), 0L, SEEK_SET) != 0)
        perror("StreamCaptureInfo:");

      // Read the entire file into memory.
      size_t newLen =
          fread(content.get(), sizeof(char), bufsize, m_TempFile.get());
      if (ferror(m_TempFile.get()) != 0)
        fputs("Error reading file", stderr);
      else
        content[newLen++] = '\0'; // Just to be safe.

      std::string result = content.get();
      close(m_DupFD);
      m_DupFD = -1;
      return result;
    }
  };

  static std::stack<StreamCaptureInfo>& GetRedirectionStack() {
    static std::stack<StreamCaptureInfo> sRedirectionStack;
    return sRedirectionStack;
  }

  void BeginStdStreamCapture(CaptureStreamKind fd_kind) {
    GetRedirectionStack().emplace((int)fd_kind);
  }

  std::string EndStdStreamCapture() {
    assert(GetRedirectionStack().size());
    StreamCaptureInfo& SCI = GetRedirectionStack().top();
    std::string result = SCI.GetCapturedString();
    GetRedirectionStack().pop();
    return result;
  }

  void CodeComplete(std::vector<std::string>& Results, const char* code,
                    unsigned complete_line /* = 1U */,
                    unsigned complete_column /* = 1U */) {
    compat::codeComplete(Results, getInterp(), code, complete_line,
                         complete_column);
  }

  } // end namespace Cpp
