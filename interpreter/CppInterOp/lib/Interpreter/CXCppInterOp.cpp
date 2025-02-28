#include "clang-c/CXCppInterOp.h"
#include "Compatibility.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/Type.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/CppInterOp.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Casting.h"
#include <cstring>
#include <iterator>
#include "clang-c/CXString.h"

// copied and tweaked from libclang
namespace clang {

CXCursorKind cxcursor_getCursorKindForDecl(const Decl* D) {
  if (!D)
    return CXCursor_UnexposedDecl;

  switch (D->getKind()) {
  case Decl::Enum:
    return CXCursor_EnumDecl;
  case Decl::EnumConstant:
    return CXCursor_EnumConstantDecl;
  case Decl::Field:
    return CXCursor_FieldDecl;
  case Decl::Function:
    return CXCursor_FunctionDecl;
  case Decl::CXXMethod:
    return CXCursor_CXXMethod;
  case Decl::CXXConstructor:
    return CXCursor_Constructor;
  case Decl::CXXDestructor:
    return CXCursor_Destructor;
  case Decl::CXXConversion:
    return CXCursor_ConversionFunction;
  case Decl::ParmVar:
    return CXCursor_ParmDecl;
  case Decl::Typedef:
    return CXCursor_TypedefDecl;
  case Decl::TypeAlias:
    return CXCursor_TypeAliasDecl;
  case Decl::TypeAliasTemplate:
    return CXCursor_TypeAliasTemplateDecl;
  case Decl::Var:
    return CXCursor_VarDecl;
  case Decl::Namespace:
    return CXCursor_Namespace;
  case Decl::NamespaceAlias:
    return CXCursor_NamespaceAlias;
  case Decl::TemplateTypeParm:
    return CXCursor_TemplateTypeParameter;
  case Decl::NonTypeTemplateParm:
    return CXCursor_NonTypeTemplateParameter;
  case Decl::TemplateTemplateParm:
    return CXCursor_TemplateTemplateParameter;
  case Decl::FunctionTemplate:
    return CXCursor_FunctionTemplate;
  case Decl::ClassTemplate:
    return CXCursor_ClassTemplate;
  case Decl::AccessSpec:
    return CXCursor_CXXAccessSpecifier;
  case Decl::ClassTemplatePartialSpecialization:
    return CXCursor_ClassTemplatePartialSpecialization;
  case Decl::UsingDirective:
    return CXCursor_UsingDirective;
  case Decl::StaticAssert:
    return CXCursor_StaticAssert;
  case Decl::Friend:
    return CXCursor_FriendDecl;
  case Decl::TranslationUnit:
    return CXCursor_TranslationUnit;

  case Decl::Using:
  case Decl::UnresolvedUsingValue:
  case Decl::UnresolvedUsingTypename:
    return CXCursor_UsingDeclaration;

  case Decl::UsingEnum:
    return CXCursor_EnumDecl;

  default:
    if (const auto* TD = dyn_cast<TagDecl>(D)) {
      switch (TD->getTagKind()) {
#if CLANG_VERSION_MAJOR >= 18
      case TagTypeKind::Interface: // fall through
      case TagTypeKind::Struct:
        return CXCursor_StructDecl;
      case TagTypeKind::Class:
        return CXCursor_ClassDecl;
      case TagTypeKind::Union:
        return CXCursor_UnionDecl;
      case TagTypeKind::Enum:
        return CXCursor_EnumDecl;
#else
      case TagTypeKind::TTK_Interface: // fall through
      case TagTypeKind::TTK_Struct:
        return CXCursor_StructDecl;
      case TagTypeKind::TTK_Class:
        return CXCursor_ClassDecl;
      case TagTypeKind::TTK_Union:
        return CXCursor_UnionDecl;
      case TagTypeKind::TTK_Enum:
        return CXCursor_EnumDecl;
#endif
      }
    }
  }

  return CXCursor_UnexposedDecl;
}

CXTypeKind cxtype_GetBuiltinTypeKind(const BuiltinType* BT) {
#define BTCASE(K)                                                              \
  case BuiltinType::K:                                                         \
    return CXType_##K
  switch (BT->getKind()) {
    BTCASE(Void);
    BTCASE(Bool);
    BTCASE(Char_U);
    BTCASE(UChar);
    BTCASE(Char16);
    BTCASE(Char32);
    BTCASE(UShort);
    BTCASE(UInt);
    BTCASE(ULong);
    BTCASE(ULongLong);
    BTCASE(UInt128);
    BTCASE(Char_S);
    BTCASE(SChar);
  case BuiltinType::WChar_S:
    return CXType_WChar;
  case BuiltinType::WChar_U:
    return CXType_WChar;
    BTCASE(Short);
    BTCASE(Int);
    BTCASE(Long);
    BTCASE(LongLong);
    BTCASE(Int128);
    BTCASE(Half);
    BTCASE(Float);
    BTCASE(Double);
    BTCASE(LongDouble);
    BTCASE(ShortAccum);
    BTCASE(Accum);
    BTCASE(LongAccum);
    BTCASE(UShortAccum);
    BTCASE(UAccum);
    BTCASE(ULongAccum);
    BTCASE(Float16);
    BTCASE(Float128);
    BTCASE(NullPtr);
  default:
    return CXType_Unexposed;
  }
#undef BTCASE
}

CXTypeKind cxtype_GetTypeKind(QualType T) {
  const Type* TP = T.getTypePtrOrNull();
  if (!TP)
    return CXType_Invalid;

#define TKCASE(K)                                                              \
  case Type::K:                                                                \
    return CXType_##K
  switch (TP->getTypeClass()) {
  case Type::Builtin:
    return cxtype_GetBuiltinTypeKind(cast<BuiltinType>(TP));
    TKCASE(Complex);
    TKCASE(Pointer);
    TKCASE(BlockPointer);
    TKCASE(LValueReference);
    TKCASE(RValueReference);
    TKCASE(Record);
    TKCASE(Enum);
    TKCASE(Typedef);
    TKCASE(ObjCInterface);
    TKCASE(ObjCObject);
    TKCASE(ObjCObjectPointer);
    TKCASE(ObjCTypeParam);
    TKCASE(FunctionNoProto);
    TKCASE(FunctionProto);
    TKCASE(ConstantArray);
    TKCASE(IncompleteArray);
    TKCASE(VariableArray);
    TKCASE(DependentSizedArray);
    TKCASE(Vector);
    TKCASE(ExtVector);
    TKCASE(MemberPointer);
    TKCASE(Auto);
    TKCASE(Elaborated);
    TKCASE(Pipe);
    TKCASE(Attributed);
#if CLANG_VERSION_MAJOR >= 16
    TKCASE(BTFTagAttributed);
#endif
    TKCASE(Atomic);
  default:
    return CXType_Unexposed;
  }
#undef TKCASE
}

// FIXME: merge with cxcursor and cxtype in the future
namespace cxscope {

CXScope MakeCXScope(const clang::Decl* D, const CXInterpreterImpl* I,
                    SourceRange RegionOfInterest = SourceRange(),
                    bool FirstInDeclGroup = true) {
  assert(D && I && "Invalid arguments!");

  CXCursorKind K = cxcursor_getCursorKindForDecl(D);

  CXScope S = {K, 0, {D, (void*)(intptr_t)(FirstInDeclGroup ? 1 : 0), I}};
  return S;
}

CXQualType MakeCXQualType(const clang::QualType Ty, CXInterpreterImpl* I) {
  CXTypeKind TK = CXType_Invalid;
  TK = cxtype_GetTypeKind(Ty);

  CXQualType CT = {TK,
                   {TK == CXType_Invalid ? nullptr : Ty.getAsOpaquePtr(),
                    static_cast<void*>(I)}};
  return CT;
}

} // namespace cxscope

} // namespace clang

CXString makeCXString(const std::string& S) {
  CXString Str;
  if (S.empty()) {
    Str.data = "";
    Str.private_flags = 0; // CXS_Unmanaged
  } else {
    Str.data = strdup(S.c_str());
    Str.private_flags = 1; // CXS_Malloc
  }
  return Str;
}

CXStringSet* makeCXStringSet(const std::vector<std::string>& Strs) {
  auto* Set = new CXStringSet; // NOLINT(*-owning-memory)
  Set->Count = Strs.size();
  Set->Strings = new CXString[Set->Count]; // NOLINT(*-owning-memory)
  for (auto En : llvm::enumerate(Strs)) {
    Set->Strings[En.index()] = makeCXString(En.value());
  }
  return Set;
}

struct CXInterpreterImpl {
  std::unique_ptr<compat::Interpreter> Interp;
  // FIXME: find a way to merge this with libclang's CXTranslationUnit
  // std::unique_ptr<CXTranslationUnitImpl> TU;
};

static inline compat::Interpreter* getInterpreter(const CXInterpreterImpl* I) {
  assert(I && "Invalid interpreter");
  return I->Interp.get();
}

CXInterpreter clang_createInterpreter(const char* const* argv, int argc) {
  auto* I = new CXInterpreterImpl(); // NOLINT(*-owning-memory)
  I->Interp = std::make_unique<compat::Interpreter>(argc, argv);
  // create a bridge between CXTranslationUnit and clang::Interpreter
  // auto AU = std::make_unique<ASTUnit>(false);
  // AU->FileMgr = I->Interp->getCompilerInstance().getFileManager();
  // AU->SourceMgr = I->Interp->getCompilerInstance().getSourceManager();
  // AU->PP = I->Interp->getCompilerInstance().getPreprocessor();
  // AU->Ctx = &I->Interp->getSema().getASTContext();
  // I->TU.reset(MakeCXTranslationUnit(static_cast<CIndexer*>(clang_createIndex(0,
  // 0)), AU));
  return I;
}

CXInterpreter clang_createInterpreterFromRawPtr(TInterp_t I) {
  auto* II = new CXInterpreterImpl(); // NOLINT(*-owning-memory)
  II->Interp.reset(static_cast<compat::Interpreter*>(I)); // NOLINT(*-cast)
  return II;
}

void* clang_Interpreter_getClangInterpreter(CXInterpreter I) {
#ifdef CPPINTEROP_USE_CLING
  return nullptr;
#else
  auto* interp = getInterpreter(I);
  auto* clInterp = &static_cast<clang::Interpreter&>(*interp);
  return clInterp;
#endif // CPPINTEROP_USE_CLING
}

TInterp_t clang_Interpreter_takeInterpreterAsPtr(CXInterpreter I) {
  return static_cast<CXInterpreterImpl*>(I)->Interp.release();
}

enum CXErrorCode clang_Interpreter_undo(CXInterpreter I, unsigned int N) {
#ifdef CPPINTEROP_USE_CLING
  return CXError_Failure;
#else
  return getInterpreter(I)->Undo(N) ? CXError_Failure : CXError_Success;
#endif // CPPINTEROP_USE_CLING
}

void clang_Interpreter_dispose(CXInterpreter I) {
  delete I; // NOLINT(*-owning-memory)
}

void clang_Interpreter_addSearchPath(CXInterpreter I, const char* dir,
                                     bool isUser, bool prepend) {
  auto* interp = getInterpreter(I);
  interp->getDynamicLibraryManager()->addSearchPath(dir, isUser, prepend);
}

void clang_Interpreter_addIncludePath(CXInterpreter I, const char* dir) {
  getInterpreter(I)->AddIncludePath(dir);
}

enum CXErrorCode clang_Interpreter_declare(CXInterpreter I, const char* code,
                                           bool silent) {
  auto* interp = getInterpreter(I);
  auto& diag = interp->getSema().getDiagnostics();

  const bool is_silent_old = diag.getSuppressAllDiagnostics();

  diag.setSuppressAllDiagnostics(silent);
  const auto result = interp->declare(code);
  diag.setSuppressAllDiagnostics(is_silent_old);

  if (result)
    return CXError_Failure;

  return CXError_Success;
}

enum CXErrorCode clang_Interpreter_process(CXInterpreter I, const char* code) {
  if (getInterpreter(I)->process(code))
    return CXError_Failure;

  return CXError_Success;
}

CXValue clang_createValue(void) {
#ifdef CPPINTEROP_USE_CLING
  auto val = std::make_unique<cling::Value>();
#else
  auto val = std::make_unique<clang::Value>();
#endif // CPPINTEROP_USE_CLING

  return val.release();
}

void clang_Value_dispose(CXValue V) {
#ifdef CPPINTEROP_USE_CLING
  delete static_cast<cling::Value*>(V); // NOLINT(*-owning-memory)
#else
  delete static_cast<clang::Value*>(V); // NOLINT(*-owning-memory)
#endif // CPPINTEROP_USE_CLING
}

enum CXErrorCode clang_Interpreter_evaluate(CXInterpreter I, const char* code,
                                            CXValue V) {
#ifdef CPPINTEROP_USE_CLING
  auto* val = static_cast<cling::Value*>(V);
#else
  auto* val = static_cast<clang::Value*>(V);
#endif // CPPINTEROP_USE_CLING

  if (getInterpreter(I)->evaluate(code, *val))
    return CXError_Failure;

  return CXError_Success;
}

CXString clang_Interpreter_lookupLibrary(CXInterpreter I,
                                         const char* lib_name) {
  auto* interp = getInterpreter(I);
  return makeCXString(
      interp->getDynamicLibraryManager()->lookupLibrary(lib_name));
}

CXInterpreter_CompilationResult
clang_Interpreter_loadLibrary(CXInterpreter I, const char* lib_stem,
                              bool lookup) {
  auto* interp = getInterpreter(I);
  return static_cast<CXInterpreter_CompilationResult>(
      interp->loadLibrary(lib_stem, lookup));
}

void clang_Interpreter_unloadLibrary(CXInterpreter I, const char* lib_stem) {
  auto* interp = getInterpreter(I);
  interp->getDynamicLibraryManager()->unloadLibrary(lib_stem);
}

CXString clang_Interpreter_searchLibrariesForSymbol(CXInterpreter I,
                                                    const char* mangled_name,
                                                    bool search_system) {
  auto* interp = getInterpreter(I);
  return makeCXString(
      interp->getDynamicLibraryManager()->searchLibrariesForSymbol(
          mangled_name, search_system));
}

namespace Cpp {
bool InsertOrReplaceJitSymbol(compat::Interpreter& I,
                              const char* linker_mangled_name,
                              uint64_t address);
} // namespace Cpp

bool clang_Interpreter_insertOrReplaceJitSymbol(CXInterpreter I,
                                                const char* linker_mangled_name,
                                                uint64_t address) {
  return Cpp::InsertOrReplaceJitSymbol(*getInterpreter(I), linker_mangled_name,
                                       address);
}

static inline clang::QualType getType(const CXQualType& Ty) {
  return clang::QualType::getFromOpaquePtr(Ty.data[0]);
}

static inline CXInterpreterImpl* getNewTU(const CXQualType& Ty) {
  return static_cast<CXInterpreterImpl*>(Ty.data[1]);
}

static inline compat::Interpreter* getInterpreter(const CXQualType& Ty) {
  return getInterpreter(static_cast<const CXInterpreterImpl*>(Ty.data[1]));
}

CXString clang_getTypeAsString(CXQualType type) {
  const clang::QualType QT = getType(type);
  const auto& C = getInterpreter(type)->getSema().getASTContext();
  clang::PrintingPolicy Policy = C.getPrintingPolicy();
  Policy.Bool = true;               // Print bool instead of _Bool.
  Policy.SuppressTagKeyword = true; // Do not print `class std::string`.
  return makeCXString(compat::FixTypeName(QT.getAsString(Policy)));
}

CXQualType clang_getComplexType(CXQualType eltype) {
  const auto& C = getInterpreter(eltype)->getSema().getASTContext();
  return clang::cxscope::MakeCXQualType(C.getComplexType(getType(eltype)),
                                        getNewTU(eltype));
}

static inline bool isNull(const CXScope& S) { return !S.data[0]; }

static inline clang::Decl* getDecl(const CXScope& S) {
  return const_cast<clang::Decl*>(static_cast<const clang::Decl*>(S.data[0]));
}

static inline const CXInterpreterImpl* getNewTU(const CXScope& S) {
  return static_cast<const CXInterpreterImpl*>(S.data[2]);
}

static inline CXCursorKind kind(const CXScope& S) { return S.kind; }

static inline compat::Interpreter* getInterpreter(const CXScope& S) {
  return getInterpreter(static_cast<const CXInterpreterImpl*>(S.data[2]));
}

void clang_scope_dump(CXScope S) { getDecl(S)->dump(); }

bool clang_hasDefaultConstructor(CXScope S) {
  auto* D = getDecl(S);

  if (const auto* CXXRD = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(D))
    return CXXRD->hasDefaultConstructor();

  return false;
}

CXScope clang_getDefaultConstructor(CXScope S) {
  if (!clang_hasDefaultConstructor(S))
    return clang::cxscope::MakeCXScope(nullptr, getNewTU(S));

  auto* CXXRD = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(getDecl(S));
  if (!CXXRD)
    return clang::cxscope::MakeCXScope(nullptr, getNewTU(S));

  const auto* Res =
      getInterpreter(S)->getSema().LookupDefaultConstructor(CXXRD);
  return clang::cxscope::MakeCXScope(Res, getNewTU(S));
}

CXScope clang_getDestructor(CXScope S) {
  auto* D = getDecl(S);

  if (auto* CXXRD = llvm::dyn_cast_or_null<clang::CXXRecordDecl>(D)) {
    getInterpreter(S)->getSema().ForceDeclarationOfImplicitMembers(CXXRD);
    return clang::cxscope::MakeCXScope(CXXRD->getDestructor(), getNewTU(S));
  }

  return clang::cxscope::MakeCXScope(nullptr, getNewTU(S));
}

CXString clang_getFunctionSignature(CXScope func) {
  if (isNull(func))
    return makeCXString("");

  auto* D = getDecl(func);
  if (const auto* FD = llvm::dyn_cast<clang::FunctionDecl>(D)) {
    std::string Signature;
    llvm::raw_string_ostream SS(Signature);
    const auto& C = getInterpreter(func)->getSema().getASTContext();
    clang::PrintingPolicy Policy = C.getPrintingPolicy();
    // Skip printing the body
    Policy.TerseOutput = true;
    Policy.FullyQualifiedName = true;
    Policy.SuppressDefaultTemplateArgs = false;
    FD->print(SS, Policy);
    SS.flush();
    return makeCXString(Signature);
  }

  return makeCXString("");
}

bool clang_isTemplatedFunction(CXScope func) {
  auto* D = getDecl(func);
  if (llvm::isa_and_nonnull<clang::FunctionTemplateDecl>(D))
    return true;

  if (const auto* FD = llvm::dyn_cast_or_null<clang::FunctionDecl>(D)) {
    const auto TK = FD->getTemplatedKind();
    return TK == clang::FunctionDecl::TemplatedKind::
                     TK_FunctionTemplateSpecialization ||
           TK == clang::FunctionDecl::TemplatedKind::
                     TK_DependentFunctionTemplateSpecialization ||
           TK == clang::FunctionDecl::TemplatedKind::TK_FunctionTemplate;
  }

  return false;
}

bool clang_existsFunctionTemplate(const char* name, CXScope parent) {
  if (kind(parent) == CXCursor_FirstInvalid || !name)
    return false;

  const auto* Within = llvm::dyn_cast<clang::DeclContext>(getDecl(parent));

  auto& S = getInterpreter(parent)->getSema();
  auto* ND = Cpp::Cpp_utils::Lookup::Named(&S, name, Within);

  if (!ND)
    return false;

  if (intptr_t(ND) != (intptr_t)-1)
    return clang_isTemplatedFunction(
        clang::cxscope::MakeCXScope(ND, getNewTU(parent)));

  // FIXME: Cycle through the Decls and check if there is a templated
  return true;
}

namespace Cpp {
TCppScope_t InstantiateTemplate(compat::Interpreter& I, TCppScope_t tmpl,
                                const TemplateArgInfo* template_args,
                                size_t template_args_size);
} // namespace Cpp

CXScope clang_instantiateTemplate(CXScope tmpl,
                                  CXTemplateArgInfo* template_args,
                                  size_t template_args_size) {
  auto* I = getInterpreter(tmpl);

  llvm::SmallVector<Cpp::TemplateArgInfo> Info;
  for (size_t i = 0; i < template_args_size; ++i) {
    Info.push_back(Cpp::TemplateArgInfo(template_args[i].Type,
                                        template_args[i].IntegralValue));
  }

  auto* D = static_cast<clang::Decl*>(Cpp::InstantiateTemplate(
      *I, static_cast<void*>(getDecl(tmpl)), Info.data(), template_args_size));

  return clang::cxscope::MakeCXScope(D, getNewTU(tmpl));
}

CXObject clang_allocate(unsigned int n) { return ::operator new(n); }

void clang_deallocate(CXObject address) { ::operator delete(address); }

namespace Cpp {
void* Construct(compat::Interpreter& interp, TCppScope_t scope,
                void* arena /*=nullptr*/);
} // namespace Cpp

CXObject clang_construct(CXScope scope, void* arena) {
  return Cpp::Construct(*getInterpreter(scope),
                        static_cast<void*>(getDecl(scope)), arena);
}

void clang_invoke(CXScope func, void* result, void** args, size_t n,
                  void* self) {
  Cpp::MakeFunctionCallable(getInterpreter(func), getDecl(func))
      .Invoke(result, {args, n}, self);
}

namespace Cpp {
void Destruct(compat::Interpreter& interp, TCppObject_t This,
              clang::Decl* Class, bool withFree);
} // namespace Cpp

void clang_destruct(CXObject This, CXScope S, bool withFree) {
  Cpp::Destruct(*getInterpreter(S), This, getDecl(S), withFree);
}