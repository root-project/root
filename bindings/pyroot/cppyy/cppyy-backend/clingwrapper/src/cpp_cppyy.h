#ifndef CPYCPPYY_CPPYY_H
#define CPYCPPYY_CPPYY_H

#include <CppInterOp/Dispatch.h>

// Standard
#include <cassert>
#include <set>
#include <string>
#include <vector>
#include <stddef.h>
#include <stdint.h>
#include <iostream>

#include "callcontext.h"
// some more types; assumes Cppyy.h follows Python.h

// using CppFinal 
#ifndef PY_LONG_LONG
#ifdef _WIN32
typedef __int64 PY_LONG_LONG;
#else
typedef long long PY_LONG_LONG;
#endif
#endif

#ifndef PY_ULONG_LONG
#ifdef _WIN32
typedef unsigned __int64   PY_ULONG_LONG;
#else
typedef unsigned long long PY_ULONG_LONG;
#endif
#endif

#ifndef PY_LONG_DOUBLE
typedef long double PY_LONG_DOUBLE;
#endif

typedef CPyCppyy::Parameter Parameter;

// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// convention to pass flag for direct calls (similar to Python's vector calls)
#define DIRECT_CALL ((size_t)1 << (8 * sizeof(size_t) - 1))
static inline size_t CALL_NARGS(size_t nargs) {
    return nargs & ~DIRECT_CALL;
}

namespace Cppyy {
    typedef Cpp::TCppScope_t    TCppScope_t;
    typedef Cpp::TCppType_t     TCppType_t;
    typedef Cpp::TCppScope_t    TCppEnum_t;
    typedef Cpp::TCppScope_t    TCppObject_t;
    typedef Cpp::TCppFunction_t TCppMethod_t;
    typedef Cpp::TCppIndex_t    TCppIndex_t;
    typedef intptr_t                TCppFuncAddr_t;

// // direct interpreter access -------------------------------------------------
    // RPY_EXPORTED
    // void AddSearchPath(const char* dir, bool isUser = true, bool prepend = false); 
    RPY_EXPORTED
    bool Compile(const std::string& code, bool silent = false);
    RPY_EXPORTED
    std::string ToString(TCppType_t klass, TCppObject_t obj);
//
// // name to opaque C++ scope representation -----------------------------------
    RPY_EXPORTED
    std::string ResolveName(const std::string& cppitem_name);
    RPY_EXPORTED
    TCppType_t ResolveType(TCppType_t cppitem_name);
    RPY_EXPORTED
    TCppType_t ResolveEnumReferenceType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t ResolveEnumPointerType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t GetRealType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t GetPointerType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t GetReferencedType(TCppType_t type, bool rvalue = false);
    RPY_EXPORTED
    std::string ResolveEnum(TCppScope_t enum_scope);
    RPY_EXPORTED
    bool IsClassType(TCppType_t type);
    RPY_EXPORTED
    bool IsPointerType(TCppType_t type);
    RPY_EXPORTED
    bool IsFunctionPointerType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t GetType(const std::string &name, bool enable_slow_lookup = false);
    RPY_EXPORTED
    bool AppendTypesSlow(const std::string &name,
                         std::vector<Cpp::TemplateArgInfo>& types, Cppyy::TCppScope_t parent = nullptr);
    RPY_EXPORTED
    TCppType_t GetComplexType(const std::string &element_type);
    RPY_EXPORTED
    TCppScope_t GetScope(const std::string& scope_name,
                         TCppScope_t parent_scope = 0);
    RPY_EXPORTED
    TCppScope_t GetUnderlyingScope(TCppScope_t scope);
    RPY_EXPORTED
    TCppScope_t GetFullScope(const std::string& scope_name);
    RPY_EXPORTED
    TCppScope_t GetTypeScope(TCppScope_t klass);
    RPY_EXPORTED
    TCppScope_t GetNamed(const std::string& scope_name,
                         TCppScope_t parent_scope = 0);
    RPY_EXPORTED
    TCppScope_t GetParentScope(TCppScope_t scope);
    RPY_EXPORTED
    TCppScope_t GetScopeFromType(TCppType_t type);
    RPY_EXPORTED
    TCppType_t  GetTypeFromScope(TCppScope_t klass);
    RPY_EXPORTED
    TCppScope_t GetGlobalScope();
    RPY_EXPORTED
    TCppScope_t GetActualClass(TCppScope_t klass, TCppObject_t obj);
    RPY_EXPORTED
    size_t      SizeOf(TCppScope_t klass);
    RPY_EXPORTED
    size_t      SizeOfType(TCppType_t type);
    RPY_EXPORTED
    size_t      SizeOf(const std::string &type) { assert(0 && "SizeOf"); return 0; }

    RPY_EXPORTED
    bool        IsBuiltin(const std::string& type_name);

    RPY_EXPORTED
    bool        IsBuiltin(TCppType_t type);

    RPY_EXPORTED
    bool        IsComplete(TCppScope_t type);

//     RPY_EXPORTED
//     inline TCppScope_t gGlobalScope = 0;      // for fast access
//
// // memory management ---------------------------------------------------------
    RPY_EXPORTED
    TCppObject_t Allocate(TCppScope_t scope);
    RPY_EXPORTED
    void         Deallocate(TCppScope_t scope, TCppObject_t instance);
    RPY_EXPORTED
    TCppObject_t Construct(TCppScope_t scope, void* arena = nullptr);
    RPY_EXPORTED
    void         Destruct(TCppScope_t scope, TCppObject_t instance);

// method/function dispatching -----------------------------------------------
    RPY_EXPORTED
    void          CallV(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    unsigned char CallB(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    char          CallC(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    short         CallH(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    int           CallI(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    long          CallL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    PY_LONG_LONG  CallLL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    float         CallF(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    double        CallD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    PY_LONG_DOUBLE CallLD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);

    RPY_EXPORTED
    void*         CallR(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORTED
    char*         CallS(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, size_t* length);
    RPY_EXPORTED
    TCppObject_t  CallConstructor(TCppMethod_t method, TCppScope_t klass, size_t nargs, void* args);
    RPY_EXPORTED
    void          CallDestructor(TCppScope_t type, TCppObject_t self);
    RPY_EXPORTED
    TCppObject_t  CallO(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, TCppType_t result_type);

    RPY_EXPORTED
    TCppFuncAddr_t GetFunctionAddress(TCppMethod_t method, bool check_enabled=true);

// // handling of function argument buffer --------------------------------------
    RPY_EXPORTED
    void*  AllocateFunctionArgs(size_t nargs);
    RPY_EXPORTED
    void   DeallocateFunctionArgs(void* args);
    RPY_EXPORTED
    size_t GetFunctionArgSizeof();
    RPY_EXPORTED
    size_t GetFunctionArgTypeoffset();

// // scope reflection information ----------------------------------------------
    RPY_EXPORTED
    bool IsNamespace(TCppScope_t scope);
    RPY_EXPORTED
    bool IsClass(TCppScope_t scope);
    RPY_EXPORTED
    bool IsTemplate(TCppScope_t scope);
    RPY_EXPORTED
    bool IsTemplateInstantiation(TCppScope_t scope);
    RPY_EXPORTED
    bool IsTypedefed(TCppScope_t scope);
    RPY_EXPORTED
    bool IsAbstract(TCppScope_t scope);
    RPY_EXPORTED
    bool IsEnumScope(TCppScope_t scope);
    RPY_EXPORTED
    bool IsEnumConstant(TCppScope_t scope);
    RPY_EXPORTED
    bool IsEnumType(TCppType_t type);
    RPY_EXPORTED
    bool IsAggregate(TCppType_t type);
    RPY_EXPORTED
    bool IsDefaultConstructable(TCppScope_t scope);
    RPY_EXPORTED
    bool IsVariable(TCppScope_t scope);

    RPY_EXPORTED
    void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

// // namespace reflection information ------------------------------------------
    RPY_EXPORTED
    std::vector<Cppyy::TCppScope_t> GetUsingNamespaces(TCppScope_t);
//
// // class reflection information ----------------------------------------------
    RPY_EXPORTED
    std::string GetFinalName(TCppType_t type);
    RPY_EXPORTED
    std::string GetScopedFinalName(TCppType_t type);
    RPY_EXPORTED
    bool        HasVirtualDestructor(TCppType_t type);
    RPY_EXPORTED
    bool        HasComplexHierarchy(TCppType_t type) { assert(0 && "HasComplexHierarchy"); return false; }
    RPY_EXPORTED
    TCppIndex_t GetNumBases(TCppScope_t klass);
    RPY_EXPORTED
    TCppIndex_t GetNumBasesLongestBranch(TCppScope_t klass);
    RPY_EXPORTED
    std::string GetBaseName(TCppScope_t klass, TCppIndex_t ibase);
    RPY_EXPORTED
    TCppScope_t GetBaseScope(TCppScope_t klass, TCppIndex_t ibase);
    RPY_EXPORTED
    bool        IsSubclass(TCppType_t derived, TCppType_t base);
    RPY_EXPORTED
    bool        IsSmartPtr(TCppScope_t klass);
    RPY_EXPORTED
    bool        GetSmartPtrInfo(const std::string&, TCppType_t* raw, TCppMethod_t* deref);
    RPY_EXPORTED
    void        AddSmartPtrType(const std::string&) { assert(0 && "AddSmartPtrType"); return; }

    RPY_EXPORTED
    void        AddTypeReducer(const std::string& reducable, const std::string& reduced) { assert(0 && "AddTypeReducer"); return; }

// // calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
    RPY_EXPORTED
    ptrdiff_t GetBaseOffset(
        TCppType_t derived, TCppType_t base, TCppObject_t address, int direction, bool rerror = false);

// // method/function reflection information ------------------------------------
    RPY_EXPORTED
    void GetClassMethods(TCppScope_t scope, std::vector<TCppMethod_t> &methods);
    RPY_EXPORTED
    std::vector<TCppScope_t> GetMethodsFromName(TCppScope_t scope,
                                                const std::string& name);

    RPY_EXPORTED
    TCppMethod_t GetMethod(TCppScope_t scope, TCppIndex_t imeth) { return 0; }

    RPY_EXPORTED
    std::string GetMethodName(TCppMethod_t);
    RPY_EXPORTED
    std::string GetMethodFullName(TCppMethod_t);
    // GetMethodMangledName is unused.
    RPY_EXPORTED
    std::string GetMethodMangledName(TCppMethod_t) { assert(0 && "GetMethodMangledName"); return ""; }
    RPY_EXPORTED
    TCppType_t GetMethodReturnType(TCppMethod_t);
    RPY_EXPORTED
    std::string GetMethodReturnTypeAsString(TCppMethod_t);
    RPY_EXPORTED
    TCppIndex_t GetMethodNumArgs(TCppMethod_t);
    RPY_EXPORTED
    TCppIndex_t GetMethodReqArgs(TCppMethod_t);
    RPY_EXPORTED
    std::string GetMethodArgName(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORTED
    TCppType_t GetMethodArgType(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORTED
    TCppIndex_t CompareMethodArgType(TCppMethod_t, TCppIndex_t iarg, const std::string &req_type);
    RPY_EXPORTED
    std::string GetMethodArgTypeAsString(TCppMethod_t method, TCppIndex_t iarg);
    RPY_EXPORTED
    std::string GetMethodArgCanonTypeAsString(TCppMethod_t method, TCppIndex_t iarg);
    RPY_EXPORTED
    std::string GetMethodArgDefault(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORTED
    std::string GetMethodSignature(TCppMethod_t, bool show_formal_args, TCppIndex_t max_args = (TCppIndex_t)-1);
    // GetMethodPrototype is unused.
    RPY_EXPORTED
    std::string GetMethodPrototype(TCppMethod_t, bool show_formal_args);
    RPY_EXPORTED
    bool        IsConstMethod(TCppMethod_t);
// // Templated method/function reflection information ------------------------------------
    RPY_EXPORTED
    void GetTemplatedMethods(TCppScope_t scope, std::vector<TCppMethod_t> &methods);
    RPY_EXPORTED
    TCppIndex_t GetNumTemplatedMethods(TCppScope_t scope, bool accept_namespace = false);
    RPY_EXPORTED
    std::string GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXPORTED
    bool        ExistsMethodTemplate(TCppScope_t scope, const std::string& name);
    RPY_EXPORTED
    bool        IsTemplatedMethod(TCppMethod_t method);
    RPY_EXPORTED
    bool        IsStaticTemplate(TCppScope_t scope, const std::string& name);
    RPY_EXPORTED
    TCppMethod_t GetMethodTemplate(
        TCppScope_t scope, const std::string& name, const std::string& proto);
    RPY_EXPORTED
    void GetClassOperators(Cppyy::TCppScope_t klass, const std::string& opname,
                           std::vector<TCppScope_t>& operators);
    RPY_EXPORTED
    TCppMethod_t  GetGlobalOperator(
        TCppType_t scope, const std::string& lc, const std::string& rc, const std::string& op);

// method properties ---------------------------------------------------------
    RPY_EXPORTED
    bool IsDeletedMethod(TCppMethod_t method);
    RPY_EXPORTED
    bool IsPublicMethod(TCppMethod_t method);
    RPY_EXPORTED
    bool IsProtectedMethod(TCppMethod_t method);
    RPY_EXPORTED
    bool IsPrivateMethod(TCppMethod_t method);
    RPY_EXPORTED
    bool IsConstructor(TCppMethod_t method);
    RPY_EXPORTED
    bool IsDestructor(TCppMethod_t method);
    RPY_EXPORTED
    bool IsStaticMethod(TCppMethod_t method);

// // data member reflection information ----------------------------------------
    // GetNumDatamembers is unused.
    // RPY_EXPORTED
    // TCppIndex_t GetNumDatamembers(TCppScope_t scope, bool accept_namespace = false) { return 0; }
    RPY_EXPORTED
    void GetDatamembers(TCppScope_t scope, std::vector<TCppScope_t>& datamembers);
    // GetDatamemberName is unused.
    // RPY_EXPORTED
    // std::string GetDatamemberName(TCppScope_t scope, TCppIndex_t idata) { return ""; }
    RPY_EXPORTED
    bool IsLambdaClass(TCppType_t type);
    RPY_EXPORTED
    TCppScope_t WrapLambdaFromVariable(TCppScope_t var);
    RPY_EXPORTED
    TCppScope_t AdaptFunctionForLambdaReturn(TCppScope_t fn);
    RPY_EXPORTED
    TCppType_t GetDatamemberType(TCppScope_t data);
    RPY_EXPORTED
    std::string GetDatamemberTypeAsString(TCppScope_t var);
    RPY_EXPORTED
    std::string GetTypeAsString(TCppType_t type);
    RPY_EXPORTED
    intptr_t    GetDatamemberOffset(TCppScope_t var, TCppScope_t klass = nullptr);
    RPY_EXPORTED
    bool CheckDatamember(TCppScope_t scope, const std::string& name);

// // data member properties ----------------------------------------------------
    RPY_EXPORTED
    bool IsPublicData(TCppScope_t var);
    RPY_EXPORTED
    bool IsProtectedData(TCppScope_t var);
    RPY_EXPORTED
    bool IsPrivateData(TCppScope_t var);
    RPY_EXPORTED
    bool IsStaticDatamember(TCppScope_t var);
    RPY_EXPORTED
    bool IsConstVar(TCppScope_t var);
    RPY_EXPORTED
    TCppScope_t ReduceReturnType(TCppScope_t fn, TCppType_t reduce);
//     RPY_EXPORTED
//     bool IsEnumData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORTED
    std::vector<long int> GetDimensions(TCppType_t type);

// // enum properties -----------------------------------------------------------
    // GetEnum is unused.
    // RPY_EXPORTED
    // TCppEnum_t  GetEnum(TCppScope_t scope, const std::string& enum_name) { return 0; }
    RPY_EXPORTED
    std::vector<TCppScope_t> GetEnumConstants(TCppScope_t scope);
    // GetEnumDataName is unused.
    // RPY_EXPORTED
    // std::string GetEnumDataName(TCppEnum_t, TCppIndex_t idata) { return ""; }
    RPY_EXPORTED
    TCppType_t  GetEnumConstantType(TCppScope_t scope);
    RPY_EXPORTED
    TCppIndex_t GetEnumDataValue(TCppScope_t scope);

    RPY_EXPORTED
    TCppScope_t InstantiateTemplate(
            TCppScope_t tmpl, Cpp::TemplateArgInfo* args, size_t args_size);

    RPY_EXPORTED
    void        DumpScope(TCppScope_t scope);
} // namespace Cppyy

#endif // !CPYCPPYY_CPPYY_H