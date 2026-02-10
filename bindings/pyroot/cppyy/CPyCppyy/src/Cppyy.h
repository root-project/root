#ifndef CPYCPPYY_CPPYY_H
#define CPYCPPYY_CPPYY_H

// Standard
#include <set>
#include <string>
#include <vector>
#include <stddef.h>
#include <stdint.h>

// import/export (after precommondefs.h from PyPy)
#ifdef _MSC_VER
#define CPPYY_IMPORT extern __declspec(dllimport)
#else
#define CPPYY_IMPORT extern
#endif

// some more types; assumes Cppyy.h follows Python.h
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

namespace Cpp {
    struct TemplateArgInfo {
      void* m_Type;
      const char* m_IntegralValue;
      TemplateArgInfo(void* type, const char* integral_value = nullptr)
        : m_Type(type), m_IntegralValue(integral_value) {}
    };
} // end namespace Cpp

namespace Cppyy {

    typedef void*       TCppScope_t;
    typedef TCppScope_t TCppType_t;
    typedef void*       TCppEnum_t;
    typedef void*       TCppObject_t;
    typedef void*       TCppMethod_t;

    typedef size_t      TCppIndex_t;
    typedef void*       TCppFuncAddr_t;

// direct interpreter access -------------------------------------------------
    CPPYY_IMPORT
    bool Compile(const std::string& code, bool silent = false);
    CPPYY_IMPORT
    std::string ToString(TCppType_t klass, TCppObject_t obj);

// name to opaque C++ scope representation -----------------------------------
    CPPYY_IMPORT
    std::string ResolveName(const std::string& cppitem_name);

    CPPYY_IMPORT
    TCppType_t ResolveEnumReferenceType(TCppType_t type);
    CPPYY_IMPORT
    TCppType_t ResolveEnumPointerType(TCppType_t type);

    CPPYY_IMPORT
    TCppType_t ResolveType(TCppType_t type);
    CPPYY_IMPORT
    TCppType_t GetRealType(TCppType_t type);
    CPPYY_IMPORT
    TCppType_t GetReferencedType(TCppType_t type, bool rvalue);
    CPPYY_IMPORT
    TCppType_t GetPointerType(TCppType_t type);
    CPPYY_IMPORT
    std::string ResolveEnum(TCppScope_t enum_type);
    CPPYY_IMPORT
    TCppScope_t GetScope(const std::string& name, TCppScope_t parent_scope = 0);
    CPPYY_IMPORT
    TCppScope_t GetUnderlyingScope(TCppScope_t scope);
    CPPYY_IMPORT
    TCppScope_t GetFullScope(const std::string& scope_name);
    CPPYY_IMPORT
    TCppScope_t GetTypeScope(TCppScope_t klass);
    CPPYY_IMPORT
    TCppScope_t GetNamed(const std::string& scope_name,
                         TCppScope_t parent_scope = 0);
    CPPYY_IMPORT
    TCppScope_t GetParentScope(TCppScope_t scope);
    CPPYY_IMPORT
    TCppScope_t GetScopeFromType(TCppScope_t type);
    CPPYY_IMPORT
    TCppType_t  GetTypeFromScope(TCppScope_t klass);
    CPPYY_IMPORT
    TCppScope_t GetGlobalScope();
    CPPYY_IMPORT
    TCppScope_t GetActualClass(TCppScope_t klass, TCppObject_t obj);
    CPPYY_IMPORT
    size_t      SizeOf(TCppScope_t klass);
    CPPYY_IMPORT
    size_t      SizeOfType(TCppType_t type);
    CPPYY_IMPORT
    size_t      SizeOf(const std::string& type_name);

    CPPYY_IMPORT
    bool        IsBuiltin(const std::string& type_name);
    CPPYY_IMPORT
    bool        IsComplete(TCppScope_t scope);

    CPPYY_IMPORT
    bool IsPointerType(TCppType_t type);

    CPPYY_IMPORT
    TCppScope_t gGlobalScope;      // for fast access

// memory management ---------------------------------------------------------
    CPPYY_IMPORT
    TCppObject_t Allocate(TCppType_t type);
    CPPYY_IMPORT
    void         Deallocate(TCppType_t type, TCppObject_t instance);
    CPPYY_IMPORT
    TCppObject_t Construct(TCppType_t type, void* arena = nullptr);
    CPPYY_IMPORT
    void         Destruct(TCppType_t type, TCppObject_t instance);

// method/function dispatching -----------------------------------------------
    CPPYY_IMPORT
    void          CallV(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    unsigned char CallB(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    char          CallC(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    short         CallH(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    int           CallI(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    long          CallL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    PY_LONG_LONG  CallLL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    float         CallF(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    double        CallD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    PY_LONG_DOUBLE CallLD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    void*         CallR(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    CPPYY_IMPORT
    char*         CallS(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, size_t* length);
    CPPYY_IMPORT
    TCppObject_t  CallConstructor(TCppMethod_t method, TCppScope_t klass, size_t nargs, void* args);
    CPPYY_IMPORT
    void          CallDestructor(TCppType_t type, TCppObject_t self);
    CPPYY_IMPORT
    TCppObject_t  CallO(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, TCppType_t result_type);

    CPPYY_IMPORT
    TCppFuncAddr_t GetFunctionAddress(TCppMethod_t method, bool check_enabled=true);

// handling of function argument buffer --------------------------------------
    CPPYY_IMPORT
    void*  AllocateFunctionArgs(size_t nargs);
    CPPYY_IMPORT
    void   DeallocateFunctionArgs(void* args);
    CPPYY_IMPORT
    size_t GetFunctionArgSizeof();
    CPPYY_IMPORT
    size_t GetFunctionArgTypeoffset();

// scope reflection information ----------------------------------------------
    CPPYY_IMPORT
    bool IsNamespace(TCppScope_t scope);
    CPPYY_IMPORT
    bool IsClass(TCppScope_t scope);
    CPPYY_IMPORT
    bool IsTemplate(TCppScope_t handle);
    CPPYY_IMPORT
    bool IsTemplateInstantiation(TCppScope_t handle);
    CPPYY_IMPORT
    bool IsTypedefed(TCppScope_t handle);
    CPPYY_IMPORT
    bool IsAbstract(TCppType_t type);
    CPPYY_IMPORT
    bool IsEnumScope(TCppScope_t scope);
    CPPYY_IMPORT
    bool IsEnumConstant(TCppScope_t scope);
    CPPYY_IMPORT
    bool IsEnumType(TCppType_t type);
    CPPYY_IMPORT
    bool IsAggregate(TCppType_t type);
    CPPYY_IMPORT
    bool IsDefaultConstructable(TCppScope_t scope);
    CPPYY_IMPORT
    bool IsVariable(TCppScope_t scope);

    CPPYY_IMPORT
    void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

// namespace reflection information ------------------------------------------
    CPPYY_IMPORT
    std::vector<TCppScope_t> GetUsingNamespaces(TCppScope_t);

// class reflection information ----------------------------------------------
    CPPYY_IMPORT
    std::string GetFinalName(TCppType_t type);
    CPPYY_IMPORT
    std::string GetScopedFinalName(TCppType_t type);
    CPPYY_IMPORT
    bool        HasVirtualDestructor(TCppType_t type);
    CPPYY_IMPORT
    bool        HasComplexHierarchy(TCppType_t type);
    CPPYY_IMPORT
    TCppIndex_t GetNumBases(TCppType_t type);
    CPPYY_IMPORT
    TCppIndex_t GetNumBasesLongestBranch(TCppType_t type);
    CPPYY_IMPORT
    std::string GetBaseName(TCppType_t type, TCppIndex_t ibase);
    CPPYY_IMPORT
    TCppScope_t GetBaseScope(TCppType_t type, TCppIndex_t ibase);
    CPPYY_IMPORT
    bool        IsSubclass(TCppType_t derived, TCppType_t base);
    CPPYY_IMPORT
    bool        IsSmartPtr(TCppType_t type);
    CPPYY_IMPORT
    bool        GetSmartPtrInfo(const std::string&, TCppType_t* raw, TCppMethod_t* deref);
    CPPYY_IMPORT
    void        AddSmartPtrType(const std::string&);

    CPPYY_IMPORT
    void        AddTypeReducer(const std::string& reducable, const std::string& reduced);

// calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
    CPPYY_IMPORT
    ptrdiff_t GetBaseOffset(
        TCppType_t derived, TCppType_t base, TCppObject_t address, int direction, bool rerror = false);

// method/function reflection information ------------------------------------
    CPPYY_IMPORT
    void GetClassMethods(TCppScope_t scope, std::vector<TCppMethod_t> &methods);
    CPPYY_IMPORT
    std::vector<TCppScope_t> GetMethodsFromName(TCppScope_t scope, const std::string& name);

    CPPYY_IMPORT
    TCppMethod_t GetMethod(TCppScope_t scope, TCppIndex_t imeth);

    CPPYY_IMPORT
    std::string GetMethodName(TCppMethod_t);
    CPPYY_IMPORT
    std::string GetMethodFullName(TCppMethod_t);
    CPPYY_IMPORT
    std::string GetMethodMangledName(TCppMethod_t);
    CPPYY_IMPORT
    TCppType_t GetMethodReturnType(TCppMethod_t);
    CPPYY_IMPORT
    std::string GetMethodReturnTypeAsString(TCppMethod_t);
    CPPYY_IMPORT
    TCppIndex_t GetMethodNumArgs(TCppMethod_t);
    CPPYY_IMPORT
    TCppIndex_t GetMethodReqArgs(TCppMethod_t);
    CPPYY_IMPORT
    std::string GetMethodArgName(TCppMethod_t, TCppIndex_t iarg);
    CPPYY_IMPORT
    TCppType_t GetMethodArgType(TCppMethod_t, TCppIndex_t iarg);
    CPPYY_IMPORT
    std::string GetMethodArgTypeAsString(TCppMethod_t, TCppIndex_t iarg);
    CPPYY_IMPORT
    std::string GetMethodArgCanonTypeAsString(TCppMethod_t, TCppIndex_t iarg);
    CPPYY_IMPORT
    TCppIndex_t CompareMethodArgType(TCppMethod_t, TCppIndex_t iarg, const std::string &req_type);
    CPPYY_IMPORT
    std::string GetMethodArgDefault(TCppMethod_t, TCppIndex_t iarg);
    CPPYY_IMPORT
    std::string GetMethodSignature(TCppMethod_t, bool show_formal_args, TCppIndex_t max_args = (TCppIndex_t)-1);
    CPPYY_IMPORT
    std::string GetMethodPrototype(TCppMethod_t, bool show_formal_args);
    CPPYY_IMPORT
    bool        IsConstMethod(TCppMethod_t);
    CPPYY_IMPORT
    void GetTemplatedMethods(TCppScope_t scope, std::vector<TCppMethod_t> &methods);
    CPPYY_IMPORT
    TCppIndex_t GetNumTemplatedMethods(TCppScope_t scope, bool accept_namespace = false);
    CPPYY_IMPORT
    std::string GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth);
    CPPYY_IMPORT
    bool        ExistsMethodTemplate(TCppScope_t scope, const std::string& name);
    CPPYY_IMPORT
    bool        IsTemplatedMethod(TCppMethod_t method);
    CPPYY_IMPORT
    bool        IsStaticTemplate(TCppScope_t scope, const std::string& name);
    CPPYY_IMPORT
    TCppMethod_t GetMethodTemplate(
        TCppScope_t scope, const std::string& name, const std::string& proto);

    CPPYY_IMPORT
    TCppMethod_t GetGlobalOperator(TCppType_t scope, const std::string &lc,
                                   const std::string &rc,
                                   const std::string &op);

// method properties ---------------------------------------------------------
    CPPYY_IMPORT
    bool IsDeletedMethod(TCppMethod_t method);
    CPPYY_IMPORT
    bool IsPublicMethod(TCppMethod_t method);
    CPPYY_IMPORT
    bool IsProtectedMethod(TCppMethod_t method);
    CPPYY_IMPORT
    bool IsConstructor(TCppMethod_t method);
    CPPYY_IMPORT
    bool IsDestructor(TCppMethod_t method);
    CPPYY_IMPORT
    bool IsStaticMethod(TCppMethod_t method);

// data member reflection information ----------------------------------------
    CPPYY_IMPORT
    void GetDatamembers(TCppScope_t scope, std::vector<TCppScope_t>& datamembers);
    CPPYY_IMPORT
    std::string GetDatamemberName(TCppScope_t scope, TCppIndex_t idata);
    CPPYY_IMPORT
    TCppScope_t ReduceReturnType(TCppScope_t fn, TCppType_t reduce);
    bool IsLambdaClass(TCppType_t type);
    CPPYY_IMPORT
    TCppScope_t WrapLambdaFromVariable(TCppScope_t var);
    CPPYY_IMPORT
    TCppScope_t AdaptFunctionForLambdaReturn(TCppScope_t fn);
    CPPYY_IMPORT
    TCppType_t  GetDatamemberType(TCppScope_t var);
    CPPYY_IMPORT
    std::string GetTypeAsString(TCppType_t type);
    CPPYY_IMPORT
    bool IsClassType(TCppType_t type);
    CPPYY_IMPORT
    bool IsFunctionPointerType(TCppType_t type);
    CPPYY_IMPORT
    TCppType_t  GetType(const std::string& name, bool enable_slow_lookup = false);
    CPPYY_IMPORT
    bool AppendTypesSlow(const std::string &name,
                         std::vector<Cpp::TemplateArgInfo>& types,
                         TCppScope_t parent = nullptr);
    CPPYY_IMPORT
    TCppType_t  GetComplexType(const std::string& element_type);
    CPPYY_IMPORT
    std::string GetDatamemberTypeAsString(TCppScope_t var);
    CPPYY_IMPORT
    intptr_t    GetDatamemberOffset(TCppScope_t var, TCppScope_t klass = nullptr);
    CPPYY_IMPORT
    bool        CheckDatamember(TCppScope_t scope, const std::string& name);

// data member properties ----------------------------------------------------
    CPPYY_IMPORT
    bool IsPublicData(TCppScope_t data);
    CPPYY_IMPORT
    bool IsProtectedData(TCppScope_t var);
    CPPYY_IMPORT
    bool IsStaticDatamember(TCppScope_t var);
    CPPYY_IMPORT
    bool IsConstVar(TCppScope_t var);
    CPPYY_IMPORT
    bool IsEnumData(TCppScope_t scope, TCppIndex_t idata);
    CPPYY_IMPORT
    std::vector<long int> GetDimensions(TCppType_t type);

// enum properties -----------------------------------------------------------
    // CPPYY_IMPORT
    // TCppEnum_t  GetEnum(TCppScope_t scope, const std::string& enum_name);
    CPPYY_IMPORT
    TCppScope_t GetEnumScope(TCppScope_t);
    CPPYY_IMPORT
    std::vector<TCppScope_t> GetEnumConstants(TCppScope_t scope);
    CPPYY_IMPORT
    TCppType_t  GetEnumConstantType(TCppScope_t scope);
    CPPYY_IMPORT
    long long   GetEnumDataValue(TCppScope_t scope);
    CPPYY_IMPORT
    TCppScope_t InstantiateTemplate(
            TCppScope_t tmpl, Cpp::TemplateArgInfo* args, size_t args_size);

    CPPYY_IMPORT
    TCppScope_t DumpScope(TCppScope_t scope);
} // namespace Cppyy

#endif // !CPYCPPYY_CPPYY_H
