#ifndef CPYCPPYY_CPPYY_H
#define CPYCPPYY_CPPYY_H

// Standard
#include <set>
#include <string>
#include <vector>
#include <stddef.h>


// ROOT types
typedef long long            Long64_t;
typedef unsigned long long   ULong64_t;
typedef long double          LongDouble_t;

namespace Cppyy {
    typedef ptrdiff_t   TCppScope_t;
    typedef TCppScope_t TCppType_t;
    typedef void*       TCppObject_t;
    typedef ptrdiff_t   TCppMethod_t;

    typedef long        TCppIndex_t;
    typedef void*       TCppFuncAddr_t;

// name to opaque C++ scope representation -----------------------------------
    RPY_EXTERN
    std::string ResolveName(const std::string& cppitem_name);
    RPY_EXTERN
    std::string ResolveEnum(const std::string& enum_type);
    RPY_EXTERN
    TCppScope_t GetScope(const std::string& scope_name);
    RPY_EXTERN
    TCppType_t  GetActualClass(TCppType_t klass, TCppObject_t obj);
    RPY_EXTERN
    size_t      SizeOf(TCppType_t klass);
    RPY_EXTERN
    size_t      SizeOf(const std::string& type_name);

    RPY_EXTERN
    bool        IsBuiltin(const std::string& type_name);
    RPY_EXTERN
    bool        IsComplete(const std::string& type_name);

    RPY_EXTERN
    TCppScope_t gGlobalScope;      // for fast access

// memory management ---------------------------------------------------------
    RPY_EXTERN
    TCppObject_t Allocate(TCppType_t type);
    RPY_EXTERN
    void         Deallocate(TCppType_t type, TCppObject_t instance);
    RPY_EXTERN
    TCppObject_t Construct(TCppType_t type);
    RPY_EXTERN
    void         Destruct(TCppType_t type, TCppObject_t instance);

// method/function dispatching -----------------------------------------------
    RPY_EXTERN
    void          CallV(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    unsigned char CallB(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    char          CallC(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    short         CallH(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    int           CallI(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    long          CallL(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    Long64_t      CallLL(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    float         CallF(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    double        CallD(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    LongDouble_t  CallLD(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    void*         CallR(TCppMethod_t method, TCppObject_t self, void* args);
    RPY_EXTERN
    char*         CallS(TCppMethod_t method, TCppObject_t self, void* args, size_t* length);
    RPY_EXTERN
    TCppObject_t  CallConstructor(TCppMethod_t method, TCppType_t type, void* args);
    RPY_EXTERN
    void          CallDestructor(TCppType_t type, TCppObject_t self);
    RPY_EXTERN
    TCppObject_t  CallO(TCppMethod_t method, TCppObject_t self, void* args, TCppType_t result_type);

    RPY_EXTERN
    TCppFuncAddr_t GetFunctionAddress(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXTERN
    TCppFuncAddr_t GetFunctionAddress(TCppMethod_t method);

// handling of function argument buffer --------------------------------------
    RPY_EXTERN
    void*  AllocateFunctionArgs(size_t nargs);
    RPY_EXTERN
    void   DeallocateFunctionArgs(void* args);
    RPY_EXTERN
    size_t GetFunctionArgSizeof();
    RPY_EXTERN
    size_t GetFunctionArgTypeoffset();

// scope reflection information ----------------------------------------------
    RPY_EXTERN
    bool IsNamespace(TCppScope_t scope);
    RPY_EXTERN
    bool IsTemplate(const std::string& template_name);
    RPY_EXTERN
    bool IsAbstract(TCppType_t type);
    RPY_EXTERN
    bool IsEnum(const std::string& type_name);

    RPY_EXTERN
    void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

// class reflection information ----------------------------------------------
    RPY_EXTERN
    std::string GetFinalName(TCppType_t type);
    RPY_EXTERN
    std::string GetScopedFinalName(TCppType_t type);
    RPY_EXTERN
    bool        HasComplexHierarchy(TCppType_t type);
    RPY_EXTERN
    TCppIndex_t GetNumBases(TCppType_t type);
    RPY_EXTERN
    std::string GetBaseName(TCppType_t type, TCppIndex_t ibase);
    RPY_EXTERN
    bool        IsSubtype(TCppType_t derived, TCppType_t base);
    RPY_EXTERN
    bool        GetSmartPtrInfo(const std::string&, TCppType_t& raw, TCppMethod_t& deref);
    RPY_EXTERN
    void        AddSmartPtrType(const std::string&);

// calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
    RPY_EXTERN
    ptrdiff_t GetBaseOffset(
        TCppType_t derived, TCppType_t base, TCppObject_t address, int direction, bool rerror = false);

// method/function reflection information ------------------------------------
    RPY_EXTERN
    TCppIndex_t GetNumMethods(TCppScope_t scope);
    RPY_EXTERN
    std::vector<TCppIndex_t> GetMethodIndicesFromName(TCppScope_t scope, const std::string& name);

    RPY_EXTERN
    TCppMethod_t GetMethod(TCppScope_t scope, TCppIndex_t imeth);

    RPY_EXTERN
    std::string GetMethodName(TCppMethod_t);
    RPY_EXTERN
    std::string GetMethodMangledName(TCppMethod_t);
    RPY_EXTERN
    std::string GetMethodResultType(TCppMethod_t);
    RPY_EXTERN
    TCppIndex_t GetMethodNumArgs(TCppMethod_t);
    RPY_EXTERN
    TCppIndex_t GetMethodReqArgs(TCppMethod_t);
    RPY_EXTERN
    std::string GetMethodArgName(TCppMethod_t, int iarg);
    RPY_EXTERN
    std::string GetMethodArgType(TCppMethod_t, int iarg);
    RPY_EXTERN
    std::string GetMethodArgDefault(TCppMethod_t, int iarg);
    RPY_EXTERN
    std::string GetMethodSignature(TCppScope_t scope, TCppIndex_t imeth, bool show_formalargs);
    RPY_EXTERN
    std::string GetMethodPrototype(TCppScope_t scope, TCppIndex_t imeth, bool show_formalargs);
    RPY_EXTERN
    bool        IsConstMethod(TCppMethod_t);

    RPY_EXTERN
    bool        ExistsMethodTemplate(TCppScope_t scope, const std::string& name);
    RPY_EXTERN
    bool        IsMethodTemplate(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXTERN
    TCppIndex_t GetMethodNumTemplateArgs(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXTERN
    std::string GetMethodTemplateArgName(TCppScope_t scope, TCppIndex_t imeth, TCppIndex_t iarg);

    RPY_EXTERN
    TCppMethod_t GetMethodTemplate(
        TCppScope_t scope, const std::string& name, const std::string& proto);
    RPY_EXTERN
    TCppIndex_t  GetGlobalOperator(
        TCppType_t scope, TCppType_t lc, TCppScope_t rc, const std::string& op);

// method properties ---------------------------------------------------------
    RPY_EXTERN
    bool IsPublicMethod(TCppMethod_t method);
    RPY_EXTERN
    bool IsConstructor(TCppMethod_t method);
    RPY_EXTERN
    bool IsDestructor(TCppMethod_t method);
    RPY_EXTERN
    bool IsStaticMethod(TCppMethod_t method);

// data member reflection information ----------------------------------------
    RPY_EXTERN
    TCppIndex_t GetNumDatamembers(TCppScope_t scope);
    RPY_EXTERN
    std::string GetDatamemberName(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    std::string GetDatamemberType(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    ptrdiff_t   GetDatamemberOffset(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    TCppIndex_t GetDatamemberIndex(TCppScope_t scope, const std::string& name);

// data member properties ----------------------------------------------------
    RPY_EXTERN
    bool IsPublicData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    bool IsStaticData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    bool IsConstData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    bool IsEnumData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXTERN
    int  GetDimensionSize(TCppScope_t scope, TCppIndex_t idata, int dimension);

} // namespace Cppyy

#endif // !CPYCPPYY_CPPYY_H
