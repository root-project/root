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
#define RPY_EXPORT extern __declspec(dllimport)
#else
#define RPY_EXPORT extern
#endif


namespace Cppyy {

    typedef size_t      TCppScope_t;
    typedef TCppScope_t TCppType_t;
    typedef void*       TCppObject_t;
    typedef intptr_t    TCppMethod_t;

    typedef size_t      TCppIndex_t;
    typedef void*       TCppFuncAddr_t;

// direct interpreter access -------------------------------------------------
    RPY_EXPORT
    bool Compile(const std::string& code);

// name to opaque C++ scope representation -----------------------------------
    RPY_EXPORT
    std::string ResolveName(const std::string& cppitem_name);
    RPY_EXPORT
    std::string ResolveEnum(const std::string& enum_type);
    RPY_EXPORT
    TCppScope_t GetScope(const std::string& scope_name);
    RPY_EXPORT
    TCppType_t  GetActualClass(TCppType_t klass, TCppObject_t obj);
    RPY_EXPORT
    size_t      SizeOf(TCppType_t klass);
    RPY_EXPORT
    size_t      SizeOf(const std::string& type_name);

    RPY_EXPORT
    bool        IsBuiltin(const std::string& type_name);
    RPY_EXPORT
    bool        IsComplete(const std::string& type_name);

    RPY_EXPORT
    TCppScope_t gGlobalScope;      // for fast access

// memory management ---------------------------------------------------------
    RPY_EXPORT
    TCppObject_t Allocate(TCppType_t type);
    RPY_EXPORT
    void         Deallocate(TCppType_t type, TCppObject_t instance);
    RPY_EXPORT
    TCppObject_t Construct(TCppType_t type);
    RPY_EXPORT
    void         Destruct(TCppType_t type, TCppObject_t instance);

// method/function dispatching -----------------------------------------------
    RPY_EXPORT
    void          CallV(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    unsigned char CallB(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    char          CallC(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    short         CallH(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    int           CallI(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    long          CallL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    Long64_t      CallLL(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    float         CallF(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    double        CallD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    LongDouble_t  CallLD(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    void*         CallR(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args);
    RPY_EXPORT
    char*         CallS(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, size_t* length);
    RPY_EXPORT
    TCppObject_t  CallConstructor(TCppMethod_t method, TCppType_t type, size_t nargs, void* args);
    RPY_EXPORT
    void          CallDestructor(TCppType_t type, TCppObject_t self);
    RPY_EXPORT
    TCppObject_t  CallO(TCppMethod_t method, TCppObject_t self, size_t nargs, void* args, TCppType_t result_type);

    RPY_EXPORT
    TCppFuncAddr_t GetFunctionAddress(TCppMethod_t method);

// handling of function argument buffer --------------------------------------
    RPY_EXPORT
    void*  AllocateFunctionArgs(size_t nargs);
    RPY_EXPORT
    void   DeallocateFunctionArgs(void* args);
    RPY_EXPORT
    size_t GetFunctionArgSizeof();
    RPY_EXPORT
    size_t GetFunctionArgTypeoffset();

// scope reflection information ----------------------------------------------
    RPY_EXPORT
    bool IsNamespace(TCppScope_t scope);
    RPY_EXPORT
    bool IsTemplate(const std::string& template_name);
    RPY_EXPORT
    bool IsAbstract(TCppType_t type);
    RPY_EXPORT
    bool IsEnum(const std::string& type_name);

    RPY_EXPORT
    void GetAllCppNames(TCppScope_t scope, std::set<std::string>& cppnames);

// class reflection information ----------------------------------------------
    RPY_EXPORT
    std::string GetFinalName(TCppType_t type);
    RPY_EXPORT
    std::string GetScopedFinalName(TCppType_t type);
    RPY_EXPORT
    bool        HasComplexHierarchy(TCppType_t type);
    RPY_EXPORT
    TCppIndex_t GetNumBases(TCppType_t type);
    RPY_EXPORT
    std::string GetBaseName(TCppType_t type, TCppIndex_t ibase);
    RPY_EXPORT
    bool        IsSubtype(TCppType_t derived, TCppType_t base);
    RPY_EXPORT
    bool        GetSmartPtrInfo(const std::string&, TCppType_t& raw, TCppMethod_t& deref);
    RPY_EXPORT
    void        AddSmartPtrType(const std::string&);

// calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0
    RPY_EXPORT
    ptrdiff_t GetBaseOffset(
        TCppType_t derived, TCppType_t base, TCppObject_t address, int direction, bool rerror = false);

// method/function reflection information ------------------------------------
    RPY_EXPORT
    TCppIndex_t GetNumMethods(TCppScope_t scope);
    RPY_EXPORT
    std::vector<TCppIndex_t> GetMethodIndicesFromName(TCppScope_t scope, const std::string& name);

    RPY_EXPORT
    TCppMethod_t GetMethod(TCppScope_t scope, TCppIndex_t imeth);

    RPY_EXPORT
    std::string GetMethodName(TCppMethod_t);
    RPY_EXPORT
    std::string GetMethodFullName(TCppMethod_t);
    RPY_EXPORT
    std::string GetMethodMangledName(TCppMethod_t);
    RPY_EXPORT
    std::string GetMethodResultType(TCppMethod_t);
    RPY_EXPORT
    TCppIndex_t GetMethodNumArgs(TCppMethod_t);
    RPY_EXPORT
    TCppIndex_t GetMethodReqArgs(TCppMethod_t);
    RPY_EXPORT
    std::string GetMethodArgName(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORT
    std::string GetMethodArgType(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORT
    std::string GetMethodArgDefault(TCppMethod_t, TCppIndex_t iarg);
    RPY_EXPORT
    std::string GetMethodSignature(TCppMethod_t, bool show_formalargs);
    RPY_EXPORT
    std::string GetMethodPrototype(TCppScope_t scope, TCppMethod_t, bool show_formalargs);
    RPY_EXPORT
    bool        IsConstMethod(TCppMethod_t);

    RPY_EXPORT
    TCppIndex_t GetNumTemplatedMethods(TCppScope_t scope);
    RPY_EXPORT
    std::string GetTemplatedMethodName(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXPORT
    bool        ExistsMethodTemplate(TCppScope_t scope, const std::string& name);
    RPY_EXPORT
    bool        IsMethodTemplate(TCppScope_t scope, TCppIndex_t imeth);
    RPY_EXPORT
    TCppMethod_t GetMethodTemplate(
        TCppScope_t scope, const std::string& name, const std::string& proto);

    RPY_EXPORT
    TCppIndex_t  GetGlobalOperator(
        TCppType_t scope, TCppType_t lc, TCppScope_t rc, const std::string& op);

// method properties ---------------------------------------------------------
    RPY_EXPORT
    bool IsPublicMethod(TCppMethod_t method);
    RPY_EXPORT
    bool IsConstructor(TCppMethod_t method);
    RPY_EXPORT
    bool IsDestructor(TCppMethod_t method);
    RPY_EXPORT
    bool IsStaticMethod(TCppMethod_t method);

// data member reflection information ----------------------------------------
    RPY_EXPORT
    TCppIndex_t GetNumDatamembers(TCppScope_t scope);
    RPY_EXPORT
    std::string GetDatamemberName(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    std::string GetDatamemberType(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    intptr_t    GetDatamemberOffset(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    TCppIndex_t GetDatamemberIndex(TCppScope_t scope, const std::string& name);

// data member properties ----------------------------------------------------
    RPY_EXPORT
    bool IsPublicData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    bool IsStaticData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    bool IsConstData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    bool IsEnumData(TCppScope_t scope, TCppIndex_t idata);
    RPY_EXPORT
    int  GetDimensionSize(TCppScope_t scope, TCppIndex_t idata, int dimension);

} // namespace Cppyy

extern "C" {
    RPY_EXPORT
    void cppyy_set_converter_creator(void* (*)(const char*, long*));
    RPY_EXPORT
    void* cppyy_create_converter(const char* type_name, long* dims);
}

#endif // !CPYCPPYY_CPPYY_H
