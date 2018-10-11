#ifndef CPPYY_CAPI
#define CPPYY_CAPI

#include <stddef.h>
#include "precommondefs.h"

#ifdef __cplusplus
extern "C" {
#endif // ifdef __cplusplus

    typedef ptrdiff_t     cppyy_scope_t;
    typedef cppyy_scope_t cppyy_type_t;
    typedef void*         cppyy_object_t;
    typedef ptrdiff_t     cppyy_method_t;

    typedef long          cppyy_index_t;
    typedef void*         cppyy_funcaddr_t;

    typedef unsigned long cppyy_exctype_t;

    /* name to opaque C++ scope representation -------------------------------- */
    RPY_EXTERN
    char* cppyy_resolve_name(const char* cppitem_name);
    RPY_EXTERN
    char* cppyy_resolve_enum(const char* enum_type);
    RPY_EXTERN
    cppyy_scope_t cppyy_get_scope(const char* scope_name);
    RPY_EXTERN
    cppyy_type_t cppyy_actual_class(cppyy_type_t klass, cppyy_object_t obj);
    RPY_EXTERN
    size_t cppyy_size_of_klass(cppyy_type_t klass);
    RPY_EXTERN
    size_t cppyy_size_of_type(const char* type_name);

    /* memory management ------------------------------------------------------ */
    RPY_EXTERN
    cppyy_object_t cppyy_allocate(cppyy_type_t type);
    RPY_EXTERN
    void cppyy_deallocate(cppyy_type_t type, cppyy_object_t self);
    RPY_EXTERN
    cppyy_object_t cppyy_construct(cppyy_type_t type);
    RPY_EXTERN
    void cppyy_destruct(cppyy_type_t type, cppyy_object_t self);

    /* method/function dispatching -------------------------------------------- */
    RPY_EXTERN
    void cppyy_call_v(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    unsigned char cppyy_call_b(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    char cppyy_call_c(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    short cppyy_call_h(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    int cppyy_call_i(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    long cppyy_call_l(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    long long cppyy_call_ll(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    float cppyy_call_f(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    double cppyy_call_d(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    long double cppyy_call_ld(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);

    RPY_EXTERN
    void* cppyy_call_r(cppyy_method_t method, cppyy_object_t self, int nargs, void* args);
    RPY_EXTERN
    char* cppyy_call_s(cppyy_method_t method, cppyy_object_t self, int nargs, void* args, size_t* length);
    RPY_EXTERN
    cppyy_object_t cppyy_constructor(cppyy_method_t method, cppyy_type_t klass, int nargs, void* args);
    RPY_EXTERN
    void cppyy_destructor(cppyy_type_t type, cppyy_object_t self);
    RPY_EXTERN
    cppyy_object_t cppyy_call_o(cppyy_method_t method, cppyy_object_t self, int nargs, void* args, cppyy_type_t result_type);

    RPY_EXTERN
    cppyy_funcaddr_t cppyy_function_address_from_index(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    cppyy_funcaddr_t cppyy_function_address_from_method(cppyy_method_t method);

    /* handling of function argument buffer ----------------------------------- */
    RPY_EXTERN
    void* cppyy_allocate_function_args(int nargs);
    RPY_EXTERN
    void cppyy_deallocate_function_args(void* args);
    RPY_EXTERN
    size_t cppyy_function_arg_sizeof();
    RPY_EXTERN
    size_t cppyy_function_arg_typeoffset();

    /* scope reflection information ------------------------------------------- */
    RPY_EXTERN
    int cppyy_is_namespace(cppyy_scope_t scope);
    RPY_EXTERN
    int cppyy_is_template(const char* template_name);
    RPY_EXTERN
    int cppyy_is_abstract(cppyy_type_t type);
    RPY_EXTERN
    int cppyy_is_enum(const char* type_name);

    RPY_EXTERN
    const char** cppyy_get_all_cpp_names(cppyy_scope_t scope, size_t* count);

    /* class reflection information ------------------------------------------- */
    RPY_EXTERN
    char* cppyy_final_name(cppyy_type_t type);
    RPY_EXTERN
    char* cppyy_scoped_final_name(cppyy_type_t type);
    RPY_EXTERN
    int cppyy_has_complex_hierarchy(cppyy_type_t type);
    RPY_EXTERN
    int cppyy_num_bases(cppyy_type_t type);
    RPY_EXTERN
    char* cppyy_base_name(cppyy_type_t type, int base_index);
    RPY_EXTERN
    int cppyy_is_subtype(cppyy_type_t derived, cppyy_type_t base);
    RPY_EXTERN
    int cppyy_smartptr_info(const char* name, cppyy_type_t* raw, cppyy_method_t* deref);
    RPY_EXTERN
    void cppyy_add_smartptr_type(const char* type_name);

    /* calculate offsets between declared and actual type, up-cast: direction > 0; down-cast: direction < 0 */
    RPY_EXTERN
    ptrdiff_t cppyy_base_offset(cppyy_type_t derived, cppyy_type_t base, cppyy_object_t address, int direction);

    /* method/function reflection information --------------------------------- */
    RPY_EXTERN
    int cppyy_num_methods(cppyy_scope_t scope);
    RPY_EXTERN
    cppyy_index_t* cppyy_method_indices_from_name(cppyy_scope_t scope, const char* name);

    RPY_EXTERN
    char* cppyy_method_name(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    char* cppyy_method_mangled_name(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    char* cppyy_method_result_type(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_method_num_args(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_method_req_args(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    char* cppyy_method_arg_type(cppyy_scope_t scope, cppyy_index_t idx, int arg_index);
    RPY_EXTERN
    char* cppyy_method_arg_default(cppyy_scope_t scope, cppyy_index_t idx, int arg_index);
    RPY_EXTERN
    char* cppyy_method_signature(cppyy_scope_t scope, cppyy_index_t idx, int show_formalargs);
    RPY_EXTERN
    char* cppyy_method_prototype(cppyy_scope_t scope, cppyy_index_t idx, int show_formalargs);
    RPY_EXTERN
    int cppyy_is_const_method(cppyy_method_t);

    RPY_EXTERN
    int cppyy_exists_method_template(cppyy_scope_t scope, const char* name);
    RPY_EXTERN
    int cppyy_method_is_template(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_method_num_template_args(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    char* cppyy_method_template_arg_name(cppyy_scope_t scope, cppyy_index_t idx, cppyy_index_t iarg);

    RPY_EXTERN
    cppyy_method_t cppyy_get_method(cppyy_scope_t scope, cppyy_index_t idx);
    RPY_EXTERN
    cppyy_index_t cppyy_get_global_operator(
        cppyy_scope_t scope, cppyy_scope_t lc, cppyy_scope_t rc, const char* op);

    /* method properties ------------------------------------------------------ */
    RPY_EXTERN
    int cppyy_is_publicmethod(cppyy_type_t type, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_is_constructor(cppyy_type_t type, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_is_destructor(cppyy_type_t type, cppyy_index_t idx);
    RPY_EXTERN
    int cppyy_is_staticmethod(cppyy_type_t type, cppyy_index_t idx);

    /* data member reflection information ------------------------------------- */
    RPY_EXTERN
    int cppyy_num_datamembers(cppyy_scope_t scope);
    RPY_EXTERN
    char* cppyy_datamember_name(cppyy_scope_t scope, int datamember_index);
    RPY_EXTERN
    char* cppyy_datamember_type(cppyy_scope_t scope, int datamember_index);
    RPY_EXTERN
    ptrdiff_t cppyy_datamember_offset(cppyy_scope_t scope, int datamember_index);
    RPY_EXTERN
    int cppyy_datamember_index(cppyy_scope_t scope, const char* name);

    /* data member properties ------------------------------------------------- */
    RPY_EXTERN
    int cppyy_is_publicdata(cppyy_type_t type, cppyy_index_t datamember_index);
    RPY_EXTERN
    int cppyy_is_staticdata(cppyy_type_t type, cppyy_index_t datamember_index);
    RPY_EXTERN
    int cppyy_is_const_data(cppyy_scope_t scope, cppyy_index_t idata);
    RPY_EXTERN
    int cppyy_is_enum_data(cppyy_scope_t scope, cppyy_index_t idata);
    RPY_EXTERN
    int cppyy_get_dimension_size(cppyy_scope_t scope, cppyy_index_t idata, int dimension);

    /* misc helpers ----------------------------------------------------------- */
    RPY_EXTERN
    long long cppyy_strtoll(const char* str);
    RPY_EXTERN
    unsigned long long cppyy_strtoull(const char* str);
    RPY_EXTERN
    void cppyy_free(void* ptr);

    RPY_EXTERN
    cppyy_object_t cppyy_charp2stdstring(const char* str, size_t sz);
    RPY_EXTERN
    const char* cppyy_stdstring2charp(cppyy_object_t ptr, size_t* lsz);
    RPY_EXTERN
    cppyy_object_t cppyy_stdstring2stdstring(cppyy_object_t ptr);

    RPY_EXTERN
    const char* cppyy_stdvector_valuetype(const char* clname);
    RPY_EXTERN
    size_t      cppyy_stdvector_valuesize(const char* clname);

#ifdef __cplusplus
}
#endif // ifdef __cplusplus

#endif // ifndef CPPYY_CAPI
