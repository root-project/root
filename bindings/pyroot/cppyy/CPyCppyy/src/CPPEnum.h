#ifndef CPYCPPYY_CPPENUM_H
#define CPYCPPYY_CPPENUM_H


namespace CPyCppyy {

// CPPEnum does not carry any additional C-side data for now, but can be of
// several types, based on the declared or compile-dependent types chosen.
typedef PyObject CPPEnum;

//- creation -----------------------------------------------------------------
CPPEnum* CPPEnum_New(const std::string& name, Cppyy::TCppScope_t scope);

PyObject* pyval_from_enum(const std::string& enum_type, PyObject* pytype,
        PyObject* btype, Cppyy::TCppScope_t enum_constant);
} // namespace CPyCppyy

#endif // !CPYCPPYY_CPPENUM_H
