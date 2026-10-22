#ifndef CPYRT_CPPENUM_H
#define CPYRT_CPPENUM_H

namespace cppjit::cpyrt {

// CPPEnum does not carry any additional C-side data for now, but can be of
// several types, based on the declared or compile-dependent types chosen.
typedef PyObject CPPEnum;

//- creation -----------------------------------------------------------------
CPPEnum* CPPEnum_New(const std::string& name, interop::TCppScope_t scope);

PyObject* pyval_from_enum(const std::string& enum_type, PyObject* pytype,
                          PyObject* btype, interop::TCppScope_t enum_constant);
} // namespace cppjit::cpyrt

#endif // !CPYRT_CPPENUM_H
