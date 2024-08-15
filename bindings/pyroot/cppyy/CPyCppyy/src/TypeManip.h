#ifndef CPYCPPYY_TYPEMANIP_H
#define CPYCPPYY_TYPEMANIP_H

#include <string>
#include <vector>


namespace CPyCppyy {

namespace TypeManip {

    std::string remove_const(const std::string& cppname);
    std::string clean_type(const std::string& cppname,
            bool template_strip = true, bool const_strip = true);
    std::string template_base(const std::string& cppname);
    std::string compound(const std::string& name);

    void cppscope_to_pyscope(std::string& cppscope);
    void cppscope_to_legalname(std::string& cppscope);
    std::string extract_namespace(const std::string& name);

    std::vector<std::string> extract_arg_types(const std::string& sig);
    Py_ssize_t array_size(const std::string& name);

} // namespace TypeManip

} // namespace CPyCppyy

#endif // !CPYCPPYY_TYPEMANIP_H
