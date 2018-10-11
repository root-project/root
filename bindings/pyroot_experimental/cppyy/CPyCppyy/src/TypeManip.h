#ifndef CPYCPPYY_TYPEMANIP_H
#define CPYCPPYY_TYPEMANIP_H

#include <string>


namespace CPyCppyy {

namespace TypeManip {

    std::string remove_const(const std::string& cppname);
    std::string clean_type(const std::string& cppname,
            bool template_strip = true, bool const_strip = true);

    void cppscope_to_pyscope(std::string& cppscope);

} // namespace TypeManip

} // namespace CPyCppyy

#endif // !CPYCPPYY_TYPEMANIP_H
