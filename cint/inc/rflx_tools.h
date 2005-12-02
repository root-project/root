//$Id: rflx_tools.h,v 1.1 2005/11/16 14:58:14 roiser Exp $

#ifndef RFLX_TOOLS_H
#define RFLX_TOOLS_H 1

#include <string>

class rflx_tools {

 public:
  
  static std::string escape_class_name(const std::string & name);
  static std::string rm_end_ref(const std::string & name);
  static std::string decorate_stl_type(const std::string & name);
  static std::string stub_type_name(const std::string & name);
  static std::string un_const(const std::string & name);

};

#endif // RFLX_TOOLS_H
