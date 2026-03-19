#ifndef MOCK_RDF_DEFINE_HEADER
#define MOCK_RDF_DEFINE_HEADER

#include <iostream>
#include <string_view>
#include <string>
#include <vector>

float foo()
{
   return 42;
}

using ColumnNames_t = std::vector<std::string>;

struct MockRDF {
   template <typename F, typename std::enable_if_t<!std::is_convertible<F, std::string>::value, int> = 0>
   void mock_define(std::string_view name, F fun, const ColumnNames_t &columns = {})
   {
      auto val = fun();
      std::cout << "Class method with name " << name << " got value " << val << "\n";
   }
};

#endif // MOCK_RDF_DEFINE_HEADER
