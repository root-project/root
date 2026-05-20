#ifndef MYLIB
#define MYLIB

#include <TGeoManager.h>

namespace mylib {
class MyClass {
public:
   using MyId = int;

   static constexpr MyId A = 2;

   static std::string StaticFunc(MyId dummy2, const char *dummy) { return "asdasd"; }
};
} // namespace mylib

#endif
