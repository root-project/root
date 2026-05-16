// Hand-written C API wrappers for functions the TableGen emitter
// cannot generate mechanically. Generated wrappers live in
// CXCppInterOpGenerated.cpp.

#include "CppInterOp/CppInterOp.h"

#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

// GetClassTemplatedMethods returns bool AND fills a vector out-param.
// The C wrapper drops the bool (caller checks arr.size > 0 instead).
CPPINTEROP_API Cpp::CppInterOpArray
cppinterop_GetClassTemplatedMethods(const char* name, void* parent) {
  std::vector<Cpp::TCppFunction_t> out;
  Cpp::GetClassTemplatedMethods(std::string(name), parent, out);
  Cpp::CppInterOpArray arr = {nullptr, out.size()};
  if (arr.size) {
    arr.data = static_cast<void**>(malloc(arr.size * sizeof(void*)));
    memcpy(arr.data, out.data(), arr.size * sizeof(void*));
  }
  return arr;
}

} // extern "C"
