#include <CppInterOp/Dispatch.h>

#define DISPATCH_API(name, type) CppAPIType::name Cpp::name = nullptr;
CPPINTEROP_API_TABLE
#undef DISPATCH_API
