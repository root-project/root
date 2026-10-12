#include "TInterpreter.h"

#include "gtest/gtest.h"

#include <dlfcn.h>

using GetProcAddress_t = void (*(const char *))();

// A CppInterOp client such as cppjit loads the library that exports
// CppGetProcAddress and looks the API up by name. Through libCore it must get
// the functions of ROOT's interpreter.
TEST(TClingDispatch, CppGetProcAddressFromLibCore)
{
   void *libCore = dlopen(ROOT_LIBCORE, RTLD_LAZY | RTLD_LOCAL);
   ASSERT_NE(libCore, nullptr) << dlerror();
   auto getProcAddress = reinterpret_cast<GetProcAddress_t *>(dlsym(libCore, "CppGetProcAddress"));
   ASSERT_NE(getProcAddress, nullptr) << dlerror();

   auto process = reinterpret_cast<int (*)(const char *)>(getProcAddress("Process"));
   ASSERT_NE(process, nullptr);
   EXPECT_EQ(process("int TClingDispatchValue = 42;"), 0);
   EXPECT_EQ(gInterpreter->Calc("TClingDispatchValue"), 42);
   EXPECT_EQ(getProcAddress("NoSuchFunction"), nullptr);

   void *libCling = dlopen(ROOT_LIBCLING, RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
   ASSERT_NE(libCling, nullptr) << "libCling should be loaded once, by ROOT";
   auto fromLibCling = reinterpret_cast<GetProcAddress_t *>(dlsym(libCling, "CppGetProcAddress"));
   ASSERT_NE(fromLibCling, nullptr) << dlerror();
   EXPECT_EQ(getProcAddress("Process"), fromLibCling("Process"));
}
