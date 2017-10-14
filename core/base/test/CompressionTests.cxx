#include "TInterpreter.h"
#include "TInterpreterValue.h"

#include "gtest/gtest.h"

// FIXME: Port to windows.
#include <dlfcn.h>

// This test is designed to check if the compile time and runtime understanding of ROOT about zlib
// match. We want to make sure that libCore.so contains the symbols coming from zlib.h and the cling
// resolves them.
TEST(ZLib, Sanity)
{
   void *func_handle = dlopen("libCore.so", RTLD_LAZY);
   ASSERT_TRUE(nullptr != func_handle);
   typedef const char *(*zlibVersion_t)();
   zlibVersion_t zlibVersion = (zlibVersion_t)dlsym(func_handle, "zlibVersion");
   const char *libCoreVersion = zlibVersion();
   dlclose(func_handle);

   ASSERT_STRNE(nullptr, libCoreVersion);

   gInterpreter->Declare("#include <zlib.h>");
   TInterpreterValue *headerVersion = gInterpreter->CreateTemporary();
   bool success = (bool)gInterpreter->Evaluate("ZLIB_VERSION", *headerVersion);
   TInterpreterValue *clingVersion = gInterpreter->CreateTemporary();
   success &= (bool)gInterpreter->Evaluate("zlibVersion()", *clingVersion);
   ASSERT_TRUE(success);

   ASSERT_STREQ(libCoreVersion, (const char *)headerVersion->GetAsPointer());
   ASSERT_STREQ(libCoreVersion, (const char *)clingVersion->GetAsPointer());
}
