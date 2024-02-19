#include "gtest/gtest.h"
#include "TSystem.h"

#include <string>

TEST(TSystem, IncludePath)
{
   ASSERT_TRUE(gSystem);

   gSystem->AddIncludePath("-I /some/path/with-xin-it -I ./some/relative-path");
   gSystem->AddIncludePath("-I %ROOTSYS%\\include -I ${ROOTSYS}/include");
#ifdef WIN32
   gSystem->AddIncludePath(
      "-I \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\"");
#endif

   testing::internal::CaptureStderr();
   gSystem->CompileMacro("Foo.C", "f");
   std::cerr.flush();
   std::string errors = testing::internal::GetCapturedStderr();

   gSystem->Unload("Foo_C");
   gSystem->CleanCompiledMacros();
#ifdef WIN32
   EXPECT_TRUE((errors.find("cl : Command line warning D9002 : ignoring unknown option ") == std::string::npos));
   gSystem->Exec("del Foo_C*");
#else
   EXPECT_TRUE((errors.find("c++: error: unrecognized command line option ") == std::string::npos));
   gSystem->Exec("rm -f Foo_C*");
#endif
}
