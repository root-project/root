#include "gtest/gtest.h"
#include "TSystem.h"
#include "TString.h"

#include <string>

#ifdef WIN32
namespace {
struct ComspecGuard {
   TString fOriginal;

   ~ComspecGuard()
   {
      gSystem->Setenv("COMSPEC", fOriginal);
   }
};

struct DebugGuard {
   Int_t fOriginal = gDebug;

   ~DebugGuard()
   {
      gDebug = fOriginal;
   }
};
} // namespace
#endif

TEST(TSystem, ExecWithComspec)
{
#ifdef WIN32
   ASSERT_TRUE(gSystem);

   const char *comspec = gSystem->Getenv("COMSPEC");
   ASSERT_NE(comspec, nullptr);
   TString cmdPath = comspec;
   if (cmdPath.BeginsWith("\"") && cmdPath.EndsWith("\"")) {
      cmdPath.Remove(0, 1);
      cmdPath.Chop();
   }
   TString normalizedCmdPath = cmdPath;
   normalizedCmdPath.ToLower();
   ASSERT_TRUE(normalizedCmdPath.EndsWith("cmd.exe")) << "This regression test needs cmd.exe as the process shell";

   ComspecGuard comspecGuard{comspec};
   gSystem->Setenv("COMSPEC", cmdPath);
   ASSERT_EQ(0, gSystem->Exec("exit 0"));
#endif
}

TEST(TSystem, QuotedComspecUsesTemporaryStderr)
{
#ifdef WIN32
   ASSERT_TRUE(gSystem);

   const char *comspec = gSystem->Getenv("COMSPEC");
   ASSERT_NE(comspec, nullptr);
   TString cmdPath = comspec;
   if (cmdPath.BeginsWith("\"") && cmdPath.EndsWith("\"")) {
      cmdPath.Remove(0, 1);
      cmdPath.Chop();
   }
   TString normalizedCmdPath = cmdPath;
   normalizedCmdPath.ToLower();
   ASSERT_TRUE(normalizedCmdPath.EndsWith("cmd.exe")) << "This regression test needs cmd.exe as the process shell";

   ComspecGuard comspecGuard{comspec};
   TString quotedCmdPath = TString::Format("\"%s\"", cmdPath.Data());
   gSystem->Setenv("COMSPEC", quotedCmdPath);

   DebugGuard debugGuard;
   gDebug = 5;
   testing::internal::CaptureStderr();
   gSystem->CompileMacro("Foo.C", "f");
   std::cerr.flush();
   std::string errors = testing::internal::GetCapturedStderr();

   EXPECT_NE(errors.find(".stderr.tmp 2>&1"), std::string::npos);
   gSystem->Unload("Foo_C");
   gSystem->CleanCompiledMacros();
#endif
}

TEST(TSystem, IncludePath)
{
   ASSERT_TRUE(gSystem);

   gSystem->AddIncludePath("-I /some/path/with-xin-it -I ./some/relative-path");
   gSystem->AddIncludePath("-I %ROOTSYS%\\include -I ${ROOTSYS}/include");
#ifdef WIN32
   gSystem->AddIncludePath(
      "-I \"C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.38.33130\\include\"");
#endif

#ifdef WIN32
   const char *comspec = gSystem->Getenv("COMSPEC");
   ASSERT_NE(comspec, nullptr);
   TString cmdPath = comspec;
   if (cmdPath.BeginsWith("\"") && cmdPath.EndsWith("\"")) {
      cmdPath.Remove(0, 1);
      cmdPath.Chop();
   }
   TString normalizedCmdPath = cmdPath;
   normalizedCmdPath.ToLower();
   ASSERT_TRUE(normalizedCmdPath.EndsWith("cmd.exe")) << "This regression test needs cmd.exe as the process shell";

   ComspecGuard comspecGuard{comspec};
   gSystem->Setenv("COMSPEC", cmdPath);
#endif

   testing::internal::CaptureStderr();
   const auto compileResult = gSystem->CompileMacro("Foo.C", "f");
   std::cerr.flush();
   std::string errors = testing::internal::GetCapturedStderr();

   EXPECT_TRUE(compileResult);
   gSystem->Unload("Foo_C");
   gSystem->CleanCompiledMacros();
#ifdef WIN32
   EXPECT_TRUE((errors.find("cl : Command line warning D9002 : ignoring unknown option ") == std::string::npos));
   // This will need to be updated once we no longer rely on rmkdepend.
   EXPECT_EQ(errors.find("Failed to run rmkdepend"), std::string::npos);
   gSystem->Exec("del Foo_C*");
#else
   EXPECT_TRUE((errors.find("c++: error: unrecognized command line option ") == std::string::npos));
   gSystem->Exec("rm -f Foo_C*");
#endif
}
