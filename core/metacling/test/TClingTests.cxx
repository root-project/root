#include "TClass.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// Copied from TFileMergerTests.cxx.
// FIXME: Factor out in a new testing library in ROOT.
namespace {
using testing::internal::GetCapturedStderr;
using testing::internal::CaptureStderr;
using testing::internal::RE;
class ExpectedErrorRAII {
   std::string ExpectedRegex;
   void pop()
   {
      std::string Seen = GetCapturedStderr();
      bool match = RE::FullMatch(Seen, RE(ExpectedRegex));
      EXPECT_TRUE(match);
      if (!match) {
         std::string msg = "Match failed!\nSeen: '" + Seen + "'\nRegex: '" + ExpectedRegex + "'\n";
         GTEST_NONFATAL_FAILURE_(msg.c_str());
      }
   }

public:
   ExpectedErrorRAII(std::string E) : ExpectedRegex(E) { CaptureStderr(); }
   ~ExpectedErrorRAII() { pop(); }
};
}

#define EXPECT_ROOT_ERROR(expression, expected_error) \
   {                                                  \
      ExpectedErrorRAII EE(expected_error);           \
      expression;                                     \
   }

// FIXME: We should probably have such a facility in TCling.
static void cleanup()
{
   // Remove AutoDict
   void* dir = gSystem->OpenDirectory(gSystem->pwd());
   const char* name = 0;
   while ((name = gSystem->GetDirEntry(dir)))
      if (!strncmp(name, "AutoDict_", 9))
         gSystem->Unlink(name);

   gSystem->FreeDirectory(dir);
}

class TClingTests : public ::testing::Test {
protected:
   // virtual void SetUp() { }

   // FIXME: We cannot rely on TearDown because it is executed at the end of
   // every test. This triggers another bug in the dictionary generation phase,
   // possibly due to concurrent file system operations.
   //virtual void TearDown() {
      // If there are failures we want to keep the created files.
      //if (!::testing::Test::HasFatalFailure())
      //   cleanup();
   //}

};

// FIXME: Merge with TearDown.
struct CleanupRAII {
   CleanupRAII() {
      if (!::testing::Test::HasFatalFailure())
         cleanup();
   }
} Cleanup;

TEST_F(TClingTests, GenerateDictionaryErrorHandling)
{
   // Check error reporting and handling.
   EXPECT_ROOT_ERROR(ASSERT_FALSE(gInterpreter->GenerateDictionary("", "")),
                     "Error in .* Cannot generate dictionary without passing classes.\n");
   EXPECT_ROOT_ERROR(ASSERT_FALSE(gInterpreter->GenerateDictionary(nullptr, nullptr)),
                     "Error in .* Cannot generate dictionary without passing classes.\n");
}

TEST_F(TClingTests, GenerateDictionaryRegression)
{
   // Make sure we do not crash or go in an infinite loop.
   ASSERT_TRUE(gInterpreter->GenerateDictionary("std::set<int>"));
   ASSERT_TRUE(gInterpreter->GenerateDictionary("std::set<int>", ""));
   ASSERT_TRUE(gInterpreter->GenerateDictionary("std::set<int>", "set"));

   // FIXME: This makes the linkdef parser go in an infinite loop.
   //ASSERT_TRUE(gInterpreter->GenerateDictionary("std::vector<std::array<int, 5>>", ""));
}

TEST_F(TClingTests, GenerateDictionary)
{
   auto cl = TClass::GetClass("vector<TNamed*>");
   ASSERT_FALSE(cl && cl->IsLoaded());

   ASSERT_TRUE(gInterpreter->GenerateDictionary("std::vector<TNamed*>"));
   cl = TClass::GetClass("vector<TNamed*>");
   ASSERT_TRUE(cl != nullptr);
   ASSERT_TRUE(cl->IsLoaded());
}

// Test ROOT-6967
TEST_F(TClingTests, GetEnumWithSameVariableName)
{
   gInterpreter->ProcessLine("int en;enum en{kNone};");
   auto en = gInterpreter->GetEnum(nullptr, "en");
   ASSERT_TRUE(en != nullptr);
}

// Check if we can get the source code of function definitions.
TEST_F(TClingTests, MakeInterpreterValue)
{
   gInterpreter->Declare("void my_func_to_print() {}");
   std::unique_ptr<TInterpreterValue> v = gInterpreter->MakeInterpreterValue();
   gInterpreter->Evaluate("my_func_to_print", *v);
   ASSERT_THAT(v->ToString(), testing::HasSubstr("void my_func_to_print"));
}

static std::string MakeLibNamePlatformIndependent(llvm::StringRef libName)
{
   if (libName.empty())
      return {};
   EXPECT_TRUE(libName.startswith("lib"));
   EXPECT_TRUE(llvm::sys::path::has_extension(libName));
   libName.consume_front("lib");
   // Remove the extension.
   return libName.substr(0, libName.find_first_of('.')).str();
}

// Check if the heavily used interface in TCling::AutoLoad returns consistent
// results.
TEST_F(TClingTests, GetClassSharedLibs)
{
   // Shortens the invocation.
   auto GetLibs = [](const char *cls) -> const char * {
      return gInterpreter->GetClassSharedLibs(cls);
   };

   llvm::StringRef lib = GetLibs("TLorentzVector");
   ASSERT_STREQ("Physics", MakeLibNamePlatformIndependent(lib).c_str());

   // FIXME: This should return GenVector. The default args of the LorentzVector
   // are shadowed by Vector4Dfwd.h.
   lib = GetLibs("ROOT::Math::LorentzVector");
   ASSERT_STREQ("", MakeLibNamePlatformIndependent(lib).c_str());

   lib = GetLibs("ROOT::Math::PxPyPzE4D<float>");
   ASSERT_STREQ("GenVector", MakeLibNamePlatformIndependent(lib).c_str());

   // FIXME: We should probably resolve again to GenVector as it contains the
   // template pattern.
   lib = GetLibs("ROOT::Math::PxPyPzE4D<int>");
   ASSERT_STREQ("", MakeLibNamePlatformIndependent(lib).c_str());

   lib = GetLibs("vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >");
   ASSERT_STREQ("GenVector", MakeLibNamePlatformIndependent(lib).c_str());

   lib = GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ");
#ifdef R__USE_CXXMODULES
   ASSERT_STREQ("GenVector", MakeLibNamePlatformIndependent(lib).c_str());
#else
   // FIXME: This is another bug in the non-modules functionality. Note the
   // trailing space...
   ASSERT_STREQ("", MakeLibNamePlatformIndependent(lib).c_str());
#endif

   // FIXME: Another bug in non-modules:
   // GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >")
   //    != GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>")
   // note the missing space.
}

static std::string MakeDepLibsPlatformIndependent(llvm::StringRef libs) {
   llvm::SmallVector<llvm::StringRef, 32> splitLibs;
   libs.trim().split(splitLibs, ' ');
   assert(!splitLibs.empty());
   std::string result = MakeLibNamePlatformIndependent(splitLibs[0]) + ' ';
   splitLibs.erase(splitLibs.begin());

   std::sort(splitLibs.begin(), splitLibs.end());
   for (llvm::StringRef lib : splitLibs)
      result += MakeLibNamePlatformIndependent(lib.trim()) + ' ';

   return llvm::StringRef(result).rtrim();
}

// Check the interface computing the dependencies of a given library.
TEST_F(TClingTests, GetSharedLibDeps)
{
   // Shortens the invocation.
   auto GetLibDeps = [](const char *lib) -> const char* {
      return gInterpreter->GetSharedLibDeps(lib, /*tryDyld*/true);
   };

   std::string SeenDeps
      = MakeDepLibsPlatformIndependent(GetLibDeps("libGenVector.so"));
#ifdef R__MACOSX
   ASSERT_TRUE(llvm::StringRef(SeenDeps).startswith("GenVector New"));
#else
   // Depends only on libCore.so but libCore.so is loaded and thus missing.
   ASSERT_STREQ("GenVector", SeenDeps.c_str());
#endif

   SeenDeps = MakeDepLibsPlatformIndependent(GetLibDeps("libTreePlayer.so"));
   llvm::StringRef SeenDepsRef = SeenDeps;

   // Depending on the configuration we expect:
   // TreePlayer Gpad Graf Graf3d Hist [Imt] [MathCore] MultiProc Net [New] Tree [tbb]..
   // FIXME: We should add a generic gtest regex matcher and use a regex here.
   ASSERT_TRUE(SeenDepsRef.startswith("TreePlayer Gpad Graf Graf3d Hist"));
   ASSERT_TRUE(SeenDepsRef.contains("MultiProc Net"));
   ASSERT_TRUE(SeenDepsRef.contains("Tree"));

   EXPECT_ROOT_ERROR(ASSERT_TRUE(nullptr == GetLibDeps("")),
                     "Error in <TCling__GetSharedLibImmediateDepsSlow>: Cannot find library ''\n");
   EXPECT_ROOT_ERROR(ASSERT_TRUE(nullptr == GetLibDeps("   ")),
                     "Error in <TCling__GetSharedLibImmediateDepsSlow>: Cannot find library '   '\n");
}
