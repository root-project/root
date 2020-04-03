#include "ROOTUnitTestSupport.h"

#include "TClass.h"
#include "TInterpreter.h"
#include "TSystem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Path.h"

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
   ROOT_EXPECT_ERROR(ASSERT_FALSE(gInterpreter->GenerateDictionary("", "")), "TInterpreter::TCling::GenerateDictionary",
                     "Cannot generate dictionary without passing classes.");
   ROOT_EXPECT_ERROR(ASSERT_FALSE(gInterpreter->GenerateDictionary(nullptr, nullptr)),
                     "TInterpreter::TCling::GenerateDictionary", "Cannot generate dictionary without passing classes.");
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

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
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
   // It may depend on tbb
   ASSERT_TRUE(llvm::StringRef(SeenDeps).startswith("GenVector"));
#else
    // Depends only on libCore.so but libCore.so is loaded and thus missing.
    ASSERT_STREQ("GenVector", SeenDeps.c_str());
#endif

   SeenDeps = MakeDepLibsPlatformIndependent(GetLibDeps("libTreePlayer.so"));
   llvm::StringRef SeenDepsRef = SeenDeps;

   // Depending on the configuration we expect:
   // TreePlayer Gpad Graf Graf3d Hist [Imt] [MathCore] MultiProc Net Tree [tbb]..
   // FIXME: We should add a generic gtest regex matcher and use a regex here.
   ASSERT_TRUE(SeenDepsRef.startswith("TreePlayer Gpad Graf Graf3d Hist"));
   ASSERT_TRUE(SeenDepsRef.contains("MultiProc Net Tree"));

   ROOT_EXPECT_ERROR(ASSERT_TRUE(nullptr == GetLibDeps("")), "TCling__GetSharedLibImmediateDepsSlow",
                     "Cannot find library ''");
   ROOT_EXPECT_ERROR(ASSERT_TRUE(nullptr == GetLibDeps("   ")), "TCling__GetSharedLibImmediateDepsSlow",
                     "Cannot find library '   '");
}
#endif

// Check the interface which interacts with the cling::LookupHelper.
TEST_F(TClingTests, ClingLookupHelper) {
  // Exception spec evaluation.
  // Emulate the LookupHelper sequence:
  // auto S = LookupHelper::findScope("ROOT::Internal::RDF", diag)
  // LookupHelper::findAnyFunction(S, "RDataFrameTake<float>", diag)
  // LookupHelper::findAnyFunction(S, "RDataFrameTake<std::vector<float>>", diag)
  auto *cl = gCling->ClassInfo_Factory("ROOT::Internal::RDF");
  gCling->GetFunction(cl, "RDataFrameTake<float>");
  gCling->GetFunction(cl, "RDataFrameTake<std::vector<float>>");
}
