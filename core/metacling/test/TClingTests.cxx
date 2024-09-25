#include "ROOT/TestSupport.hxx"

#include "TClass.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TSystem.h"

#include "gmock/gmock.h"

#include <sstream>
#include <string>
#include <vector>

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
   ROOT_EXPECT_ERROR(EXPECT_FALSE(gInterpreter->GenerateDictionary("", "")), "TInterpreter::TCling::GenerateDictionary",
                     "Cannot generate dictionary without passing classes.");
   ROOT_EXPECT_ERROR(EXPECT_FALSE(gInterpreter->GenerateDictionary(nullptr, nullptr)),
                     "TInterpreter::TCling::GenerateDictionary", "Cannot generate dictionary without passing classes.");
}

TEST_F(TClingTests, GenerateDictionaryRegression)
{
   // Make sure we do not crash or go in an infinite loop.
   EXPECT_TRUE(gInterpreter->GenerateDictionary("std::set<int>"));
   EXPECT_TRUE(gInterpreter->GenerateDictionary("std::set<int>", ""));
   EXPECT_TRUE(gInterpreter->GenerateDictionary("std::set<int>", "set"));

   // FIXME: This makes the linkdef parser go in an infinite loop.
   //EXPECT_TRUE(gInterpreter->GenerateDictionary("std::vector<std::array<int, 5>>", ""));
}

#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
TEST_F(TClingTests, GenerateDictionary)
{
   auto cl = TClass::GetClass("vector<TNamed*>");
   EXPECT_FALSE(cl && cl->IsLoaded());

   EXPECT_TRUE(gInterpreter->GenerateDictionary("std::vector<TNamed*>"));
   cl = TClass::GetClass("vector<TNamed*>");
   EXPECT_TRUE(cl != nullptr);
   EXPECT_TRUE(cl->IsLoaded());
}
#endif

// Test ROOT-6967
TEST_F(TClingTests, GetEnumWithSameVariableName)
{
   gInterpreter->ProcessLine("int en;enum en{kNone};");
   auto en = gInterpreter->GetEnum(nullptr, "en");
   EXPECT_TRUE(en != nullptr);
}

TEST_F(TClingTests, SuccessfulAutoInjection)
{
   gInterpreter->ProcessLine("int SuccessfulAutoInjectionTest(int j) { return 0; }");
   gInterpreter->ProcessLine("success = SuccessfulAutoInjectionTest(3)");

   auto success = gInterpreter->GetDataMember(nullptr, "success");
   EXPECT_TRUE(success != nullptr);
}

TEST_F(TClingTests, FailedAutoInjection)
{
   gInterpreter->ProcessLine("int FailedAutoInjectionTest(int j) { return 0; }");
   gInterpreter->ProcessLine("failed = FailedAutoInjectionTest(3, 6)");

   auto failed = gInterpreter->GetDataMember(nullptr, "failed");
   EXPECT_TRUE(failed == nullptr);
}

// Check if we can get the source code of function definitions.
TEST_F(TClingTests, MakeInterpreterValue)
{
   gInterpreter->Declare("void my_func_to_print() {}");
   std::unique_ptr<TInterpreterValue> v = gInterpreter->MakeInterpreterValue();
   gInterpreter->Evaluate("my_func_to_print", *v);
   EXPECT_THAT(v->ToString(), testing::HasSubstr("void my_func_to_print"));
}

static std::string MakeLibNamePlatformIndependent(const std::string &libName)
{
   // Return an empty string if input is not a library name.
   // Sometimes, libName can be the binary name (i.e. TClingTest, for this test)
   if (libName.empty() || libName.compare(0, 3, "lib") != 0)
      return {};

   // Return library name without lib prefix and extension.
   return libName.substr(3, libName.find('.') - 3);
}

// Check if the heavily used interface in TCling::AutoLoad returns consistent
// results.
TEST_F(TClingTests, GetClassSharedLibs)
{
   // Shortens the invocation.
   auto GetLibs = [](const char *cls) -> std::string {
      if (const char *val = gInterpreter->GetClassSharedLibs(cls,false))
         return val;
      return "";
   };

   std::string lib = GetLibs("TLorentzVector");
   EXPECT_EQ("Physics", MakeLibNamePlatformIndependent(lib));

   // FIXME: This should return GenVector. The default args of the LorentzVector
   // are shadowed by Vector4Dfwd.h.
   lib = GetLibs("ROOT::Math::LorentzVector");
   EXPECT_EQ("", MakeLibNamePlatformIndependent(lib));

   lib = GetLibs("ROOT::Math::PxPyPzE4D<float>");
   EXPECT_EQ("GenVector", MakeLibNamePlatformIndependent(lib));

   // FIXME: We should probably resolve again to GenVector as it contains the
   // template pattern.
   lib = GetLibs("ROOT::Math::PxPyPzE4D<int>");
   EXPECT_EQ("", MakeLibNamePlatformIndependent(lib));

   lib = GetLibs("vector<ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > >");
   EXPECT_EQ("GenVector", MakeLibNamePlatformIndependent(lib));

   lib = GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<double> > ");
#ifdef R__USE_CXXMODULES
   EXPECT_EQ("GenVector", MakeLibNamePlatformIndependent(lib));
#else
   // FIXME: This is another bug in the non-modules functionality. Note the
   // trailing space...
   EXPECT_EQ("", MakeLibNamePlatformIndependent(lib));
#endif

   // FIXME: Another bug in non-modules:
   // GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float> >")
   //    != GetLibs("ROOT::Math::LorentzVector<ROOT::Math::PxPyPzE4D<float>>")
   // note the missing space.
}

static std::string MakeDepLibsPlatformIndependent(const std::string &libs) {
   auto trim = [](const std::string &s) {
      std::string ret = s;
      while (!ret.empty() && std::isspace(ret[0]))
         ret.erase(0, 1);
      while (!ret.empty() && std::isspace(ret[ret.size() - 1]))
         ret.erase(ret.size() - 1, 1);
      return ret;
   };

   auto split = [](const std::string &s) -> std::vector<std::string> {
      std::vector<std::string> ret;
      std::istringstream istr(s);
      std::string part;
      while (std::getline(istr, part, ' '))
         ret.push_back(part);
      return ret;
   };

   std::vector<std::string> splitLibs = split(trim(libs));
   assert(!splitLibs.empty());

   std::sort(splitLibs.begin() + 1, splitLibs.end());
   std::transform(splitLibs.begin(), splitLibs.end(), splitLibs.begin(), MakeLibNamePlatformIndependent);

   std::string result;
   for (std::string lib : splitLibs)
      if (!lib.empty())
         result += lib + ' ';

   return trim(result);
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
   EXPECT_EQ(SeenDeps.substr(0, 9), "GenVector");
#else
    // Depends only on libCore.so but libCore.so is loaded and thus missing.
    EXPECT_STREQ("GenVector", SeenDeps.c_str());
#endif

   SeenDeps = MakeDepLibsPlatformIndependent(GetLibDeps("libTreePlayer.so"));
   std::string SeenDepsRef = SeenDeps;

   // Depending on the configuration we expect:
   // TreePlayer Gpad Graf Graf3d Hist [Imt] [MathCore] MultiProc Net Tree [tbb]..
   // FIXME: We should add a generic gtest regex matcher and use a regex here.
   EXPECT_EQ(SeenDepsRef.compare(0, 32, "TreePlayer Gpad Graf Graf3d Hist"), 0);
   EXPECT_NE(SeenDepsRef.find("MultiProc Net Tree"), std::string::npos);

   ROOT_EXPECT_ERROR(EXPECT_TRUE(nullptr == GetLibDeps("")), "TCling__GetSharedLibImmediateDepsSlow",
                     "Cannot find library ''");
   ROOT_EXPECT_ERROR(EXPECT_TRUE(nullptr == GetLibDeps("   ")), "TCling__GetSharedLibImmediateDepsSlow",
                     "Cannot find library '   '");
}
#endif

// Check that a warning message is generated when using auto-injection.
TEST_F(TClingTests, WarningAutoInjection)
{
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kWarning, "cling", "declaration without the 'auto' keyword is deprecated",
                      /*matchFullMessage=*/false);

   gInterpreter->ProcessLine("/* no auto */ t = new int;");

   auto t = gInterpreter->GetDataMember(nullptr, "t");
   EXPECT_TRUE(t != nullptr);
}

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


// Check that compiled and interpreted statics share the same address.
TEST_F(TClingTests, ROOT10499) {
#if !defined(_MSC_VER) || defined(R__ENABLE_BROKEN_WIN_TESTS)
   EXPECT_EQ((void*)&std::cout, (void*)gInterpreter->Calc("&std::cout"));
   EXPECT_EQ((void*)&std::cerr, (void*)gInterpreter->Calc("&std::cerr"));
   // strangely enough, this works on the command prompt, but not in this test...
   EXPECT_EQ((void*)&errno, (void*)gInterpreter->Calc("&errno"));
#endif
}

// ROOT-6913
TEST_F(TClingTests, ClassInfoProperty)
{
   gInterpreter->Declare(R"cpp(
class ClassHasExplicitCtor{
   public:
      ClassHasExplicitCtor(){};
};

class ClassHasExplicitDtor{
   public:
      ~ClassHasExplicitDtor(){};
};

class ClassHasImplicitCtor {
   ClassHasExplicitCtor c;
};

class ClassHasImplicitDtor{
   ClassHasExplicitDtor c;
};

class ClassHasDefaultCtor{
   ClassHasDefaultCtor(){};
};

class ClassIsAbstract{
   virtual void PureVirtual() = 0;
};

class ClassHasVirtual{
   virtual void Virtual() {};
};

class ClassHasAssignOpr{
   ClassHasAssignOpr& operator=(const ClassHasAssignOpr&);
};

// according to the C++ standard, being a POD implies being an aggregate
struct ClassIsAggregate {
    int x;
    double y;
};
                          )cpp");

   const std::vector<std::pair<std::string, Long_t>> classNPPairs{
      {"ClassHasImplicitCtor", kClassHasImplicitCtor}, {"ClassHasExplicitCtor", kClassHasExplicitCtor},
      {"ClassHasExplicitDtor", kClassHasExplicitDtor}, {"ClassHasImplicitDtor", kClassHasImplicitDtor},
      {"ClassHasDefaultCtor", kClassHasDefaultCtor},   {"ClassHasDefaultCtor", kClassIsValid},
      {"ClassIsAbstract", kClassIsAbstract},           {"ClassHasVirtual", kClassHasVirtual},
      {"ClassHasAssignOpr", kClassHasAssignOpr},       {"ClassIsAggregate", kClassIsAggregate}};

   for (auto &[clName, clPropRef] : classNPPairs) {
      auto cl = TClass::GetClass(clName.c_str());
      const auto prop = gInterpreter->ClassInfo_ClassProperty(cl->GetClassInfo());
      EXPECT_TRUE(prop & clPropRef) << "Error checking property for class " << clName;
   }
}

// #12108
TEST_F(TClingTests, constexprFunctionReturn)
{
   gInterpreter->Declare(R"cpp(
namespace issue_12108{

template <typename T> struct ValueAndPushforward {
  T value;
};

constexpr ValueAndPushforward<double> Ln10_pushforward() {
    return {2.3025850929940459};
}

ValueAndPushforward<double> Ln10_pushforwardNoconstexpr() {
    return {2.3025850929940459};
}

auto val1 = Ln10_pushforwardNoconstexpr();
auto val2 = Ln10_pushforward();

}
   )cpp");

   auto val1 = *(double *)(uintptr_t)gInterpreter->ProcessLine("&issue_12108::val1.value;");
   auto val2 = *(double *)(uintptr_t)gInterpreter->ProcessLine("&issue_12108::val2.value;");

   EXPECT_DOUBLE_EQ(2.3025850929940459, val1);
   EXPECT_DOUBLE_EQ(2.3025850929940459, val2);
}

// #15511
TEST_F(TClingTests, ManyConstConstructors)
{
   std::vector<int> constructors;
   std::ostringstream declareConstructors;
   declareConstructors << "namespace issue_15511 { auto constructors = "
                       << "reinterpret_cast<std::vector<int> *>(" << reinterpret_cast<uintptr_t>(&constructors)
                       << "); }";
   gInterpreter->Declare(declareConstructors.str().c_str());

   gInterpreter->Declare(R"cpp(
namespace issue_15511 {
struct Constructor {
  Constructor(int value) { constructors->push_back(value); }
};

const Constructor c0(0);
const Constructor c1(1);
const Constructor c2(2);
const Constructor c3(3);
const Constructor c4(4);
const Constructor c5(5);
const Constructor c6(6);
const Constructor c7(7);
const Constructor c8(8);
const Constructor c9(9);
const Constructor c10(10);
const Constructor c11(11);
const Constructor c12(12);
const Constructor c13(13);
const Constructor c14(14);
const Constructor c15(15);
const Constructor c16(16);
const Constructor c17(17);
const Constructor c18(18);
const Constructor c19(19);
}
   )cpp");

   ASSERT_EQ(constructors.size(), 20);
   for (int i = 0; i < 20; i++) {
      EXPECT_EQ(constructors[i], i);
   }
}

// #8828
TEST_F(TClingTests, RefreshNSShadowing)
{
   // These two lines would make the test fail because of two reasons:
   // - An assertion failure "Assertion failed: (detail::isPresent(Val) && "dyn_cast on a non-existent value"), function dyn_cast, file Casting.h, line 662."
   // - An error "Error in <TInterpreter::RefreshClassInfo>: Should not need to update the classInfo a non type decl: Detail"
   // This is why there is no check performed.
   ROOT::TestSupport::CheckDiagsRAII diags;
   diags.requiredDiag(kError, "TInterpreter::RefreshClassInfo", "Should not need to update the classInfo a non type decl: Detail");
   gInterpreter->Declare("namespace std { namespace Detail {} }; auto c = TClass::GetClass(\"Detail\");");
   gInterpreter->ProcessLine("namespace Detail {}");
}

// #8367
TEST_F(TClingTests, UndeclaredIdentifierCrash)
{
   auto expectedError = R"(error: use of undeclared identifier 'i'
 for(i=0; i < 0;); // the second usage of `i` was enough to get a segfault
          ^
)";
   using namespace ROOT::TestSupport;
   CheckDiagsRAII diagRAII;
   diagRAII.requiredDiag(kError, "cling", expectedError, false);
   gInterpreter->ProcessLine("for(i=0; i < 0;); // the second usage of `i` was enough to get a segfault");
}
