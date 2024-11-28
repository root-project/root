#include "ROOT/TestSupport.hxx"
#include "TInterpreter.h"
#include "TCollection.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

// These tests check the generated wrapper functions that allow calling C++ functions.
// via C interface.
// Usually we only test that a wrapper function compiles correctly, but sometimes
// we also do some further checks on the wrapper here. See the specific assertions for more
// explanation on this.
// NOTE: This wrapper interface should be replaced in the future with a proper
// way of calling these functions that doesn't require parsing generated strings
// of C++ code, so if these tests fail because this interface was replaced by another
// system, feel free to delete them as these tests here don't represent things the user
// should do in his code.

// A class that creates a CallFunc and deletes it at the end of its scope.
class CallFuncRAII {
  ClassInfo_t *fScope = nullptr;
  CallFunc_t *fmc = nullptr;
public:
  CallFuncRAII(const char *scopeName, const char* method, const char* proto) {
    fScope = gInterpreter->ClassInfo_Factory(scopeName);
    fmc = gInterpreter->CallFunc_Factory();
    Longptr_t offset = 0;
    gInterpreter->CallFunc_SetFuncProto(fmc, fScope, method, proto, &offset);
  }
  ~CallFuncRAII() {
   // Cleanup
    gInterpreter->CallFunc_Delete(fmc);
    gInterpreter->ClassInfo_Delete(fScope);
  }

  std::string GetWrapper() {
    return gInterpreter->CallFunc_GetWrapperCode(fmc);
  }

  CallFunc_t * GetCF() const { return fmc; }
};

// root-project/root#11930
TEST(TClingCallFunc, Void)
{
   gInterpreter->Declare(R"cpp(
                           void ExecVoid() { }
                           )cpp");

   CallFuncRAII CfRAII("", "ExecVoid", "");

   // Should not crash
   long result = gInterpreter->CallFunc_ExecInt(CfRAII.GetCF(), /*address*/0);
   ASSERT_TRUE(result == 0);

};

TEST(TClingCallFunc, Conversions)
{
   gInterpreter->Declare(R"cpp(
                           bool CheckFunc() { return false; }
                           )cpp");

   CallFuncRAII CfRAII("", "CheckFunc", "");
   // Here the bool return result is represented as a unsigned int, however,
   // long has a different representation and if we do not cast it propertly
   // the bits are `1` instead `0`.
   long result = gInterpreter->CallFunc_ExecInt(CfRAII.GetCF(), /*address*/0);
   ASSERT_TRUE(result == 0);
}

TEST(TClingCallFunc, FunctionWrapper)
{
   gInterpreter->Declare(R"cpp(
                           bool FunctionWrapperFunc(int b, float f) { return false; }
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFunc", "int, float");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   // Test that we cast this function to the right function type.
   ASSERT_TRUE(wrapper.find("((bool (&)(int, float))FunctionWrapperFunc)") != wrapper.npos);
}

TEST(TClingCallFunc, FunctionWrapperPointer)
{
   gInterpreter->Declare(R"cpp(
                           int *FunctionWrapperFuncPtr(int b, float f) { return nullptr; }
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFuncPtr", "int, float");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   // Test that we cast this function to the right function type.
   auto FirstCallPos = wrapper.find("((int* (&)(int, float))FunctionWrapperFuncPtr)");
   ASSERT_TRUE(FirstCallPos != wrapper.npos);
   auto SecondCallPos = wrapper.find("((int* (&)(int, float))FunctionWrapperFuncPtr)", FirstCallPos + 1);
   ASSERT_TRUE(SecondCallPos != wrapper.npos);
}

TEST(TClingCallFunc, FunctionWrapperReference)
{
   gInterpreter->Declare(R"cpp(
                           int &FunctionWrapperFuncRef(int* b, float f) { static int j; return j; }
                           )cpp");


   CallFuncRAII CfRAII("", "FunctionWrapperFuncRef", "int*, float");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   // Test that we cast this function to the right function type.
   auto FirstCallPos = wrapper.find("((int& (&)(int *, float))FunctionWrapperFuncRef)");
   ASSERT_TRUE(FirstCallPos != wrapper.npos);
   auto SecondCallPos = wrapper.find("((int& (&)(int *, float))FunctionWrapperFuncRef)", FirstCallPos + 1);
   ASSERT_TRUE(SecondCallPos != wrapper.npos);
}

TEST(TClingCallFunc, FunctionWrapperVoid)
{
   gInterpreter->Declare(R"cpp(
                           void FunctionWrapperFuncVoid(int j) {}
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFuncVoid", "int");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
}

TEST(TClingCallFunc, FunctionWrapperRValueRefArg)
{
   gInterpreter->Declare(R"cpp(
                           void FunctionWrapperFuncRValueRefArg(int&& j) {}
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFuncRValueRefArg", "int&&");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
}

TEST(TClingCallFunc, FunctionWrapperVariadic)
{
   gInterpreter->Declare(R"cpp(
                           void FunctionWrapperFuncVariadic(int j, ...) {}
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFuncVariadic", "int");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   // Make sure we didn't forget the ... in the variadic function signature.
   ASSERT_TRUE(wrapper.find("((void (&)(int, ...))FunctionWrapperFuncVariadic)") != wrapper.npos);
}

TEST(TClingCallFunc, FunctionWrapperDefaultArg)
{
   gInterpreter->Declare(R"cpp(
                           int FunctionWrapperFuncDefaultArg(int j = 0) { return j; }
                           )cpp");

   CallFuncRAII CfRAII("", "FunctionWrapperFuncDefaultArg", "");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));

   // Test that we don't cast the function because otherwise we lose the information
   // about the default call.
   auto FirstCallPos = wrapper.find("FunctionWrapperFuncDefaultArg()");
   ASSERT_TRUE(FirstCallPos != wrapper.npos);
   auto SecondCallPos = wrapper.find("FunctionWrapperFuncDefaultArg()", FirstCallPos + 1);
   ASSERT_TRUE(SecondCallPos != wrapper.npos);
}

TEST(TClingCallFunc, TemplateFunctionWrapper)
{
   gInterpreter->Declare(R"cpp(
                           template<typename T> bool TemplateFunctionWrapperFunc(T b) {return false;}
                           )cpp");

   CallFuncRAII CfRAII("", "TemplateFunctionWrapperFunc", "int");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   // Test that we cast this template function to the right function type.
   ASSERT_TRUE(wrapper.find("((bool (&)(int))TemplateFunctionWrapperFunc<int>)") != wrapper.npos);
}

TEST(TClingCallFunc, FunctionWrapperIncompleteReturnType)
{
   gInterpreter->Declare(R"cpp(
                           class FunctionWrapperFwd;
                           FunctionWrapperFwd* FunctionWrapperIncompleteType() { return nullptr;}
                           )cpp");
   CallFuncRAII CfRAII("", "FunctionWrapperIncompleteType", "");
   std::string wrapper = CfRAII.GetWrapper();

   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
}

TEST(TClingCallFunc, MemberMethodWrapper)
{
   gInterpreter->Declare(R"cpp(
                           struct TClingCallFunc_TestClass1 {
                             bool foo(int) { return false; }
                           };
                           )cpp");

   CallFuncRAII CfRAII("TClingCallFunc_TestClass1", "foo", "int");
   std::string wrapper = CfRAII.GetWrapper();

   // We just test that the wrapper compiles. This is a regression test to make sure
   // we never try to cast a member function as we do above.
   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
}

TEST(TClingCallFunc, DISABLED_OverloadedTemplate)
{
   // FIXME: It's currently possible that we generate a wrapper that is similar to the
   // call we use to initialize k2. In this case we generate a wrapper that doesn't
   // compile and we crash. As it seems to be impossible/difficult to call template
   // operators through the interface we use in the other tests, we just demonstrate
   // the failing declaration of the wrapper here.
   gInterpreter->Declare(R"cpp(
                           namespace OverloadedTemplate {
                             struct X {} x;
                             template<typename T> T operator==(X, bool);
                             template<typename T> bool operator==(X, T);

                             bool k1 = x == true;
                             bool k2 = ((bool(&)(X,bool))::operator==<bool>)(x, true);
                           }
                           )cpp");
}

TEST(TClingCallFunc, FunctionWrapperNodiscard)
{
   gInterpreter->Declare(R"cpp(
                           struct TClingCallFunc_Nodiscard1 {
                           #if __cplusplus >= 201703L
                           [[nodiscard]]
                           #endif
                             bool foo(int) { return false; }
                           };
                           )cpp");

   CallFuncRAII CfRAII("TClingCallFunc_Nodiscard1", "foo", "int");
   std::string wrapper = CfRAII.GetWrapper();

   {
      using ::testing::Not;
      using ::testing::HasSubstr;
      ROOT::TestSupport::FilterDiagsRAII RAII([] (int /*level*/, Bool_t /*abort*/,
                                                    const char * /*location*/, const char *msg) {
         EXPECT_THAT(msg, Not(HasSubstr("-Wunused-result")));
      });
      ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
   }
}

TEST(TClingCallFunc, FunctionWrapperSharedPtr)
{
  gInterpreter->Declare(R"cpp(
                          enum E { A=0, B };
                          void add_sp(std::shared_ptr<std::vector<E>> vals) {
                             vals->push_back(E::A);
                          }
                          )cpp");

  CallFuncRAII CfRAII("", "add_sp", "std::shared_ptr<std::vector<E>>");
   std::string wrapper = CfRAII.GetWrapper();
   ASSERT_TRUE(gInterpreter->Declare(wrapper.c_str()));
}

#include "TClass.h"
#include "TMethod.h"

TEST(TClingCallFunc, ROOT_6523) {
  std::string theString("blaaah");
  std::string searchFor("aaa");
  TClass* stringClass = TClass::GetClass("std::string");
  assert(stringClass != nullptr);
  stringClass->GetListOfMethods();
  TCollection const* overloads = stringClass->GetListOfMethodOverloads("find");
  assert(overloads != nullptr);
  size_t value;
  void const* args[1];
  args[0] = &searchFor;
  for (auto fn : *overloads) {
    TMethod* method = (TMethod *)(fn);
    std::string signature(method->GetSignature());
    if (signature.find("basic_string") == std::string::npos)
      continue;
    method->Dump();
    gInterpreter->ExecuteWithArgsAndReturn(method, &theString, args, 1, &value);
    ASSERT_EQ(value, 2);
    break;
  }
}

TEST(TClingCallFunc, GH_14405) {
  gInterpreter->Declare(R"cpp(
                          float testfunc(int a, int b, float c) {
                            return a + b * c;
                          }
                          )cpp");

  CallFuncRAII CfRAII("", "testfunc", "int, int, float");
  CallFunc_t * cf = CfRAII.GetCF();
  gInterpreter->CallFunc_SetArg(cf, 1);
  gInterpreter->CallFunc_SetArg(cf, 2);
  gInterpreter->CallFunc_SetArg(cf, 3.14f);
  double result = gInterpreter->CallFunc_ExecDouble(cf, /*address=*/0);
  EXPECT_NEAR(result, 7.28, /*abs_error=*/1e-6);

  gInterpreter->CallFunc_ResetArg(cf);

  gInterpreter->CallFunc_SetArg(cf, 1);
  gInterpreter->CallFunc_SetArg(cf, 2);
  gInterpreter->CallFunc_SetArg(cf, 3.14);
  result = gInterpreter->CallFunc_ExecDouble(cf, /*address=*/0);
  EXPECT_NEAR(result, 7.28, /*abs_error=*/1e-6);
}

TEST(TClingCallFunc, GH_14425)
{
   gInterpreter->Declare(R"cpp(
                           struct GH_14425 {
                              int fMember;
                              GH_14425(int m = 1) : fMember(m) {}
                              GH_14425(const GH_14425&) = delete;
                              GH_14425(GH_14425&&) = default;
                           };
                           int GH_14425_f(GH_14425 p = GH_14425()) { return p.fMember; }
                           int GH_14425_g(GH_14425 p) { return p.fMember; }
                           struct GH_14425_Copyable {
                              int fMember;
                              GH_14425_Copyable(int m = 1) : fMember(m) {}
                              GH_14425_Copyable(const GH_14425_Copyable &o) : fMember(o.fMember) {}
                              GH_14425_Copyable(GH_14425_Copyable &&o) : fMember(o.fMember) { o.fMember = 0; }
                           };
                           int GH_14425_h(GH_14425_Copyable p) { return p.fMember; }
                           struct GH_14425_TriviallyCopyable {
                              int fMember;
                              GH_14425_TriviallyCopyable(int m = 1) : fMember(m) {}
                              GH_14425_TriviallyCopyable(const GH_14425_TriviallyCopyable &) = default;
                              GH_14425_TriviallyCopyable(GH_14425_TriviallyCopyable &&o) : fMember(o.fMember) { o.fMember = 0; }
                           };
                           int GH_14425_i(GH_14425_TriviallyCopyable p) { return p.fMember; }
                           struct GH_14425_Default {
                              int fMember;
                              GH_14425_Default(GH_14425 p = GH_14425()) : fMember(p.fMember) {}
                           };
                           struct GH_14425_Required {
                              int fMember;
                              GH_14425_Required(GH_14425 p) : fMember(p.fMember) {}
                           };
                           )cpp");
   CallFuncRAII CfDefaultRAII("", "GH_14425_f", "");
   int valDefault = gInterpreter->CallFunc_ExecInt(CfDefaultRAII.GetCF(), /*address*/ 0);
   EXPECT_EQ(valDefault, 1);

   CallFuncRAII CfArgumentRAII("", "GH_14425_f", "GH_14425");
   CallFunc_t *CfArgument = CfArgumentRAII.GetCF();
   // Cheat a bit: GH_14425 has only one int fMember in memory...
   int objArgument = 2;
   gInterpreter->CallFunc_SetArg(CfArgument, &objArgument);
   int valArgument = gInterpreter->CallFunc_ExecInt(CfArgument, /*address*/ 0);
   EXPECT_EQ(valArgument, 2);

   CallFuncRAII CfRequiredRAII("", "GH_14425_g", "GH_14425");
   CallFunc_t *CfRequired = CfRequiredRAII.GetCF();
   // Cheat a bit: GH_14425 has only one int fMember in memory...
   int objRequired = 3;
   gInterpreter->CallFunc_SetArg(CfRequired, &objRequired);
   int valRequired = gInterpreter->CallFunc_ExecInt(CfRequired, /*address*/ 0);
   EXPECT_EQ(valRequired, 3);

   CallFuncRAII CfCopyableRAII("", "GH_14425_h", "GH_14425_Copyable");
   CallFunc_t *CfCopyable = CfCopyableRAII.GetCF();
   // Cheat a bit: GH_14425_Copyable has only one int fMember in memory...
   int objCopyable = 4;
   gInterpreter->CallFunc_SetArg(CfCopyable, &objCopyable);
   int valCopyable = gInterpreter->CallFunc_ExecInt(CfCopyable, /*address*/ 0);
   EXPECT_EQ(valCopyable, 4);
   // The original value should not have changed; if it did, TClingCallFunc called the move constructor.
   EXPECT_EQ(objCopyable, 4);

   CallFuncRAII CfTriviallyCopyableRAII("", "GH_14425_i", "GH_14425_TriviallyCopyable");
   CallFunc_t *CfTriviallyCopyable = CfTriviallyCopyableRAII.GetCF();
   // Cheat a bit: GH_14425_TriviallyCopyable has only one int fMember in memory...
   int objTriviallyCopyable = 5;
   gInterpreter->CallFunc_SetArg(CfTriviallyCopyable, &objTriviallyCopyable);
   int valTriviallyCopyable = gInterpreter->CallFunc_ExecInt(CfTriviallyCopyable, /*address*/ 0);
   EXPECT_EQ(valTriviallyCopyable, 5);
   // The original value should not have changed; if it did, TClingCallFunc called the move constructor.
   EXPECT_EQ(objTriviallyCopyable, 5);

   CallFuncRAII CfConstructorDefaultRAII("GH_14425_Default", "GH_14425_Default", "");
   int *valConstructorDefault;
   gInterpreter->CallFunc_ExecWithReturn(CfConstructorDefaultRAII.GetCF(), /*address*/ 0, &valConstructorDefault);
   EXPECT_EQ(*valConstructorDefault, 1);

   CallFuncRAII CfConstructorArgumentRAII("GH_14425_Default", "GH_14425_Default", "GH_14425");
   CallFunc_t *CfConstructorArgument = CfConstructorArgumentRAII.GetCF();
   // Cheat a bit: GH_14425 has only one int fMember in memory...
   int objConstructorArgument = 2;
   gInterpreter->CallFunc_SetArg(CfConstructorArgument, &objConstructorArgument);
   int *valConstructorArgument;
   gInterpreter->CallFunc_ExecWithReturn(CfConstructorArgument, /*address*/ 0, &valConstructorArgument);
   EXPECT_EQ(*valConstructorArgument, 2);

   CallFuncRAII CfConstructorRequiredRAII("GH_14425_Required", "GH_14425_Required", "GH_14425");
   CallFunc_t *CfConstructorRequired = CfConstructorRequiredRAII.GetCF();
   // Cheat a bit: GH_14425 has only one int fMember in memory...
   int objConstructorRequired = 3;
   gInterpreter->CallFunc_SetArg(CfConstructorRequired, &objConstructorRequired);
   int *valConstructorRequired;
   gInterpreter->CallFunc_ExecWithReturn(CfConstructorRequired, /*address*/ 0, &valConstructorRequired);
   EXPECT_EQ(*valConstructorRequired, 3);
}

TEST(TClingCallFunc, GH_14425_Virtual)
{
   // Virtual classes are a bit more complicated, we need to declare them both compiled and in the interpreter.
   struct GH_14425_Virtual {
      int fMember;
      GH_14425_Virtual(int m = 1) : fMember(m) {}
      GH_14425_Virtual(const GH_14425_Virtual &) = default;
      GH_14425_Virtual(GH_14425_Virtual &&o) : fMember(o.fMember) { o.fMember = 0; }
      virtual void f() {}
   };
   struct GH_14425_Virtual_User {
      int fMember;
      GH_14425_Virtual_User(int m = 1) : fMember(m) {}
      GH_14425_Virtual_User(const GH_14425_Virtual_User &o) : fMember(o.fMember) {}
      GH_14425_Virtual_User(GH_14425_Virtual_User &&o) : fMember(o.fMember) { o.fMember = 0; }
      virtual void f() {}
   };
   gInterpreter->Declare(R"cpp(
                           struct GH_14425_Virtual {
                              int fMember;
                              GH_14425_Virtual(int m = 1) : fMember(m) {}
                              GH_14425_Virtual(const GH_14425_Virtual &) = default;
                              GH_14425_Virtual(GH_14425_Virtual &&o) : fMember(o.fMember) { o.fMember = 0; }
                              virtual void f() {}
                           };
                           int GH_14425_v(GH_14425_Virtual p) { return p.fMember; }
                           struct GH_14425_Virtual_User {
                              int fMember;
                              GH_14425_Virtual_User(int m = 1) : fMember(m) {}
                              GH_14425_Virtual_User(const GH_14425_Virtual_User &o) : fMember(o.fMember) {}
                              GH_14425_Virtual_User(GH_14425_Virtual_User &&o) : fMember(o.fMember) { o.fMember = 0; }
                              virtual void f() {}
                           };
                           int GH_14425_vu(GH_14425_Virtual_User p) { return p.fMember; }
                           )cpp");
   CallFuncRAII CfVirtualRAII("", "GH_14425_v", "GH_14425_Virtual");
   CallFunc_t *CfVirtual = CfVirtualRAII.GetCF();
   GH_14425_Virtual objVirtual(2);
   gInterpreter->CallFunc_SetArg(CfVirtual, &objVirtual);
   int valVirtual = gInterpreter->CallFunc_ExecInt(CfVirtual, /*address*/ 0);
   EXPECT_EQ(valVirtual, 2);
   // The original value should not have changed; if it did, TClingCallFunc called the move constructor.
   EXPECT_EQ(objVirtual.fMember, 2);

   CallFuncRAII CfVirtualUserRAII("", "GH_14425_vu", "GH_14425_Virtual_User");
   CallFunc_t *CfVirtualUser = CfVirtualUserRAII.GetCF();
   GH_14425_Virtual_User objVirtualUser(3);
   gInterpreter->CallFunc_SetArg(CfVirtualUser, &objVirtualUser);
   int valVirtualUser = gInterpreter->CallFunc_ExecInt(CfVirtualUser, /*address*/ 0);
   EXPECT_EQ(valVirtualUser, 3);
   // The original value should not have changed; if it did, TClingCallFunc called the move constructor.
   EXPECT_EQ(objVirtualUser.fMember, 3);
}

TEST(TClingCallFunc, GH_14425_Templates)
{
   // While according to the C++ standard, GH_14425_Moveable has no move-constructor (which must not be a template),
   // it has a template constructor that can be instantiated to the move-constructor signature. MSVC uses this for their
   // implementation of std::unique_ptr.
   gInterpreter->Declare(R"cpp(
                           struct GH_14425_Moveable {
                              int fMember = 0;
                              GH_14425_Moveable(int m = 1) : fMember(m) {};
                              GH_14425_Moveable(const GH_14425_Moveable&) = delete;
                              template <typename T>
                              GH_14425_Moveable(T &&t) : fMember(t.fMember) {}
                           };
                           int GH_14425_Moveable_f(GH_14425_Moveable p) { return p.fMember; }
                           struct GH_14425_Moveable_Required {
                              int fMember;
                              GH_14425_Moveable_Required(GH_14425_Moveable p) : fMember(p.fMember) {}
                           };
                           template <typename T> struct GH_14425_T {
                              T fMember = 0;
                              GH_14425_T(T m = 1) : fMember(m) {}
                              GH_14425_T(const GH_14425_T&) = delete;
                              GH_14425_T(GH_14425_T &&t) = default;
                           };
                           template <typename T>
                           T GH_14425_t(GH_14425_T<T> p) { return p.fMember; }
                           template <typename T> struct GH_14425_I {
                              std::unique_ptr<T> fMember;
                              GH_14425_I(std::unique_ptr<T> m) : fMember(std::move(m)) {}
                           };
                           template <typename T>
                           T GH_14425_i(GH_14425_I<T> p) { return *p.fMember; }
                           )cpp");
   CallFuncRAII CfMoveableRAII("", "GH_14425_Moveable_f", "GH_14425_Moveable");
   CallFunc_t *CfMoveable = CfMoveableRAII.GetCF();
   // Cheat a bit: GH_14425_Moveable has only one int fMember in memory...
   int objMoveable = 4;
   gInterpreter->CallFunc_SetArg(CfMoveable, &objMoveable);
   int valMoveable = gInterpreter->CallFunc_ExecInt(CfMoveable, /*address*/ 0);
   EXPECT_EQ(valMoveable, 4);

   CallFuncRAII CfConstructorRequiredRAII("GH_14425_Moveable_Required", "GH_14425_Moveable_Required",
                                          "GH_14425_Moveable");
   CallFunc_t *CfConstructorRequired = CfConstructorRequiredRAII.GetCF();
   // Cheat a bit: GH_14425_Moveable has only one int fMember in memory...
   int objConstructorRequired = 5;
   gInterpreter->CallFunc_SetArg(CfConstructorRequired, &objConstructorRequired);
   int *valConstructor;
   gInterpreter->CallFunc_ExecWithReturn(CfConstructorRequired, /*address*/ 0, &valConstructor);
   EXPECT_EQ(*valConstructor, 5);

   CallFuncRAII CfTRAII("", "GH_14425_t<int>", "GH_14425_T<int>");
   CallFunc_t *CfT = CfTRAII.GetCF();
   // Cheat a bit: GH_14425_T<int> has only one int fMember in memory...
   int objT = 6;
   gInterpreter->CallFunc_SetArg(CfT, &objT);
   int valT = gInterpreter->CallFunc_ExecInt(CfT, /*address*/ 0);
   EXPECT_EQ(valT, 6);

   CallFuncRAII CfIRAII("", "GH_14425_i<int>", "GH_14425_I<int>");
   CallFunc_t *CfI = CfIRAII.GetCF();
   // Cheat a bit: GH_14425_I<int> has only one std::unique_ptr<int> fMember in memory...
   auto objI = std::make_unique<int>(7);
   gInterpreter->CallFunc_SetArg(CfI, &objI);
   int valI = gInterpreter->CallFunc_ExecInt(CfI, /*address*/ 0);
   EXPECT_EQ(valI, 7);
}
