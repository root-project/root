#include "ROOT/TestSupport.hxx"
#include "TInterpreter.h"

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
