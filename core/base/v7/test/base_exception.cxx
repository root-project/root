#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <ROOT/RError.hxx>

#include <stdexcept>

using RException = ROOT::Experimental::RException;

namespace {

/// Used to verify that wrapped return values are not unnecessarily copied
struct ComplexReturnType {
   static int gNCopies;
   ComplexReturnType() { gNCopies++; }
   ComplexReturnType(const ComplexReturnType &) { gNCopies++; }
   ComplexReturnType(ComplexReturnType &&other) = default;
   ComplexReturnType &operator= (const ComplexReturnType &) { return *this; }
   ComplexReturnType &operator= (ComplexReturnType &&other) = default;
};
int ComplexReturnType::gNCopies = 0;

static ROOT::Experimental::RResult<void> TestFailure()
{
   R__FAIL("test failure");
}

static ROOT::Experimental::RResult<void> TestSuccess()
{
   return ROOT::Experimental::RResult<void>::Success();
}

static ROOT::Experimental::RResult<int> TestSyscall(bool succeed)
{
   if (succeed)
      return 42;
   R__FAIL("failure");
}

static ROOT::Experimental::RResult<int> TestChain(bool succeed)
{
   auto rv = TestSyscall(succeed);
   return R__FORWARD_RESULT(rv);
}

static ROOT::Experimental::RResult<ComplexReturnType> TestComplex()
{
   return ComplexReturnType();
}

class ExceptionX : public std::runtime_error {
public:
   explicit ExceptionX(const std::string &what) : std::runtime_error(what) {}
};

} // anonymous namespace


TEST(Exception, Report)
{
   try {
      TestChain(false);
      EXPECT_TRUE(false) << "Above line should have thrown!";
   } catch (const RException& e) {
      ASSERT_EQ(2U, e.GetError().GetStackTrace().size());
      EXPECT_THAT(e.GetError().GetStackTrace().at(0).fFunction, ::testing::HasSubstr("TestSyscall(bool)"));
      EXPECT_THAT(e.GetError().GetStackTrace().at(1).fFunction, ::testing::HasSubstr("TestChain(bool)"));
   }
}


TEST(Exception, ForwardResult)
{
   auto res = TestChain(true);
   ASSERT_TRUE(static_cast<bool>(res));
   EXPECT_EQ(42, res.Get());
}


TEST(Exception, DiscardReturnValue)
{
   EXPECT_THROW(TestFailure(), RException);
   EXPECT_NO_THROW(TestSuccess());
}


TEST(Exception, CheckReturnValue)
{
   auto rv = TestFailure();
   EXPECT_FALSE(rv);
   // No exception / crash when the scope closes
}


TEST(Exception, DoubleThrow)
{
   try {
      auto rv = TestFailure();
      // Throwing ExceptionX will destruct rv along the way. Since rv carries an error state, it would normally
      // throw an exception itself. In this test, we verify that rv surpresses throwing an exception if another
      // exception is currently active.
      throw ExceptionX("something else went wrong");
   } catch (const ExceptionX&) {
      // This will only catch ExceptionX but not RException. In case rv mistakenly throws an exception,
      // we would notice the test failure by a crash of the unit test.
   }
}


TEST(Exception, Syscall)
{
   auto fd = TestSyscall(true);
   if (!fd) {
      // In production code, we would expect error handling code other than throw
      EXPECT_THROW(fd.Throw(), RException);
   }
   EXPECT_EQ(42, fd.Get());

   EXPECT_THROW(TestSyscall(false).Get(), RException);
}

TEST(Exception, ComplexReturnType)
{
   auto res = TestComplex();
   EXPECT_EQ(1, ComplexReturnType::gNCopies);
}
