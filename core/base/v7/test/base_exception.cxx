#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <ROOT/RError.hxx>

#include <stdexcept>

using RException = ROOT::Experimental::RException;

namespace {

static ROOT::Experimental::RResult<void> TestFailure()
{
   R__FAIL("test failure");
}

static ROOT::Experimental::RResult<void> TestSuccess()
{
   R__SUCCESS
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
   R__FORWARD_RESULT(rv);
}

class ExceptionX : public std::runtime_error {
public:
   explicit ExceptionX(const std::string &what) : std::runtime_error(what) {}
};

} // anonymous namespace


TEST(Exception, Report)
{
   bool exceptionThrown = false;
   try {
      TestChain(false);
   } catch (const RException& e) {
      exceptionThrown = true;
      ASSERT_EQ(2U, e.GetError().GetStackTrace().size());
      EXPECT_THAT(e.GetError().GetStackTrace().at(0).fFunction, ::testing::HasSubstr("TestSyscall(bool)"));
      EXPECT_THAT(e.GetError().GetStackTrace().at(1).fFunction, ::testing::HasSubstr("TestChain(bool)"));
   }
   EXPECT_TRUE(exceptionThrown);
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
