#include "gtest/gtest.h"

#include <ROOT/RError.hxx>

#include <stdexcept>

using RException = ROOT::Experimental::RException;
using RStatusBool = ROOT::Experimental::RStatusBool;
using RStatusSyscall = ROOT::Experimental::RStatusSyscall;

namespace {

static RStatusBool TestFailure()
{
   return RStatusBool(false);
}

static RStatusBool TestSuccess()
{
   return RStatusBool(true);
}

static RStatusSyscall MockFileOpen(bool succeed)
{
   if (succeed)
      return RStatusSyscall(42);
   return RStatusSyscall::Fail("Not allowed to succeed");
}

class ExceptionX : public std::runtime_error {
public:
   explicit ExceptionX(const std::string &what) : std::runtime_error(what) {}
};

} // anonymous namespace


TEST(Exception, InstantExceptions)
{
   ROOT::Experimental::SetThrowInstantExceptions(true); // the default
   bool passedFailure = false;
   bool exceptionThrown = false;

   try {
      TestFailure();
      passedFailure = true;
   } catch (const RException&) {
      exceptionThrown = true;
   }
   EXPECT_FALSE(passedFailure);
   EXPECT_TRUE(exceptionThrown);
}


TEST(Exception, DiscardReturnValue)
{
   ROOT::Experimental::SetThrowInstantExceptions(false);
   bool exceptionThrown;

   try {
      exceptionThrown = false;
      TestFailure();
   } catch (const RException&) {
      exceptionThrown = true;
   }
   EXPECT_TRUE(exceptionThrown);

   try {
      exceptionThrown = false;
      TestSuccess();
   } catch (const RException&) {
      exceptionThrown = true;
   }
   EXPECT_FALSE(exceptionThrown);
}


TEST(Exception, CheckReturnValue)
{
   ROOT::Experimental::SetThrowInstantExceptions(false);
   auto rv = TestFailure();
   EXPECT_TRUE(rv.IsError());
   // No exception / crash when the scope closes
}


TEST(Exception, DoubleThrow)
{
   ROOT::Experimental::SetThrowInstantExceptions(false);
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
   ROOT::Experimental::SetThrowInstantExceptions(true);
   auto fd = MockFileOpen(true);
   ASSERT_TRUE(fd.IsValid());
   EXPECT_EQ(42, fd);

   EXPECT_THROW(MockFileOpen(false), RException);
}
