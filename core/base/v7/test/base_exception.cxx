#include "gtest/gtest.h"

#include <ROOT/RError.hxx>

#include <cerrno>
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
   errno = EPERM;
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

   try {
      TestFailure();
      passedFailure = true;
   } catch (const RException&) {
   }
   EXPECT_FALSE(passedFailure);
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
      throw ExceptionX("something else went wrong");
   } catch (const ExceptionX&) {
      // This will not catch RException, so that the program crashes if rv throws
   }
}


TEST(Exception, Syscall)
{
   ROOT::Experimental::SetThrowInstantExceptions(true);
   auto fd = MockFileOpen(true);
   EXPECT_EQ(42, fd);

   EXPECT_THROW(MockFileOpen(false), RException);
}
