#include <TPython.h>

#include "gtest/gtest.h"

// CTest sets PYTHONHOME to a nonexistent installation, so initialization must fail.
TEST(TPython, InitializationFailure)
{
   EXPECT_FALSE(TPython::Exec("pass"));
   // Retrying must also fail cleanly after a partially initialized interpreter.
   EXPECT_FALSE(TPython::Exec("pass"));
}
