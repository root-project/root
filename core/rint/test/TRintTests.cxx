#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "TRint.h"

using testing::internal::CaptureStderr;
using testing::internal::GetCapturedStderr;

TEST(TRint, UnrecognizedOptions)
{
   // Create array of options.
   // We need to create it as a dynamic array for the following reasons:
   // - TRint constructor accepts a char** so we construct directly that type
   // - TRint will modify this array, removing recognized options and leaving
   //   only unrecognized ones, so we can't create an std::vector and pass its
   //   data to TRint directly.
   int argc{4};
   char e1[]{"-q"};
   char e2[]{"-z"};
   char e3[]{"--nonexistingoption"};
   char e4[]{"-b"};
   char *argv[]{e1, e2, e3, e4};

   CaptureStderr();
   // Unrecognized options will be printed to stderr
   TRint app{"App", &argc, argv};
   std::string trinterr = GetCapturedStderr();

   const std::string expected{"root: unrecognized option '-z'\n"
                              "root: unrecognized option '--nonexistingoption'\n"
                              "Try 'root --help' for more information.\n"};

   EXPECT_EQ(trinterr, expected);
}
