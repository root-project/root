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
   char **argv = new char *[argc];
   argv[0] = const_cast<char *>("-q");
   argv[1] = const_cast<char *>("-z");
   argv[2] = const_cast<char *>("--nonexistingoption");
   argv[3] = const_cast<char *>("-b");

   CaptureStderr();
   // Unrecognized options will be printed to stderr
   TRint app{"App", &argc, argv};
   std::string trinterr = GetCapturedStderr();

   const std::string expected{"root: unrecognized option '-z'\n"
                              "root: unrecognized option '--nonexistingoption'\n"
                              "Try 'root --help' for more information.\n"};

   EXPECT_EQ(trinterr, expected);

   // Properly delete the array
   for (int i = 0; i < argc; i++) {
      delete[] argv[i];
   }
   delete[] argv;
   argv = nullptr;
}
