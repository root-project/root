#ifndef ROOT7_RNTuple_Test_Fork
#define ROOT7_RNTuple_Test_Fork

#include <functional>
#include <string_view>
#include <cstdint>
#include <sstream>

#include <unistd.h>
#include <sys/wait.h>

#include "gtest/gtest.h"

#include <TInterpreter.h>

// These helpers are split into an *Impl function and a macro to check for and propagate fatal failures to the caller.

inline void ExecInForkImpl(std::function<void(void)> fn)
{
   pid_t pid = fork();
   if (pid == -1) {
      FAIL() << "fork() failed";
   }

   if (pid == 0) {
      fn();
      if (::testing::Test::HasFatalFailure())
         exit(EXIT_FAILURE);
      exit(EXIT_SUCCESS);
   }

   int wstatus;
   if (waitpid(pid, &wstatus, 0) == -1) {
      FAIL() << "waitpid() failed";
   } else if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
      FAIL() << "child did not exit successfully";
   }
}

inline void DeclarePointerImpl(std::string_view type, std::string_view variable, void *ptr)
{
   auto ptrString = std::to_string(reinterpret_cast<std::uintptr_t>(ptr));
   std::ostringstream ptrDeclare;
   ptrDeclare << type << " *" << variable << " = ";
   ptrDeclare << "reinterpret_cast<" << type << " *>(" << ptrString << ");";
   ASSERT_TRUE(gInterpreter->Declare(ptrDeclare.str().c_str()));
}

inline void ProcessLineImpl(const char *line)
{
   TInterpreter::EErrorCode error = TInterpreter::kNoError;
   gInterpreter->ProcessLine(line, &error);
   ASSERT_EQ(error, TInterpreter::kNoError);
}

#define ExecInFork(old)                       \
   do {                                       \
      ExecInForkImpl(old);                    \
      if (::testing::Test::HasFatalFailure()) \
         return;                              \
   } while (0)

#define DeclarePointer(type, variable, ptr)    \
   do {                                        \
      DeclarePointerImpl(type, variable, ptr); \
      if (::testing::Test::HasFatalFailure())  \
         return;                               \
   } while (0)

#define ProcessLine(line)                     \
   do {                                       \
      ProcessLineImpl(line);                  \
      if (::testing::Test::HasFatalFailure()) \
         return;                              \
   } while (0)

#endif // ROOT7_RNTuple_Test_Fork
