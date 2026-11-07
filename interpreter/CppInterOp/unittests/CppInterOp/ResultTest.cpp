//===- ResultTest.cpp - Tests for Cpp::Result<T> ---------------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Value-semantic tests for Cpp::Result<T>: construction, accessors,
// copy/move (including the Unchecked-flag transfer that catches the
// moved-from-aborts-in-dtor trap), .ignore() opt-out, and death paths
// for dropped errors and value()-on-error.
//
//===----------------------------------------------------------------------===//

#include "CppInterOp/CppInterOp.h"
#include "CppInterOp/CppInterOpTypes.h"
#include "CppInterOp/Error.h"

#include "../../lib/CppInterOp/ErrorInternal.h"
#include "../../lib/CppInterOp/InterpreterInfo.h"
#include "../../lib/CppInterOp/Tracing.h" // INTEROP_FUNC_SIG

#include "Utils.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>
#include <utility>

// Size invariants. Each row caps Cpp::Result<T> at llvm::Expected<T>'s
// size; a refactor that inflates the layout fails the build.

static_assert(sizeof(Cpp::ErrorRef) == sizeof(void*),
              "ErrorRef must be a single pointer");

// Result<void> in release is just the ErrorRef pointer. Debug adds the
// Unchecked bit; allow up to llvm::Error's debug-fudge ceiling.
static_assert(sizeof(Cpp::Result<void>) <= sizeof(llvm::Error) * 2,
              "Result<void> should not exceed llvm::Error layout");

static_assert(sizeof(Cpp::Result<Cpp::DeclRef>) <=
                  sizeof(llvm::Expected<void*>),
              "Result<DeclRef> must not exceed llvm::Expected<void*>");

static_assert(sizeof(Cpp::Result<int>) <= sizeof(llvm::Expected<int>),
              "Result<int> must not exceed llvm::Expected<int>");

namespace {
// Inline-encoded Status::NotFound: round-trips through the tag bit,
// allocates nothing, costs no refcount work.
Cpp::ErrorRef MakeErr() {
  return Cpp::ErrorRef::makeInline(Cpp::Status::NotFound);
}
} // namespace

TEST(ResultTest, Result_OkFromValue) {
  Cpp::Result<int> R{42};
  EXPECT_TRUE(R.ok());
  EXPECT_TRUE(static_cast<bool>(R));
  EXPECT_EQ(R.status(), Cpp::Status::Ok);
  EXPECT_TRUE(R.error().isOk());
  EXPECT_EQ(R.value(), 42);
  EXPECT_EQ(R.value_or(-1), 42);
}

TEST(ResultTest, Result_DefaultIsOk) {
  // Default-constructed Result<T> holds a default-constructed T; ok().
  Cpp::Result<int> R;
  EXPECT_TRUE(R.ok());
  EXPECT_EQ(R.value(), 0);
}

TEST(ResultTest, Result_ErrorFromErrorRef) {
  Cpp::Result<int> R{MakeErr()};
  EXPECT_FALSE(R.ok());
  EXPECT_FALSE(static_cast<bool>(R));
  EXPECT_EQ(R.status(), Cpp::Status::NotFound);
  EXPECT_FALSE(R.error().isOk());
  EXPECT_TRUE(R.error().isInline());
  EXPECT_FALSE(R.error().isSlice());
  EXPECT_EQ(R.value_or(-1), -1);
}

TEST(ResultTest, ResultVoid_DefaultIsOk) {
  Cpp::Result<void> R;
  EXPECT_TRUE(R.ok());
  EXPECT_EQ(R.status(), Cpp::Status::Ok);
  EXPECT_TRUE(R.error().isOk());
}

TEST(ResultTest, ResultVoid_FromErrorRef) {
  Cpp::Result<void> R{MakeErr()};
  EXPECT_FALSE(R.ok());
  EXPECT_EQ(R.status(), Cpp::Status::NotFound);
  EXPECT_TRUE(R.error().isInline());
}

TEST(ResultTest, Copy_Ok) {
  Cpp::Result<int> A{7};
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Cpp::Result<int> B{A}; // exercises the copy ctor; do not collapse to a ref
  EXPECT_TRUE(B.ok());
  EXPECT_EQ(B.value(), 7);
  // The source remains usable after copy; we explicitly check it here
  // so its dtor doesn't trip the must-handle guard.
  EXPECT_EQ(A.value(), 7);
}

TEST(ResultTest, Copy_Error) {
  Cpp::Result<int> A{MakeErr()};
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  Cpp::Result<int> B{A}; // exercises the copy ctor; do not collapse to a ref
  EXPECT_FALSE(B.ok());
  EXPECT_FALSE(A.ok()); // mark A checked so its dtor doesn't abort
}

TEST(ResultTest, Move_Ok) {
  Cpp::Result<int> A{11};
  Cpp::Result<int> B{std::move(A)};
  EXPECT_TRUE(B.ok());
  EXPECT_EQ(B.value(), 11);
  // A was Ok-valued; the move-from state is undefined for the value
  // but still ok-tagged. No abort on A's dtor.
}

// Pins the bug where Result<void>'s default move ctor failed to
// transfer the Unchecked obligation, causing the source's dtor to
// abort even though the value semantically went to the destination
// (most visible when threaded through a tracing record() wrapper).
TEST(ResultTest, MoveCtor_Void_TransfersUncheckedToSource) {
  Cpp::Result<void> A{MakeErr()};
  Cpp::Result<void> B{std::move(A)};
  // Check only B; A must not abort at scope exit despite never being
  // inspected directly.
  EXPECT_FALSE(B.ok());
}

TEST(ResultTest, MoveCtor_NonVoid_TransfersUncheckedToSource) {
  Cpp::Result<int> A{MakeErr()};
  Cpp::Result<int> B{std::move(A)};
  EXPECT_FALSE(B.ok());
}

// Non-trivial T exercises the placement-new + explicit destructor
// paths in ctor/dtor. Result<int> alone wouldn't catch a missing
// T::~T() call -- int has no destructor to miss. ASan / leak checker
// flags any miss here.
TEST(ResultTest, Result_NonTrivialT_PlacementNewAndDtor) {
  {
    Cpp::Result<std::string> R{std::string("hello")};
    EXPECT_TRUE(R.ok());
    EXPECT_EQ(R.value(), "hello");
  }

  Cpp::Result<std::string> A{std::string("world")};
  Cpp::Result<std::string> B{std::move(A)};
  EXPECT_TRUE(B.ok());
  EXPECT_EQ(B.value(), "world");
}

// .ignore() silences the dropped-error abort. Without it, the inner
// blocks would die at scope exit.
TEST(ResultTest, Ignore_SuppressesDroppedErrorAbort) {
  {
    Cpp::Result<int> R{MakeErr()};
    R.ignore();
  }
  {
    Cpp::Result<void> R{MakeErr()};
    R.ignore();
  }
  SUCCEED();
}

#ifndef EMSCRIPTEN

#ifndef NDEBUG
TEST(ResultDeathTest, DroppedError_Aborts) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH({ (void)Cpp::Result<int>(MakeErr()); },
               ".*destroyed without check.*");
}
#endif // !NDEBUG

// value() on an error always aborts, debug or release.
TEST(ResultDeathTest, ValueOnError_Aborts) {
  ::testing::FLAGS_gtest_death_test_style = "threadsafe";
  EXPECT_DEATH({ (void)Cpp::Result<int>(MakeErr()).value(); },
               ".*value\\(\\) called on an error-bearing Result.*");
}

#endif // !EMSCRIPTEN

//
// Every Status value round-trips through makeInline -> ErrorRef bits ->
// status(). Status::Ok canonicalises to the null encoding; the rest
// pack into a single uintptr_t with no allocation.

TEST(ResultTest, InlineEncoding_StatusRoundTrip) {
  const Cpp::Status All[] = {
      Cpp::Status::NotFound,     Cpp::Status::InvalidArgument,
      Cpp::Status::Ambiguous,    Cpp::Status::ParseError,
      Cpp::Status::CompileError,
  };
  for (Cpp::Status S : All) {
    Cpp::ErrorRef E = Cpp::ErrorRef::makeInline(S);
    EXPECT_FALSE(E.isOk());
    EXPECT_TRUE(E.isInline());
    EXPECT_FALSE(E.isSlice());
    EXPECT_EQ(E.status(), S);
    // Inline errors carry no body.
    EXPECT_EQ(E.diagnostics().size(), 0U);
    EXPECT_EQ(E.producer(), nullptr);
  }
}

TEST(ResultTest, InlineEncoding_OkCanonicalisesToNull) {
  Cpp::ErrorRef E = Cpp::ErrorRef::makeInline(Cpp::Status::Ok);
  EXPECT_TRUE(E.isOk());
  EXPECT_FALSE(E.isInline());
  EXPECT_FALSE(E.isSlice());
  EXPECT_EQ(E.bits, 0U);
}

namespace {
// makeError / drainError need a live interpreter to allocate slices
// against.
struct ResultBoundaryTest : public ::testing::Test {
  void SetUp() override {
    if (!Cpp::GetInterpreter())
      Cpp::CreateInterpreter();
  }
};
} // namespace

TEST_F(ResultBoundaryTest, MakeError_SliceCarriesMessageAndProducer) {
  Cpp::InterpreterInfo* II = Cpp::GetInterpInfo();
  Cpp::ClearPending(II);
  Cpp::Result<void> R{Cpp::makeError(
      II, Cpp::Status::ParseError, "syntax error", __func__, INTEROP_FUNC_SIG)};
  ASSERT_FALSE(R.ok());
  EXPECT_EQ(R.status(), Cpp::Status::ParseError);
  EXPECT_TRUE(R.error().isSlice());
  EXPECT_FALSE(R.error().isInline());
  ASSERT_GE(R.error().diagnostics().size(), 1U);
  EXPECT_STREQ(R.error().diagnostics()[0].message(), "syntax error");
  EXPECT_EQ(R.error().diagnostics()[0].severity(),
            Cpp::DiagnosticSeverity::Error);
  EXPECT_STREQ(R.error().producer(), __func__);
}

// drainError lifts an llvm::Error chain into a slice, preserving Status
// and any structured payload the chain carries.
TEST_F(ResultBoundaryTest, DrainError_StatusErrorFixesCode) {
  Cpp::InterpreterInfo* II = Cpp::GetInterpInfo();
  Cpp::ClearPending(II);
  llvm::Error E = llvm::make_error<Cpp::StatusError>(Cpp::Status::NotFound,
                                                     "no such symbol");
  Cpp::Result<void> R{Cpp::drainError(II, std::move(E), __func__, nullptr)};
  ASSERT_FALSE(R.ok());
  EXPECT_EQ(R.status(), Cpp::Status::NotFound);
}

// share() bumps the slice refcount so the captured error survives the
// originating Result's destruction.
TEST_F(ResultBoundaryTest, CapturedError_OutlivesResult) {
  Cpp::InterpreterInfo* II = Cpp::GetInterpInfo();
  Cpp::ClearPending(II);
  Cpp::CapturedError C;
  size_t DiagCount = 0;
  {
    Cpp::Result<void> R{Cpp::makeError(
        II, Cpp::Status::ParseError, "captured-error test", __func__, nullptr)};
    ASSERT_FALSE(R.ok());
    DiagCount = R.error().diagnostics().size();
    C = R.share();
  } // R out of scope; slice survives because C holds a ref.
  EXPECT_FALSE(C.ok());
  EXPECT_EQ(C.status(), Cpp::Status::ParseError);
  EXPECT_EQ(C.diagnostics().size(), DiagCount);
}

// Inline errors have no slice; capture is a pure value copy with no
// refcount bookkeeping.
TEST(ResultTest, CapturedError_InlineHasNoSlice) {
  Cpp::Result<int> R{Cpp::ErrorRef::makeInline(Cpp::Status::InvalidArgument)};
  Cpp::CapturedError C = R.share();
  EXPECT_FALSE(C.ok());
  EXPECT_EQ(C.status(), Cpp::Status::InvalidArgument);
  EXPECT_TRUE(C.ref().isInline());
  EXPECT_FALSE(C.ref().isSlice());
}

// record() copies into an owning value-type so the caller can drop the
// Result and the slice can be freed.
TEST_F(ResultBoundaryTest, ErrorRecord_OwnsDiagnostics) {
  Cpp::ErrorRecord Rec;
  {
    Cpp::InterpreterInfo* II = Cpp::GetInterpInfo();
    Cpp::ClearPending(II);
    Cpp::Result<void> R{Cpp::makeError(
        II, Cpp::Status::ParseError, "recordable message", __func__, nullptr)};
    ASSERT_FALSE(R.ok());
    Rec = R.error().record();
  } // R dies; slice freed. Rec stays usable.
  EXPECT_EQ(Rec.Code, Cpp::Status::ParseError);
  ASSERT_GE(Rec.Diagnostics.size(), 1U);
  EXPECT_EQ(Rec.Diagnostics[0].Message, "recordable message");
}

TEST(ResultTest, ErrorRecord_InlineProducesEmpty) {
  Cpp::ErrorRef E = Cpp::ErrorRef::makeInline(Cpp::Status::InvalidArgument);
  Cpp::ErrorRecord Rec = E.record();
  EXPECT_EQ(Rec.Code, Cpp::Status::InvalidArgument);
  EXPECT_TRUE(Rec.Diagnostics.empty());
}

static_assert(sizeof(Cpp::ErrorRef) == sizeof(void*),
              "ErrorRef stays one pointer-word");
static_assert(sizeof(Cpp::Status) == 1, "Status enum stays one byte");
