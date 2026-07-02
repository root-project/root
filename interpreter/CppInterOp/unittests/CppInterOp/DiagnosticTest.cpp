//===- DiagnosticTest.cpp - Tests for Cpp::DiagnosticRef -------*- C++ -*-===//
//
// Part of the compiler-research project, under the Apache License v2.0 with
// LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CppInterOp/CppInterOp.h"
#include "CppInterOp/Error.h"

#include "../../lib/CppInterOp/ErrorInternal.h"

#include "Utils.h"

#include "gtest/gtest.h"

#include <cstring>

// Size invariants. Refactors that inflate the public handle or the
// severity enum fail the build.
static_assert(sizeof(Cpp::DiagnosticRef) == sizeof(void*),
              "DiagnosticRef must be a single pointer");
static_assert(sizeof(Cpp::DiagnosticSeverity) == 1,
              "DiagnosticSeverity must stay one byte");

TEST(DiagnosticTest, AccessorsRoundTripStoredView) {
  Cpp::StoredDiagView V;
  V.Message = "expected ';' after expression";
  V.File = "/tmp/foo.cpp";
  V.Line = 12;
  V.Column = 5;
  V.Sev = Cpp::DiagnosticSeverity::Error;

  Cpp::DiagnosticRef D{&V};
  EXPECT_FALSE(D.isNull());
  EXPECT_EQ(Cpp::GetDiagnosticSeverity(D), Cpp::DiagnosticSeverity::Error);
  EXPECT_STREQ(Cpp::GetDiagnosticMessage(D), "expected ';' after expression");
  EXPECT_STREQ(Cpp::GetDiagnosticFile(D), "/tmp/foo.cpp");
  EXPECT_EQ(Cpp::GetDiagnosticLine(D), 12U);
  EXPECT_EQ(Cpp::GetDiagnosticColumn(D), 5U);
}

TEST(DiagnosticTest, SeverityRoundTrip) {
  const Cpp::DiagnosticSeverity AllSev[] = {
      Cpp::DiagnosticSeverity::Note,
      Cpp::DiagnosticSeverity::Warning,
      Cpp::DiagnosticSeverity::Error,
      Cpp::DiagnosticSeverity::Fatal,
  };
  for (Cpp::DiagnosticSeverity S : AllSev) {
    Cpp::StoredDiagView V;
    V.Sev = S;
    Cpp::DiagnosticRef D{&V};
    EXPECT_EQ(Cpp::GetDiagnosticSeverity(D), S);
  }
}

TEST(DiagnosticTest, NullRefReturnsDefaults) {
  Cpp::DiagnosticRef D;
  EXPECT_TRUE(D.isNull());
  EXPECT_EQ(Cpp::GetDiagnosticSeverity(D), Cpp::DiagnosticSeverity::Note);
  EXPECT_STREQ(Cpp::GetDiagnosticMessage(D), "");
  EXPECT_STREQ(Cpp::GetDiagnosticFile(D), "");
  EXPECT_EQ(Cpp::GetDiagnosticLine(D), 0U);
  EXPECT_EQ(Cpp::GetDiagnosticColumn(D), 0U);
}

// Empty strings must surface as "" not nullptr so callers can pass
// the result to strlen / strcmp without a separate null guard.
TEST(DiagnosticTest, EmptyStringsReturnEmptyCStrings) {
  Cpp::StoredDiagView V;
  Cpp::DiagnosticRef D{&V};
  ASSERT_NE(Cpp::GetDiagnosticMessage(D), nullptr);
  ASSERT_NE(Cpp::GetDiagnosticFile(D), nullptr);
  EXPECT_STREQ(Cpp::GetDiagnosticMessage(D), "");
  EXPECT_STREQ(Cpp::GetDiagnosticFile(D), "");
}

TEST(DiagnosticTest, TwoIndependentDiagsKeepTheirFields) {
  Cpp::StoredDiagView A;
  A.Message = "first";
  A.Line = 1;
  Cpp::StoredDiagView B;
  B.Message = "second";
  B.Line = 99;

  Cpp::DiagnosticRef DA{&A};
  Cpp::DiagnosticRef DB{&B};
  EXPECT_STREQ(Cpp::GetDiagnosticMessage(DA), "first");
  EXPECT_EQ(Cpp::GetDiagnosticLine(DA), 1U);
  EXPECT_STREQ(Cpp::GetDiagnosticMessage(DB), "second");
  EXPECT_EQ(Cpp::GetDiagnosticLine(DB), 99U);
}

// Out-of-range index must yield a null DiagnosticRef rather than
// indexing past the end of the deque.
TYPED_TEST(CPPINTEROP_TEST_MODE, GetPendingDiagnostic_OutOfRangeIsNull) {
  TestFixture::CreateInterpreter();
  Cpp::ClearPendingDiagnostics();
  EXPECT_TRUE(Cpp::GetPendingDiagnostic(0).isNull());
  EXPECT_TRUE(Cpp::GetPendingDiagnostic(999).isNull());
}

// End-to-end: the diagnostic consumer wired into every interpreter
// must capture a real Clang parse error and surface it through the
// pending-diagnostic accessors.
TYPED_TEST(CPPINTEROP_TEST_MODE, Consumer_CapturesParseError) {
#ifdef _WIN32
  // The chained TextDiagnosticPrinter emits "error: expected expression"
  // to stderr, which MSBuild's check-cppinterop custom-build rule scans
  // and treats as a build failure regardless of the gtest result.
  GTEST_SKIP() << "chained diagnostic trips MSBuild error scanning on Windows";
#endif
  TestFixture::CreateInterpreter();
  Cpp::ClearPendingDiagnostics();
  ASSERT_EQ(Cpp::GetPendingDiagnosticCount(), 0U);

  // Trigger a parse error. silent=false lets the diagnostic reach the
  // consumer; silent=true sets SuppressAllDiagnostics, which short-
  // circuits before the consumer chain.
  EXPECT_NE(0, Cpp::Declare("int err = ;", /*silent=*/false));

  ASSERT_GT(Cpp::GetPendingDiagnosticCount(), 0U);
  Cpp::DiagnosticRef D = Cpp::GetPendingDiagnostic(0);
  EXPECT_FALSE(D.isNull());
  EXPECT_EQ(Cpp::GetDiagnosticSeverity(D), Cpp::DiagnosticSeverity::Error);
  EXPECT_GT(std::strlen(Cpp::GetDiagnosticMessage(D)), 0U);

  Cpp::ClearPendingDiagnostics();
  EXPECT_EQ(Cpp::GetPendingDiagnosticCount(), 0U);
}

// Successful Declare must not leave stale diagnostics behind.
TYPED_TEST(CPPINTEROP_TEST_MODE, Consumer_OkDeclareLeavesBufferEmpty) {
  TestFixture::CreateInterpreter();
  Cpp::ClearPendingDiagnostics();
  EXPECT_EQ(0, Cpp::Declare("int ok = 7;", /*silent=*/false));
  // Warnings might accrue, but no error-level diagnostics should.
  for (unsigned I = 0, N = Cpp::GetPendingDiagnosticCount(); I < N; ++I) {
    Cpp::DiagnosticRef D = Cpp::GetPendingDiagnostic(I);
    EXPECT_NE(Cpp::GetDiagnosticSeverity(D), Cpp::DiagnosticSeverity::Error);
    EXPECT_NE(Cpp::GetDiagnosticSeverity(D), Cpp::DiagnosticSeverity::Fatal);
  }
  Cpp::ClearPendingDiagnostics();
}

// silent=true sets SuppressAllDiagnostics on the engine, which
// short-circuits before any consumer in the chain runs. Pin this
// contract: a failed declare under silent=true reports failure via
// the return code but the consumer captures nothing.
TYPED_TEST(CPPINTEROP_TEST_MODE, Consumer_SilentSuppressesCapture) {
  TestFixture::CreateInterpreter();
  Cpp::ClearPendingDiagnostics();
  EXPECT_NE(0, Cpp::Declare("int err = ;", /*silent=*/true));
  EXPECT_EQ(Cpp::GetPendingDiagnosticCount(), 0U);
}
