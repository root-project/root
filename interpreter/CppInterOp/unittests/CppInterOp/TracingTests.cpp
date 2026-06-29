#include "../../lib/CppInterOp/Tracing.h"

#include "clang/Basic/Version.h"

#include "CppInterOp/CppInterOp.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <regex>
#include <sstream>

using namespace CppInterOp::Tracing;
using ::testing::HasSubstr;
using ::testing::Not;

// Helper: join the full trace log into a single string for matching.
static std::string getFullLog() {
  std::string result;
  for (const auto& entry : TheTraceInfo->getLog())
    result += entry + "\n";
  return result;
}

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

class TracingTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    saved_death_test_style_ = ::testing::GTEST_FLAG(death_test_style);
    ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  }
  static void TearDownTestSuite() {
    ::testing::GTEST_FLAG(death_test_style) = saved_death_test_style_;
  }
  void SetUp() override {
    if (TheTraceInfo)
      TheTraceInfo->clear();
    TheTraceInfo = nullptr;
    InitTracing();
  }

private:
  static std::string saved_death_test_style_;
};

std::string TracingTest::saved_death_test_style_;

class NoTracingTest : public ::testing::Test {
protected:
  void SetUp() override { TheTraceInfo = nullptr; }
};

// ---------------------------------------------------------------------------
// Helpers — small functions that use the macros
// ---------------------------------------------------------------------------

int UnannotatedFunction(int x) {
  INTEROP_TRACE(x);
  return x * 2; // Missing INTEROP_RETURN!
}

int AnnotatedFunction(int x) {
  INTEROP_TRACE(x);
  return INTEROP_RETURN(x * 2);
}

void VoidFunc() {
  INTEROP_TRACE();
  return INTEROP_VOID_RETURN();
}

void FillHandles(std::vector<void*>& out) {
  INTEROP_TRACE(INTEROP_OUT(out));
  out.push_back((void*)0xA);
  out.push_back((void*)0xB);
  out.push_back((void*)0xC);
  return INTEROP_VOID_RETURN();
}

// ---------------------------------------------------------------------------
// Tests: disabled tracing
// ---------------------------------------------------------------------------

TEST_F(NoTracingTest, ZeroCostWhenDisabled) {
  EXPECT_EQ(UnannotatedFunction(5), 10);
}

TEST_F(NoTracingTest, RespectsEnvVar) {
  TraceInfo TI;
  EXPECT_FALSE(TI.isEnabled());
}

// ---------------------------------------------------------------------------
// Tests: basic tracing
// ---------------------------------------------------------------------------

TEST_F(TracingTest, DetectUnannotatedBranch) {
#ifndef NDEBUG
  EXPECT_DEATH({ UnannotatedFunction(5); }, "Unannotated exit branch detected");
#endif
}

// A function where INTEROP_TRACE is missing an argument — simulates a
// developer adding a parameter but forgetting to update the trace call.
void MissingTraceArg(int x, int y) {
  INTEROP_TRACE(x); // only passes x, not y
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, DetectMissingTraceArgument) {
#ifndef NDEBUG
  EXPECT_DEATH({ MissingTraceArg(1, 2); }, "argument count mismatch");
#endif
}

// A function with an out-param where INTEROP_OUT is forgotten.
void MissingOutParam(int x, std::vector<void*>& out) {
  INTEROP_TRACE(x); // missing INTEROP_OUT(out)
  out.push_back((void*)0x1);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, DetectMissingOutParam) {
#ifndef NDEBUG
  std::vector<void*> v;
  EXPECT_DEATH({ MissingOutParam(1, v); }, "argument count mismatch");
#endif
}

TEST_F(TracingTest, CountParamsHandlesVoidParamList) {
  // MSVC's __FUNCSIG__ uses "(void)" for zero-parameter functions.
  // Verify countParams handles both styles correctly.
  using TR = CppInterOp::Tracing::TraceRegion;
  EXPECT_EQ(TR::countParams("void foo()"), 0U);
  EXPECT_EQ(TR::countParams("void foo(void)"), 0U);
  EXPECT_EQ(TR::countParams("void *__cdecl Cpp::GetInterpreter(void)"), 0U);
  EXPECT_EQ(TR::countParams("void foo(int)"), 1U);
  EXPECT_EQ(TR::countParams("void foo(int, double)"), 2U);
  EXPECT_EQ(TR::countParams("void foo(std::vector<int>&)"), 1U);
  EXPECT_EQ(TR::countParams("void foo(std::map<int, int>&)"), 1U);
  EXPECT_EQ(TR::countParams("void foo(int, std::vector<int>&)"), 2U);
}

TEST_F(TracingTest, OutMaskLookupMatchesTd) {
  using TR = CppInterOp::Tracing::TraceRegion;
  // Public APIs marked OutArg<...> in CppInterOp.td.
  EXPECT_EQ(TR::lookupOutMask("GetAllCppNames"), 0b10ULL); // arg 1 OUT
  EXPECT_EQ(TR::lookupOutMask("CodeComplete"), 0b1ULL);    // arg 0 OUT
  EXPECT_EQ(TR::lookupOutMask("GetOperator"), 0b100ULL);   // arg 2 OUT
  // Public APIs without OUT args.
  EXPECT_EQ(TR::lookupOutMask("Process"), 0ULL);
  // Test helpers / unknown names produce nullopt -- the runtime check
  // skips them, so helpers like FillHandles can still use INTEROP_OUT.
  EXPECT_FALSE(TR::lookupOutMask("FillHandles").has_value());
}

TEST_F(TracingTest, ComputeOutMaskFromArgTypes) {
  using TR = CppInterOp::Tracing::TraceRegion;
  using OP = CppInterOp::Tracing::OutParam;
  EXPECT_EQ((TR::computeOutMask<int, double>()), 0ULL);
  EXPECT_EQ((TR::computeOutMask<int, OP>()), 0b10ULL);
  EXPECT_EQ((TR::computeOutMask<OP, int, OP>()), 0b101ULL);
}

TEST_F(TracingTest, FormatOutMaskMismatchMessage) {
  // Pure formatter; non-death test gives real coverage on the diagnostic
  // path that EXPECT_DEATH hides inside a forked child.
  using TR = CppInterOp::Tracing::TraceRegion;
  std::string Msg = TR::formatOutMaskMismatchMessage("FooApi", 0b10ULL, 0b1ULL);
  EXPECT_THAT(Msg, HasSubstr("INTEROP_OUT coverage mismatch in 'FooApi'"));
  EXPECT_THAT(Msg, HasSubstr("mask 0x2"));
  EXPECT_THAT(Msg, HasSubstr("args 0x1"));
}

// Scalar-pointer OUT helper: a function with a `bool*` out-param can
// now wrap it with INTEROP_OUT and the reproducer renders `nullptr`.
intptr_t ScalarOutDummy(const char* code, bool* err) {
  INTEROP_TRACE(code, INTEROP_OUT(err));
  if (err)
    *err = false;
  return INTEROP_RETURN((intptr_t)42);
}

TEST_F(TracingTest, ScalarPointerOutRendersAsNullptr) {
  bool err = true;
  EXPECT_EQ(ScalarOutDummy("x", &err), 42);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::ScalarOutDummy(\"x\", nullptr)"));
}

// Shadows the public API name Cpp::Process (which is not OUT in the .td)
// to drive lookupOutMask("Process") -> 0 while wrapping the arg with
// INTEROP_OUT yields actual=1; the mismatch must fire the OUT-coverage
// assert, not silently pass.
void Process(int* x) {
  INTEROP_TRACE(INTEROP_OUT(x));
  (void)x;
}

TEST_F(TracingTest, DetectOutCoverageMismatch) {
#ifndef NDEBUG
  int dummy = 0;
  EXPECT_DEATH({ Process(&dummy); }, "INTEROP_OUT coverage mismatch");
#endif
}

TEST_F(TracingTest, ValidAnnotatedBranch) {
  EXPECT_EQ(AnnotatedFunction(5), 10);
}

TEST_F(TracingTest, VoidFunctionTracing) {
  VoidFunc();
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::VoidFunc();"));
}

// ---------------------------------------------------------------------------
// Tests: INTEROP_TRACE with and without arguments (__VA_OPT__ coverage)
// ---------------------------------------------------------------------------

void NoArgTrace() {
  INTEROP_TRACE();
  return INTEROP_VOID_RETURN();
}

void WithOutParamTrace(std::vector<void*>& out) {
  INTEROP_TRACE(INTEROP_OUT(out));
  out.push_back((void*)0xDE);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, TraceWithNoArgs) {
  NoArgTrace();
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::NoArgTrace();"));
}

TEST_F(TracingTest, TraceWithOutParamArg) {
  // OUT pointer-container: preamble decl + alias at the call site.
  std::vector<void*> v;
  WithOutParamTrace(v);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, HasSubstr("Cpp::WithOutParamTrace(_out0);"));
  EXPECT_EQ(v.size(), 1u);
}

// ---------------------------------------------------------------------------
// Tests: argument formatting in reproducer output
// ---------------------------------------------------------------------------

void* FuncReturningHandle() {
  INTEROP_TRACE();
  return INTEROP_RETURN((void*)0xBEEF);
}

void FuncTakingHandle(void* h) {
  INTEROP_TRACE(h);
  return INTEROP_VOID_RETURN();
}

void FuncTakingString(const char* s) {
  INTEROP_TRACE(s);
  return INTEROP_VOID_RETURN();
}

void FuncTakingInt(int x) {
  INTEROP_TRACE(x);
  return INTEROP_VOID_RETURN();
}

void FuncTakingMixed(void* h, const char* s, int x) {
  INTEROP_TRACE(h, s, x);
  return INTEROP_VOID_RETURN();
}

void FuncWithHandleAndOut(void* h, std::vector<void*>& out) {
  INTEROP_TRACE(h, INTEROP_OUT(out));
  out.push_back((void*)0xF00);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingHandle) {
  void* h = FuncReturningHandle();
  FuncTakingHandle(h);
  auto output = getFullLog();
  // The handle should be registered as v1, then reused.
  EXPECT_THAT(output, HasSubstr("auto v1 = Cpp::FuncReturningHandle()"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingHandle(v1)"));
}

TEST_F(TracingTest, ArgFormattingUnregisteredHandleAnnotated) {
  // A non-null pointer the tracer never produced -- the call is the
  // first reference to it. The reproducer cannot bind it to a name,
  // so flag the gap with `nullptr /*unknown*/` rather than silently
  // collapsing to a bare `nullptr` (the previously-dead branch in
  // ReproBuffer::append; lookupHandle now returns "" for unregistered).
  // 0xDEADBEEF is unsigned int (>= 2^31) -- cast through uintptr_t
  // to keep MSVC's C4312 silent on 64-bit, where void* is wider.
  FuncTakingHandle(reinterpret_cast<void*>(static_cast<uintptr_t>(0xDEADBEEF)));
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingHandle(nullptr /*unknown*/)"));
}

TEST_F(TracingTest, ArgFormattingNullHandlePlain) {
  // Genuine null renders as plain `nullptr` -- never carries the
  // /*unknown*/ annotation, which is reserved for unregistered
  // non-null pointers.
  FuncTakingHandle(nullptr);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingHandle(nullptr)"));
  EXPECT_THAT(output, Not(HasSubstr("/*unknown*/")));
}

TEST_F(TracingTest, ArgFormattingString) {
  // Plain printable-ASCII content takes the `"..."` branch -- the
  // raw-literal form is reserved for content that would mis-encode as
  // a plain literal (see ArgFormattingString* tests below).
  FuncTakingString("hello");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingString(\"hello\")"));
}

TEST_F(TracingTest, ArgFormattingEmptyStringIsPlain) {
  // Empty string takes the no-special-char branch -- emit `""` rather
  // than `R"CPPI()CPPI"`.
  FuncTakingString("");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingString(\"\")"));
  EXPECT_THAT(output, Not(HasSubstr("R\"CPPI(")));
}

TEST_F(TracingTest, ArgFormattingStringWithEmbeddedDoubleQuote) {
  // Double quote forces the raw-literal fallback -- a plain `"..."`
  // would terminate the string at the embedded quote.
  FuncTakingString("a\"b");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("R\"CPPI(a\"b)CPPI\""));
}

TEST_F(TracingTest, ArgFormattingStringWithEmbeddedBackslash) {
  // Backslash forces raw-literal -- in a plain literal it would start
  // an escape sequence.
  FuncTakingString("a\\b");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("R\"CPPI(a\\b)CPPI\""));
}

TEST_F(TracingTest, ArgFormattingStringWithControlChar) {
  // Newline (a control char, < 0x20) forces raw-literal so the line
  // doesn't get split mid-string in the reproducer.
  FuncTakingString("a\nb");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("R\"CPPI(a\nb)CPPI\""));
}

TEST_F(TracingTest, ArgFormattingMultiLineStringUsesRawLiteral) {
  // Multi-line source blocks (typical of Cpp::Declare / Cpp::Process)
  // have several embedded newlines; raw-literal preserves the layout
  // verbatim so the reproducer compiles unchanged.
  FuncTakingString("void f() {\n  return 42;\n}\n");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output,
              HasSubstr("R\"CPPI(void f() {\n  return 42;\n}\n)CPPI\""));
}

TEST_F(TracingTest, ArgFormattingStringWithHighBitStaysPlain) {
  // High-bit (UTF-8) bytes are >= 0x20 and != 0x7f, so they keep the
  // plain-literal path -- the source bytes pass through unchanged and
  // the reproducer compiles with the same encoding.
  FuncTakingString("r\xc3\xa9sum\xc3\xa9"); // "résumé" in UTF-8
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output,
              HasSubstr("Cpp::FuncTakingString(\"r\xc3\xa9sum\xc3\xa9\")"));
  EXPECT_THAT(output, Not(HasSubstr("R\"CPPI(")));
}

TEST_F(TracingTest, ArgFormattingInt) {
  FuncTakingInt(42);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingInt(42)"));
}

TEST_F(TracingTest, ArgFormattingMixed) {
  void* h = FuncReturningHandle();
  FuncTakingMixed(h, "world", 99);
  auto output = getFullLog();
  EXPECT_THAT(output,
              HasSubstr("Cpp::FuncTakingMixed(v1, \"world\", 99)"));
}

TEST_F(TracingTest, ArgFormattingOutParamAliased) {
  // OUT alias preserves left-to-right argument order: (v1, _out0).
  void* h = FuncReturningHandle();
  std::vector<void*> out;
  FuncWithHandleAndOut(h, out);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncWithHandleAndOut(v1, _out0);"));
}

void FuncWithTwoOuts(std::vector<void*>& a, std::vector<void*>& b) {
  INTEROP_TRACE(INTEROP_OUT(a), INTEROP_OUT(b));
  a.push_back((void*)0x11);
  b.push_back((void*)0x22);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingTwoOutParamsGetDistinctIndices) {
  // Two OUTs on one call get distinct indices in arg order.
  std::vector<void*> a, b;
  FuncWithTwoOuts(a, b);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out1;"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncWithTwoOuts(_out0, _out1);"));
}

TEST_F(TracingTest, ArgFormattingOutParamDistinctSourcesGetDistinctIndices) {
  // Two calls with DIFFERENT source vectors get distinct `_outN`
  // slots, both declared in the preamble.
  std::vector<void*> a, b;
  WithOutParamTrace(a);
  WithOutParamTrace(b);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out1;"));
  EXPECT_THAT(output, HasSubstr("Cpp::WithOutParamTrace(_out0);"));
  EXPECT_THAT(output, HasSubstr("Cpp::WithOutParamTrace(_out1);"));
}

TEST_F(TracingTest, ArgFormattingOutParamSameSourceAliasesSlot) {
  // Reusing the same source vector across calls reuses the same
  // `_outN` -- one preamble decl, both call sites alias it -- so the
  // reproducer faithfully replays the API's append-on-call contract.
  std::vector<void*> v;
  WithOutParamTrace(v);
  WithOutParamTrace(v);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, Not(HasSubstr("std::vector<void*> _out1;")));
  EXPECT_THAT(output, HasSubstr("Cpp::WithOutParamTrace(_out0);"));
  EXPECT_THAT(output, Not(HasSubstr("Cpp::WithOutParamTrace(_out1);")));
}

TEST_F(TracingTest, OutParamElementUsableInLaterCall) {
  // The OUT-param element decl makes the handle bind in the
  // reproducer: a downstream call that takes one of the elements as
  // input renders to a name that's actually declared earlier.
  void* h = FuncReturningHandle();
  std::vector<void*> out;
  FuncWithHandleAndOut(h, out);
  ASSERT_FALSE(out.empty());
  FuncTakingHandle(out[0]);
  auto output = getFullLog();
  // The element's decl reads `_out0[0]` and binds a handle name (vN);
  // the downstream call references the same name.
  TraceInfo& TI = *TheTraceInfo;
  std::string elem = TI.lookupHandle(out[0]);
  ASSERT_FALSE(elem.empty());
  EXPECT_THAT(output,
              HasSubstr("void* " + elem + " = _out0.size() > 0 ? _out0[0]"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingHandle(" + elem + ")"));
}

// Test: when two functions return the same pointer, the second should not
// redeclare the handle variable — it should emit a comment instead.
void* ReturnSameHandle() {
  INTEROP_TRACE();
  return INTEROP_RETURN((void*)0xDA1E);
}

TEST_F(TracingTest, DuplicateHandleNotRedeclared) {
  void* h1 = ReturnSameHandle(); // first call → "auto v1 = ..."
  void* h2 = ReturnSameHandle(); // same pointer → "/*v1*/ ..."
  (void)h1;
  (void)h2;
  auto output = getFullLog();

  // Count occurrences of "auto v1 ="
  size_t firstDecl = output.find("auto v1 =");
  ASSERT_NE(firstDecl, std::string::npos) << "First call should declare v1";

  // The second occurrence should NOT be "auto v1 =" but "/*v1*/"
  size_t secondCall = output.find("Cpp::ReturnSameHandle()", firstDecl + 1);
  ASSERT_NE(secondCall, std::string::npos) << "Should have a second call";

  // There should be exactly one "auto v1 =" in the output.
  size_t secondDecl = output.find("auto v1 =", firstDecl + 1);
  EXPECT_EQ(secondDecl, std::string::npos)
      << "Handle v1 should not be redeclared. Output:\n"
      << output;

  // The second call should have the comment form.
  EXPECT_THAT(output, HasSubstr("/*v1*/"));
}

// Test: when a pointer-returning function returns nullptr, the output
// should annotate it as /*nullptr*/ so the reader knows.
void* ReturnNull() {
  INTEROP_TRACE();
  return INTEROP_RETURN(nullptr);
}

TEST_F(TracingTest, NullptrReturnAnnotated) {
  ReturnNull();
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("/*nullptr*/"));
  EXPECT_THAT(output, HasSubstr("Cpp::ReturnNull()"));
}

// Test: non-void, non-pointer returns (int, bool, size_t) should NOT
// get any handle annotation.
int ReturnInt() {
  INTEROP_TRACE();
  return INTEROP_RETURN(42);
}

TEST_F(TracingTest, NonPointerReturnNoAnnotation) {
  ReturnInt();
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, Not(HasSubstr("auto")));
  EXPECT_THAT(output, Not(HasSubstr("nullptr")));
  EXPECT_THAT(output, HasSubstr("Cpp::ReturnInt()"));
}

// Vector input args are formatted as braced init lists of
// recursively-formatted elements -- no longer the `{...}` placeholder
// that produced uncompilable reproducers.
void FuncTakingStrVector(const std::vector<const char*>& args) {
  INTEROP_TRACE(args);
  return INTEROP_VOID_RETURN();
}

void FuncTakingIntVector(const std::vector<int>& args) {
  INTEROP_TRACE(args);
  return INTEROP_VOID_RETURN();
}

void FuncTakingNestedVector(const std::vector<std::vector<int>>& args) {
  INTEROP_TRACE(args);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingVectorOfStrings) {
  // Exercises the per-element loop with the `, ` separator (>1 element)
  // and the const-char* overload of append (plain-quote form).
  std::vector<const char*> v = {"a", "b"};
  FuncTakingStrVector(v);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingStrVector({\"a\", \"b\"})"));
}

TEST_F(TracingTest, ArgFormattingVectorOfInts) {
  // Pins the integer-element path through append(int).
  std::vector<int> v = {1, 2, 3};
  FuncTakingIntVector(v);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingIntVector({1, 2, 3})"));
}

TEST_F(TracingTest, ArgFormattingEmptyVector) {
  // Empty vector skips the loop -- emit `{}` with no separator. Pins
  // the First-flag branch where it never flips.
  std::vector<int> v;
  FuncTakingIntVector(v);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingIntVector({})"));
}

TEST_F(TracingTest, ArgFormattingNestedVector) {
  // Recursive call: append(vector<vector<int>>) dispatches to
  // append(vector<int>) for each element.
  std::vector<std::vector<int>> v = {{1, 2}, {3}};
  FuncTakingNestedVector(v);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingNestedVector({{1, 2}, {3}})"));
}

// ---------------------------------------------------------------------------
// Tests: returned std::vector<P*> -- elements registered as handles
// ---------------------------------------------------------------------------

std::vector<void*> ReturnPointerVector() {
  INTEROP_TRACE();
  std::vector<void*> v;
  v.push_back((void*)0x101);
  v.push_back((void*)0x202);
  return INTEROP_RETURN(v);
}

std::vector<void*> ReturnEmptyPointerVector() {
  INTEROP_TRACE();
  std::vector<void*> v;
  return INTEROP_RETURN(v);
}

std::vector<void*> ReturnVectorWithNullElement() {
  INTEROP_TRACE();
  std::vector<void*> v;
  v.push_back((void*)0x303);
  v.push_back(nullptr);
  v.push_back((void*)0x404);
  return INTEROP_RETURN(v);
}

TEST_F(TracingTest, ReturnedPointerVectorProducerAndConsumer) {
  // Producer side: `auto _retN = ...` prelude + bounds-guarded
  // `_retN[i]` per element. Consumer side: a subsequent call taking
  // one of those pointers prints it by the same handle name (vN), so
  // the reproducer's producing line and the consumer's argument refer
  // to the same binding.
  TraceInfo& TI = *TheTraceInfo;
  auto v = ReturnPointerVector();
  std::string h0 = TI.lookupHandle(v[0]);
  std::string h1 = TI.lookupHandle(v[1]);
  ASSERT_NE(h0, "nullptr");
  ASSERT_NE(h1, "nullptr");

  FuncTakingHandle(v[0]);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("auto _ret0 = Cpp::ReturnPointerVector()"));
  EXPECT_THAT(output,
              HasSubstr("void* " + h0 + " = _ret0.size() > 0 ? _ret0[0]"));
  EXPECT_THAT(output,
              HasSubstr("void* " + h1 + " = _ret0.size() > 1 ? _ret0[1]"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingHandle(" + h0 + ")"));
}

TEST_F(TracingTest, ReturnedPointerVectorEmptyEmitsNoDecls) {
  // Empty vector returns the placeholder + zero element decls. The
  // RetDecls loop never appends, so the only log line for this call is
  // the call itself.
  size_t before = TheTraceInfo->getLog().size();
  ReturnEmptyPointerVector();
  size_t after = TheTraceInfo->getLog().size();
  EXPECT_EQ(after - before, 1u)
      << "Empty vector return should emit only the call line, no element decls";
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output,
              HasSubstr("auto _ret0 = Cpp::ReturnEmptyPointerVector()"));
}

TEST_F(TracingTest, ReturnedPointerVectorDuplicateElementSkipsDecl) {
  // If two returned vectors share an element pointer, only the first
  // call registers (and emits a decl for) it; the second skips
  // because lookupHandle no longer says "nullptr".
  ReturnPointerVector();
  ReturnPointerVector();
  auto output = getFullLog();
  // First call's decls are present; the second call's decls are
  // suppressed (the elements already have names).
  EXPECT_THAT(output, HasSubstr("auto _ret0 = Cpp::ReturnPointerVector()"));
  EXPECT_THAT(output, HasSubstr("auto _ret1 = Cpp::ReturnPointerVector()"));
  // _ret1 should have no `void* vN = _ret1[i]` lines for elements that
  // were already registered.
  EXPECT_THAT(output, Not(HasSubstr("_ret1.size() > 0 ? _ret1[0]")));
}

TEST_F(TracingTest, ReturnedPointerVectorNullElementSkipsDecl) {
  // Null elements are skipped (`if (!p) continue;`) so the reproducer
  // doesn't emit a decl that would shadow the global `nullptr`. The
  // surrounding non-null elements still get their decls at their
  // original indices.
  ReturnVectorWithNullElement();
  auto output = getFullLog();
  EXPECT_THAT(output,
              HasSubstr("auto _ret0 = Cpp::ReturnVectorWithNullElement()"));
  EXPECT_THAT(output, HasSubstr("_ret0.size() > 0 ? _ret0[0] : nullptr"));
  EXPECT_THAT(output, Not(HasSubstr("_ret0.size() > 1 ? _ret0[1]")));
  EXPECT_THAT(output, HasSubstr("_ret0.size() > 2 ? _ret0[2] : nullptr"));
}

// ---------------------------------------------------------------------------
// Reproducer-compilability fixtures: enum cast, escaped strings, const void*.
// Each one used to emit "?" or invalid C++ in the captured trace; the tests
// pin the new behaviour so the auto-generated reproducer compiles.
// ---------------------------------------------------------------------------

namespace tracing_test_enum {
enum Flavor : unsigned char { Red = 1, Green = 2, Blue = 3 };
} // namespace tracing_test_enum

void FuncTakingEnum(tracing_test_enum::Flavor f) {
  INTEROP_TRACE(f);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingEnumStaticCast) {
  // Enums route through the std::is_enum_v overload and emit
  // "static_cast<EnumName>(N)" so the reproducer compiles. Used to
  // emit "?" via the catch-all template.
  FuncTakingEnum(tracing_test_enum::Blue);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("static_cast<"));
  EXPECT_THAT(output, HasSubstr("Flavor"));
  EXPECT_THAT(output, HasSubstr(">(3)"));
  EXPECT_THAT(output, Not(HasSubstr("?")));
}

void FuncTakingTrickyString(const char* s) {
  INTEROP_TRACE(s);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingStringEscapes) {
  // Strings are emitted as a raw literal R"CPPI(...)CPPI" so newlines,
  // quotes, and backslashes flow through verbatim and the reproducer
  // compiles without a separate escape pass.
  FuncTakingTrickyString("line1\nline2\twith \"quotes\" and a \\backslash");
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("R\"CPPI("));
  EXPECT_THAT(output, HasSubstr(")CPPI\""));
  // Verbatim content survives -- the raw literal preserves the
  // newline / tab / quote / backslash bytes as-is.
  EXPECT_THAT(output,
              HasSubstr("line1\nline2\twith \"quotes\" and a \\backslash"));
}

const void* FuncReturningConstHandle() {
  INTEROP_TRACE();
  return INTEROP_RETURN((const void*)0xC0FFEE);
}

void FuncTakingConstHandle(const void* h) {
  INTEROP_TRACE(h);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingConstVoidHandle) {
  // const void* takes the second overload; before this fix the catch-all
  // template absorbed "const T&" matches and emitted "?".
  const void* h = FuncReturningConstHandle();
  FuncTakingConstHandle(h);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("auto v1 = Cpp::FuncReturningConstHandle()"));
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingConstHandle(v1)"));
  EXPECT_THAT(output, Not(HasSubstr("?")));
}

TEST_F(TracingTest, ArgFormattingConstVoidNullptr) {
  // null const void* must emit literal nullptr, not the empty handle name.
  FuncTakingConstHandle(nullptr);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingConstHandle(nullptr)"));
}

void FuncTakingTemplateArg(Cpp::TemplateArgInfo tai) {
  INTEROP_TRACE(tai);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, ArgFormattingTemplateArgInfoTypeAndIntegralValue) {
  // m_Type renders as the registered handle name; m_IntegralValue
  // takes the raw-string path.
  TraceInfo& TI = *TheTraceInfo;
  void* tp = reinterpret_cast<void*>(static_cast<uintptr_t>(0x12340000));
  std::string h = TI.getOrRegisterHandle(tp);
  Cpp::TemplateArgInfo tai{tp, "5"};
  FuncTakingTemplateArg(tai);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingTemplateArg(Cpp::"
                                "TemplateArgInfo{" +
                                h + ", \"5\"})"));
}

TEST_F(TracingTest, ArgFormattingTemplateArgInfoNullFields) {
  // Null m_IntegralValue must round-trip to `nullptr` (the
  // constructor's default), not the empty string `""`.
  Cpp::TemplateArgInfo tai{nullptr, nullptr};
  FuncTakingTemplateArg(tai);
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("Cpp::FuncTakingTemplateArg(Cpp::"
                                "TemplateArgInfo{nullptr, nullptr})"));
}

// ---------------------------------------------------------------------------
// Tests: handle registry
// ---------------------------------------------------------------------------

TEST_F(TracingTest, HandleRegistry) {
  TraceInfo& TI = *TheTraceInfo;
  void* ptr1 = (void*)0x1234;
  void* ptr2 = (void*)0x5678;

  std::string v1 = TI.getOrRegisterHandle(ptr1);
  EXPECT_FALSE(v1.empty());
  EXPECT_EQ(v1[0], 'v');

  EXPECT_EQ(TI.getOrRegisterHandle(ptr1), v1);
  EXPECT_EQ(TI.lookupHandle(ptr1), v1);

  std::string v2 = TI.getOrRegisterHandle(ptr2);
  EXPECT_NE(v1, v2);
  EXPECT_EQ(TI.lookupHandle(ptr2), v2);

  EXPECT_EQ(TI.lookupHandle(nullptr), "nullptr");

  // An unregistered non-null pointer returns "" so callers can tell
  // "the tracer has no handle for this" from "this is the literal
  // nullptr". The two cases used to collide on "nullptr" and the
  // distinction is what lets ReproBuffer annotate unknown args.
  EXPECT_EQ(TI.lookupHandle((const void*)0xBADADD), "");

  // getOrRegisterHandle(nullptr) should return empty, not register.
  EXPECT_EQ(TI.getOrRegisterHandle(nullptr), "");
}

// ---------------------------------------------------------------------------
// Tests: timer stacking
// ---------------------------------------------------------------------------

TEST_F(TracingTest, TimerStackPushPop) {
  TraceInfo& TI = *TheTraceInfo;

  llvm::Timer& T1 = TI.getTimer("FuncA");
  llvm::Timer& T2 = TI.getTimer("FuncB");

  // Push T1 — it should be running.
  TI.pushTimer(&T1);
  EXPECT_TRUE(T1.isRunning());

  // Push T2 — T1 paused, T2 running (covers line 75: nested stop).
  TI.pushTimer(&T2);
  EXPECT_FALSE(T1.isRunning());
  EXPECT_TRUE(T2.isRunning());

  // Pop T2 — T1 resumes (covers line 86: restart parent).
  TI.popTimer();
  EXPECT_TRUE(T1.isRunning());
  EXPECT_FALSE(T2.isRunning());

  // Pop T1 — nothing running.
  TI.popTimer();
  EXPECT_FALSE(T1.isRunning());

  // Pop on empty stack — should not crash (covers line 82).
  TI.popTimer();
}

TEST_F(TracingTest, ClearWithRunningTimers) {
  TraceInfo& TI = *TheTraceInfo;

  llvm::Timer& T1 = TI.getTimer("RunningTimer");
  TI.pushTimer(&T1);
  EXPECT_TRUE(T1.isRunning());

  // clear() stops running timers and destroys them.
  // We only verify it doesn't crash — T1 is invalid after clear().
  TI.clear();
}

// ---------------------------------------------------------------------------
// Tests: StartTracing activates tracing if not yet active
// ---------------------------------------------------------------------------

TEST(StartTracingActivation, ActivatesIfNotActive) {
  // Save and clear TheTraceInfo to simulate tracing not being active.
  auto* saved = TheTraceInfo;
  TheTraceInfo = nullptr;

  // StartTracing should call InitTracing (covers line 151).
  std::string Path = CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  ASSERT_NE(TheTraceInfo, nullptr);
  ASSERT_FALSE(Path.empty());

  CppInterOp::Tracing::StopTracing();
  llvm::sys::fs::remove(Path);

  // Restore so other tests aren't affected.
  TheTraceInfo = saved;
}

// ---------------------------------------------------------------------------
// Tests: countParams edge cases
// ---------------------------------------------------------------------------

TEST_F(TracingTest, CountParamsEdgeCases) {
  using TR = CppInterOp::Tracing::TraceRegion;

  // No parenthesis at all (covers line 333).
  EXPECT_EQ(TR::countParams("nosig"), 0U);

  // Spaces after opening paren (covers line 338).
  EXPECT_EQ(TR::countParams("void foo( int x )"), 1U);
  EXPECT_EQ(TR::countParams("void foo(  )"), 0U);
}

// ---------------------------------------------------------------------------
// Tests: handle chaining across calls
// ---------------------------------------------------------------------------

TEST_F(TracingTest, ChainedReproducerLogic) {
  {
    void* h;
    {
      INTEROP_TRACE();
      h = (void*)0x111;
      INTEROP_RETURN(h);
    }
    {
      INTEROP_TRACE();
      INTEROP_RETURN(0);
    }
  }
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("auto v1 ="));
}

// ---------------------------------------------------------------------------
// Tests: nested-call suppression
// ---------------------------------------------------------------------------

void* NestedInnerReturn() {
  INTEROP_TRACE();
  return INTEROP_RETURN((void*)0x2000);
}

void* NestedOuterReturn() {
  INTEROP_TRACE();
  return INTEROP_RETURN(NestedInnerReturn());
}

TEST_F(TracingTest, NestedCallSuppressedButHandleRegistered) {
  // The inner call doesn't appear in the log, yet its handle is
  // registered so the outer call's `auto vN = ...` line binds the
  // pointer the inner produced.
  void* h = NestedOuterReturn();
  TraceInfo& TI = *TheTraceInfo;
  EXPECT_FALSE(TI.lookupHandle(h).empty());
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("auto v1 = Cpp::NestedOuterReturn()"));
  EXPECT_THAT(output, Not(HasSubstr("Cpp::NestedInnerReturn")));
}

void NestedInnerNoOp() {
  INTEROP_TRACE();
  return INTEROP_VOID_RETURN();
}

void NestedOuterWithOut(std::vector<void*>& out) {
  INTEROP_TRACE(INTEROP_OUT(out));
  NestedInnerNoOp();
  out.push_back((void*)0x3000);
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, NestedSuppressionDoesNotLeakOutCounter) {
  // captureArg must not bump m_OutCount for nested OUT-params --
  // otherwise the outer call would emit `_out1` while the inner
  // claimed `_out0` silently.
  std::vector<void*> v;
  NestedOuterWithOut(v);
  auto output = getFullLog();
  EXPECT_THAT(output, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(output, HasSubstr("Cpp::NestedOuterWithOut(_out0);"));
  EXPECT_THAT(output, Not(HasSubstr("_out1")));
}

// ---------------------------------------------------------------------------
// Tests: ctor-side placeholder for aborted calls
// ---------------------------------------------------------------------------

void PlaceholderProbe() {
  INTEROP_TRACE();
  // Mid-call: the ctor-emitted placeholder is already in the log.
  auto& log = TheTraceInfo->getLog();
  ASSERT_FALSE(log.empty());
  EXPECT_THAT(log.back(), HasSubstr("Cpp::PlaceholderProbe"));
  EXPECT_THAT(log.back(), HasSubstr("[aborted before return]"));
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, PlaceholderInCtorReplacedByDtor) {
  // Mid-call assertions live inside PlaceholderProbe; here we only
  // verify the replacement: after the call returns, the slot reads
  // the completed `// [N ns]` line, not the placeholder.
  PlaceholderProbe();
  auto& log = TheTraceInfo->getLog();
  ASSERT_FALSE(log.empty());
  EXPECT_THAT(log.back(), Not(HasSubstr("aborted before return")));
  EXPECT_THAT(log.back(),
              ::testing::MatchesRegex(R"(.*Cpp::PlaceholderProbe.*ns\].*)"));
}

// ---------------------------------------------------------------------------
// Tests: out-parameters
// ---------------------------------------------------------------------------

TEST_F(TracingTest, OutParamRegistersHandles) {
  TraceInfo& TI = *TheTraceInfo;

  std::vector<void*> results;
  FillHandles(results);

  // The three handles pushed inside FillHandles should now be registered.
  EXPECT_NE(TI.lookupHandle((void*)0xA), "nullptr");
  EXPECT_NE(TI.lookupHandle((void*)0xB), "nullptr");
  EXPECT_NE(TI.lookupHandle((void*)0xC), "nullptr");
}

TEST_F(TracingTest, OutParamHandlesUsableInLaterCalls) {
  TraceInfo& TI = *TheTraceInfo;

  std::vector<void*> results;
  FillHandles(results);

  // The handle should have been registered by the OutParam mechanism.
  std::string handleName = TI.lookupHandle(results[0]);
  EXPECT_EQ(handleName[0], 'v');
}

TEST_F(NoTracingTest, OutParamNoOpWhenDisabled) {
  std::vector<void*> results;
  // Should not crash when tracing is disabled.
  FillHandles(results);
  EXPECT_EQ(results.size(), 3u);
}

// Out-param with non-pointer elements (e.g., std::vector<std::string>)
// should compile and work — handles are only registered for pointer types.
void FillStrings(std::vector<std::string>& out) {
  INTEROP_TRACE(INTEROP_OUT(out));
  out.push_back("hello");
  out.push_back("world");
  return INTEROP_VOID_RETURN();
}

TEST_F(TracingTest, OutParamNonPointerElementType) {
  // Non-pointer OUTs stay on the legacy skip path -- no `_outN`
  // preamble is emitted and the call has zero args.
  std::vector<std::string> results;
  FillStrings(results);
  auto output = getFullLog();

  EXPECT_EQ(results.size(), 2u);
  EXPECT_EQ(results[0], "hello");
  EXPECT_EQ(results[1], "world");
  EXPECT_THAT(output, HasSubstr("Cpp::FillStrings();"));
  EXPECT_THAT(output, Not(HasSubstr("_out0")));
}

TEST_F(NoTracingTest, OutParamNonPointerDisabled) {
  std::vector<std::string> results;
  FillStrings(results);
  EXPECT_EQ(results.size(), 2u);
}

// ---------------------------------------------------------------------------
// Tests: writeToFile
// ---------------------------------------------------------------------------

TEST_F(TracingTest, WriteToFileProducesValidReproducer) {
  // Generate some trace entries.
  {
    INTEROP_TRACE();
    INTEROP_RETURN((void*)0xF00);
  }
  {
    INTEROP_TRACE();
    return INTEROP_VOID_RETURN();
  }

  TraceInfo& TI = *TheTraceInfo;
  std::string Path = TI.writeToFile();
  ASSERT_FALSE(Path.empty());

  // Read the file back.
  std::ifstream ifs(Path);
  ASSERT_TRUE(ifs.good());
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

  // Both prologue branches must travel together.
  EXPECT_THAT(content, HasSubstr("#ifdef CPPINTEROP_USE_DISPATCH"));
  EXPECT_THAT(content, HasSubstr("#include <CppInterOp/Dispatch.h>"));
  EXPECT_THAT(content, HasSubstr("#include <CppInterOp/CppInterOp.h>"));
  EXPECT_THAT(content, HasSubstr("CPPINTEROP_API_FUNC"));
  EXPECT_THAT(content, HasSubstr("Cpp::LoadDispatchAPI"));
  EXPECT_THAT(content, HasSubstr("void reproducer()"));
  EXPECT_THAT(content, HasSubstr("Cpp::"));
  EXPECT_THAT(content, HasSubstr("Build context"));
  EXPECT_THAT(content, HasSubstr("build type:"));
  EXPECT_THAT(content, HasSubstr("cmake-cache:"));

  // Clean up.
  llvm::sys::fs::remove(Path);
}

TEST_F(TracingTest, WriteToFileContainsOutParamCalls) {
  std::vector<void*> results;
  FillHandles(results);

  TraceInfo& TI = *TheTraceInfo;
  std::string Path = TI.writeToFile();
  ASSERT_FALSE(Path.empty());

  std::ifstream ifs(Path);
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());

  // Persisted reproducer mirrors the in-memory log.
  EXPECT_THAT(content, HasSubstr("std::vector<void*> _out0;"));
  EXPECT_THAT(content, HasSubstr("Cpp::FillHandles(_out0);"));

  llvm::sys::fs::remove(Path);
}

TEST_F(TracingTest, WriteToFileSuppressesSelfTrace) {
  // Without the m_Dumping guard, GetVersion/GetBuildInfo (both
  // INTEROP_TRACE'd) would leak into the reproducer body.
  AnnotatedFunction(7);
  AnnotatedFunction(8);

  TraceInfo& TI = *TheTraceInfo;
  size_t before = TI.getLog().size();
  std::string Path = TI.writeToFile();
  ASSERT_FALSE(Path.empty());
  EXPECT_EQ(TI.getLog().size(), before);

  std::ifstream ifs(Path);
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  EXPECT_THAT(content, Not(HasSubstr("Cpp::GetVersion()")));
  EXPECT_THAT(content, Not(HasSubstr("Cpp::GetBuildInfo()")));
  EXPECT_THAT(content, HasSubstr("// Replay scope:"));

  llvm::sys::fs::remove(Path);
}

// ---------------------------------------------------------------------------
// Tests: reproducer compiles via interpreter
// ---------------------------------------------------------------------------

#ifndef EMSCRIPTEN
TEST_F(TracingTest, ReproducerCompilesViaInterpreter) {
  // Create an interpreter so we can exercise real API calls.
  Cpp::CreateInterpreter({});

  // Verify tracing is active after interpreter creation.
  ASSERT_NE(TheTraceInfo, nullptr)
      << "TheTraceInfo was cleared during CreateInterpreter";

  // Clear any prior trace state so the reproducer only contains our calls.
  TheTraceInfo->clear();

  // Exercise real API calls that will be recorded.
  auto GlobalScope = Cpp::GetGlobalScope();
  ASSERT_NE(GlobalScope, nullptr);

  Cpp::Declare("namespace ReproNS { struct Foo { int x; }; }");

  auto ReproNS = Cpp::GetScope("ReproNS", GlobalScope);
  ASSERT_NE(ReproNS, nullptr);

  auto Foo = Cpp::GetScope("Foo", ReproNS);
  ASSERT_NE(Foo, nullptr);

  EXPECT_TRUE(Cpp::IsClass(Foo));
  EXPECT_FALSE(Cpp::IsNamespace(Foo));

  auto FooType = Cpp::GetTypeFromScope(Foo);
  ASSERT_NE(FooType, nullptr);

  EXPECT_FALSE(Cpp::IsFunctionProtoType(FooType));

  Cpp::GetName(Foo);
  Cpp::SizeOf(Foo);

  // Write the reproducer.
  TraceInfo& TI = *TheTraceInfo;
  std::string Path = TI.writeToFile();
  ASSERT_FALSE(Path.empty());

  // Read it back.
  std::ifstream ifs(Path);
  ASSERT_TRUE(ifs.good());
  std::string content((std::istreambuf_iterator<char>(ifs)),
                      std::istreambuf_iterator<char>());
  ifs.close();

  // Verify the reproducer has proper structure and content.
  EXPECT_THAT(content, HasSubstr("#include <CppInterOp/Dispatch.h>"));
  EXPECT_THAT(content, HasSubstr("void reproducer()"));
  EXPECT_THAT(content, HasSubstr("Cpp::"));

  // Create a fresh interpreter and #include the reproducer file as-is.
  Cpp::CreateInterpreter({});
  Cpp::AddIncludePath(CPPINTEROP_DIR "/include");
  // Generated .inc files live under the build tree.
  Cpp::AddIncludePath(CPPINTEROP_BINARY_DIR "/include");

  std::string includeDirective = "#include \"" + Path + "\"";
  EXPECT_EQ(Cpp::Process(includeDirective.c_str()), 0)
      << "Reproducer failed to compile via #include:\n"
      << content;

  // Drives the static-link `#else` branch via the in-process interpreter;
  // the dispatch path is for standalone builds, not exercised here.
  EXPECT_EQ(Cpp::Process("reproducer();"), 0) << "Reproducer failed to execute";

  llvm::sys::fs::remove(Path);
}
#endif // !EMSCRIPTEN

// ---------------------------------------------------------------------------
// Tests: JitCall wrapper source and Invoke are logged
// ---------------------------------------------------------------------------

TEST_F(TracingTest, JitCallWrapperSourceLogged) {
#ifdef EMSCRIPTEN
#if CLANG_VERSION_MAJOR > 21
  GTEST_SKIP() << "Test fails for Emscipten builds using LLVM 22";
#endif
#endif
  Cpp::CreateInterpreter({});
  ASSERT_NE(TheTraceInfo, nullptr);

  // Placement new requires <new>.
  Cpp::Declare("#include <new>");
  Cpp::Declare("namespace WrapNS { int add(int a, int b) { return a + b; } }");
  auto Func = Cpp::GetNamed("add", Cpp::GetScope("WrapNS"));
  ASSERT_TRUE(Func);

  TheTraceInfo->clear();

  // MakeFunctionCallable triggers make_wrapper which logs the wrapper source.
  auto JC = Cpp::MakeFunctionCallable(Cpp::FuncRef{Func.data});
  ASSERT_TRUE(JC.isValid());

  auto log = getFullLog();
  EXPECT_THAT(log, HasSubstr("=== Wrapper for"));
  EXPECT_THAT(log, HasSubstr("=== End wrapper ==="));
  // The wrapper source should contain the function name.
  EXPECT_THAT(log, HasSubstr("add"));
}

#ifndef NDEBUG
TEST_F(TracingTest, JitCallInvokeLogged) {
  Cpp::CreateInterpreter({});
  ASSERT_NE(TheTraceInfo, nullptr);

  Cpp::Declare("#include <new>");
  Cpp::Declare("namespace InvNS { int square(int x) { return x * x; } }");
  auto Func = Cpp::GetNamed("square", Cpp::GetScope("InvNS"));
  ASSERT_TRUE(Func);

  auto JC = Cpp::MakeFunctionCallable(Cpp::FuncRef{Func.data});
  ASSERT_TRUE(JC.isValid());

  // Clear so only the Invoke shows up.
  TheTraceInfo->clear();

  int arg = 7;
  int result = 0;
  void* args[] = {&arg};
  JC.Invoke(&result, {args, 1});

  auto log = getFullLog();
  // ReportInvokeStart should have logged the invoke.
  EXPECT_THAT(log, HasSubstr("JitCall::Invoke"));
  EXPECT_THAT(log, HasSubstr("square"));
  EXPECT_THAT(log, HasSubstr("nargs=1"));

  // And the function should have actually executed.
  EXPECT_EQ(result, 49);
}

TEST_F(TracingTest, JitCallConstructorInvokeLogged) {
  // Pins the InvokeConstructor inline-guard: ReportInvokeStart must fire
  // for kConstructorCall just as it does for kGenericCall.
  Cpp::CreateInterpreter({});
  ASSERT_NE(TheTraceInfo, nullptr);

  Cpp::Declare("#include <new>");
  Cpp::Declare("namespace InvCtorNS { struct Bar { int x = 7; }; }");
  auto ClassBar = Cpp::GetScope("Bar", Cpp::GetScope("InvCtorNS"));
  ASSERT_TRUE(ClassBar);
  auto CtorD = Cpp::GetDefaultConstructor(ClassBar);
  ASSERT_TRUE(CtorD);
  // Resolve the dtor up-front so the cleanup at the end of the test
  // can free the heap-allocated Bar (LeakSanitizer otherwise flags
  // the 4-byte `int x` payload).
  auto DtorD = Cpp::GetDestructor(ClassBar);
  ASSERT_TRUE(DtorD);

  auto CtorJC = Cpp::MakeFunctionCallable(CtorD);
  auto DtorJC = Cpp::MakeFunctionCallable(DtorD);
  ASSERT_TRUE(CtorJC.isValid());
  ASSERT_TRUE(DtorJC.isValid());

  TheTraceInfo->clear();
  void* obj = nullptr;
  CtorJC.Invoke(&obj);
  ASSERT_NE(obj, nullptr);

  auto log = getFullLog();
  EXPECT_THAT(log, HasSubstr("JitCall::Invoke"));
  EXPECT_THAT(log, HasSubstr("Bar"));

  DtorJC.Invoke(obj);
}

TEST_F(TracingTest, JitCallDestructorInvokeLogged) {
  // Pins the InvokeDestructor inline-guard: ReportInvokeStart must emit
  // a JitCall::InvokeDestructor line for the dtor path.
  Cpp::CreateInterpreter({});
  ASSERT_NE(TheTraceInfo, nullptr);

  Cpp::Declare("#include <new>");
  Cpp::Declare("namespace InvDtorNS { struct Baz { ~Baz() {} }; }");
  auto ClassBaz = Cpp::GetScope("Baz", Cpp::GetScope("InvDtorNS"));
  ASSERT_TRUE(ClassBaz);
  auto CtorD = Cpp::GetDefaultConstructor(ClassBaz);
  ASSERT_TRUE(CtorD);
  auto DtorD = Cpp::GetDestructor(ClassBaz);
  ASSERT_TRUE(DtorD);

  auto CtorJC = Cpp::MakeFunctionCallable(CtorD);
  auto DtorJC = Cpp::MakeFunctionCallable(DtorD);
  ASSERT_TRUE(CtorJC.isValid());
  ASSERT_TRUE(DtorJC.isValid());

  void* obj = nullptr;
  CtorJC.Invoke(&obj);
  ASSERT_NE(obj, nullptr);

  TheTraceInfo->clear();
  DtorJC.Invoke(obj);

  auto log = getFullLog();
  EXPECT_THAT(log, HasSubstr("JitCall::InvokeDestructor"));
  EXPECT_THAT(log, HasSubstr("Baz"));
}

TEST_F(TracingTest, JitCallReturnedPointerRegistered) {
  // Factory returning T*: the wrapper writes the new pointer at
  // *result; the return-hook registers it as a vN handle so later
  // trace lines that consume it render the name instead of
  // `nullptr /*unknown*/`.
  Cpp::CreateInterpreter({});
  ASSERT_NE(TheTraceInfo, nullptr);

  Cpp::Declare("#include <new>");
  Cpp::Declare("namespace InvRetNS { struct Foo {}; "
               "Foo* makeFoo() { return new Foo(); } }");
  auto MakeFD = Cpp::GetNamed("makeFoo", Cpp::GetScope("InvRetNS"));
  ASSERT_TRUE(MakeFD);
  auto MakeJC = Cpp::MakeFunctionCallable(Cpp::FuncRef{MakeFD.data});
  ASSERT_TRUE(MakeJC.isValid());

  // Sanity: an unrelated pointer remains unregistered.
  EXPECT_TRUE(TheTraceInfo->lookupHandle(reinterpret_cast<void*>(0x1)).empty());

  void* foo = nullptr;
  MakeJC.Invoke(&foo);
  ASSERT_NE(foo, nullptr);

  // After Invoke the return-hook has registered foo; lookupHandle
  // yields its assigned name instead of "".
  EXPECT_FALSE(TheTraceInfo->lookupHandle(foo).empty());

  // makeFoo heap-allocated a trivially-destructible Foo; free it so
  // LeakSanitizer doesn't flag the allocation.
  ::operator delete(foo);
}
#endif

// Verify that log entries include timing annotations.
TEST_F(TracingTest, LogEntriesIncludeTimingAnnotation) {
  VoidFunc();
  auto output = TheTraceInfo->getLastLogEntry();
  EXPECT_THAT(output, HasSubstr("// ["));
  EXPECT_THAT(output, HasSubstr("ns]"));
}

// ---------------------------------------------------------------------------
// Tests: StartTracing / StopTracing region-based tracing
// ---------------------------------------------------------------------------

#ifndef EMSCRIPTEN
static std::string ReadFileToString(const std::string& path) {
  std::ifstream ifs(path);
  return {std::istreambuf_iterator<char>(ifs),
          std::istreambuf_iterator<char>()};
}
// This test reads source files from the build tree via CPPINTEROP_DIR.
TEST_F(TracingTest, StartStopTracingWritesToFile) {
  // StartTracing begins recording; StopTracing writes the file.
  std::string Path = CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  ASSERT_FALSE(Path.empty());

  // Calls within the region are recorded.
  VoidFunc();
  AnnotatedFunction(5);

  CppInterOp::Tracing::StopTracing();

  // The file should exist and contain the traced calls.
  std::string content = ReadFileToString(Path);
  ASSERT_FALSE(content.empty());
  EXPECT_THAT(content, HasSubstr("Cpp::VoidFunc()"));
  EXPECT_THAT(content, HasSubstr("Cpp::AnnotatedFunction("));
  EXPECT_THAT(content, HasSubstr("#include <CppInterOp/Dispatch.h>"));

  llvm::sys::fs::remove(Path);
}

TEST_F(TracingTest, OnlyRegionCallsAreRecorded) {
  // Calls before StartTracing should not appear.
  VoidFunc();

  std::string Path = CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  ASSERT_FALSE(Path.empty());

  AnnotatedFunction(42);

  CppInterOp::Tracing::StopTracing();

  // Calls after StopTracing should not appear either.
  VoidFunc();

  std::string content = ReadFileToString(Path);
  // Should contain only the call within the region.
  EXPECT_THAT(content, HasSubstr("Cpp::AnnotatedFunction(42)"));
  // VoidFunc was called before and after — should NOT be in the file.
  // Count occurrences of VoidFunc in the reproducer body.
  auto bodyStart = content.find("void reproducer()");
  ASSERT_NE(bodyStart, std::string::npos);
  std::string body = content.substr(bodyStart);
  EXPECT_EQ(body.find("VoidFunc"), std::string::npos)
      << "VoidFunc should not appear in the reproducer body:\n"
      << body;

  llvm::sys::fs::remove(Path);
}

TEST_F(TracingTest, StartTracingWithEnvVarNarrowsToRegion) {
  // When CPPINTEROP_LOG=1 is set (tracing already active from SetUp),
  // StartTracing/StopTracing should still narrow to just the region.
  ASSERT_NE(TheTraceInfo, nullptr);

  // This call is before the region.
  VoidFunc();

  std::string Path = CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  ASSERT_FALSE(Path.empty());

  AnnotatedFunction(7);

  CppInterOp::Tracing::StopTracing();

  std::string content = ReadFileToString(Path);
  auto bodyStart = content.find("void reproducer()");
  ASSERT_NE(bodyStart, std::string::npos);
  std::string body = content.substr(bodyStart);
  EXPECT_THAT(body, HasSubstr("Cpp::AnnotatedFunction(7)"));
  EXPECT_EQ(body.find("VoidFunc"), std::string::npos);

  llvm::sys::fs::remove(Path);
}

TEST_F(TracingTest, MultipleStartStopRegions) {
  // Multiple regions should each produce their own file.
  std::string Path1 =
      CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  AnnotatedFunction(1);
  CppInterOp::Tracing::StopTracing();

  std::string Path2 =
      CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  AnnotatedFunction(2);
  CppInterOp::Tracing::StopTracing();

  ASSERT_NE(Path1, Path2);

  std::string content1 = ReadFileToString(Path1);
  std::string content2 = ReadFileToString(Path2);

  EXPECT_THAT(content1, HasSubstr("Cpp::AnnotatedFunction(1)"));
  EXPECT_THAT(content2, HasSubstr("Cpp::AnnotatedFunction(2)"));

  // Each file should only have its own call.
  EXPECT_EQ(content1.find("AnnotatedFunction(2)"), std::string::npos);
  EXPECT_EQ(content2.find("AnnotatedFunction(1)"), std::string::npos);

  llvm::sys::fs::remove(Path1);
  llvm::sys::fs::remove(Path2);
}

// ---------------------------------------------------------------------------
// Tests: StartTracing WriteOnStdErr option
// ---------------------------------------------------------------------------

TEST_F(TracingTest, WriteOnStdErrStreamsEntriesImmediately) {
  testing::internal::CaptureStderr();

  // Default is WriteOnStdErr=true — no file is created.
  std::string Path = CppInterOp::Tracing::StartTracing();
  EXPECT_TRUE(Path.empty());

  AnnotatedFunction(99);

  // Capture stderr *before* StopTracing — the entry must already be there.
  std::string Mid = testing::internal::GetCapturedStderr();
  EXPECT_THAT(Mid, HasSubstr("Cpp::AnnotatedFunction(99)"))
      << "Entry should appear on stderr immediately, not after StopTracing";

  // Restart capture for the rest.
  testing::internal::CaptureStderr();
  VoidFunc();
  CppInterOp::Tracing::StopTracing("test-version");
  std::string Rest = testing::internal::GetCapturedStderr();
  EXPECT_THAT(Rest, HasSubstr("Cpp::VoidFunc()"));
}

TEST_F(TracingTest, WriteToFileWhenStdErrDisabled) {
  std::string Path = CppInterOp::Tracing::StartTracing(/*WriteOnStdErr=*/false);
  ASSERT_FALSE(Path.empty());

  AnnotatedFunction(77);

  CppInterOp::Tracing::StopTracing();

  // The file should contain the full reproducer.
  std::string FileContent = ReadFileToString(Path);
  EXPECT_THAT(FileContent, HasSubstr("#include <CppInterOp/Dispatch.h>"));
  EXPECT_THAT(FileContent, HasSubstr("Cpp::AnnotatedFunction(77)"));

  llvm::sys::fs::remove(Path);
}

// ---------------------------------------------------------------------------
// Tests: all CPPINTEROP_API functions must have INTEROP_TRACE
// ---------------------------------------------------------------------------

TEST(TracingCoverageTest, AllPublicAPIsAreTraced) {
  // 1. Parse the header and generated declarations to extract all
  //    CPPINTEROP_API function names.
  std::string Header =
      ReadFileToString(CPPINTEROP_DIR "/include/CppInterOp/CppInterOp.h");
  ASSERT_FALSE(Header.empty()) << "Could not read CppInterOp.h";
  // Function declarations are generated into CppInterOpDecl.inc.
  std::string DeclInc = ReadFileToString(
      CPPINTEROP_BINARY_DIR "/include/CppInterOp/CppInterOpDecl.inc");
  if (DeclInc.empty())
    DeclInc = ReadFileToString(CPPINTEROP_DIR
                               "/include/CppInterOp/CppInterOpDecl.inc");
  Header += DeclInc;

  std::regex ApiRe(R"(CPPINTEROP_API\s+[\w:<>\s\*&]+?\s+(\w+)\s*\()");
  std::set<std::string> ApiNames;
  for (std::sregex_iterator it(Header.begin(), Header.end(), ApiRe), end;
       it != end; ++it) {
    ApiNames.insert((*it)[1].str());
  }

  // Exclude JitCall member functions declared with CPPINTEROP_API — they
  // are class members, not free API functions we annotate.
  ApiNames.erase("AreArgumentsValid");
  ApiNames.erase("ReportInvokeStart");
  ApiNames.erase("ReportInvokeEnd");
  // Exclude the JitCall trace-hook impls: they are reached only from
  // INTEROP_TRACE-instrumented inline code; tracing them recursively
  // would just nest entries for no benefit.
  ApiNames.erase("CppInterOpTraceJitCallInvokeImpl");
  ApiNames.erase("CppInterOpTraceJitCallInvokeDestructorImpl");
  ApiNames.erase("CppInterOpTraceJitCallInvokeReturnImpl");

  ASSERT_GT(ApiNames.size(), 100u)
      << "Suspiciously few API functions found — regex may be broken";

  // 2. For each API name, find its definition in the implementation and
  //    verify INTEROP_TRACE appears shortly after the opening brace.
  std::string Impl =
      ReadFileToString(CPPINTEROP_DIR "/lib/CppInterOp/CppInterOp.cpp");
  ASSERT_FALSE(Impl.empty()) << "Could not read CppInterOp.cpp";

  std::vector<std::string> Missing;
  for (const auto& Name : ApiNames) {
    std::string Pattern = Name + "(";
    size_t Pos = 0;
    bool Found = false;
    while ((Pos = Impl.find(Pattern, Pos)) != std::string::npos) {
      // Find the opening brace of the function body. Wide-signature window
      // (500) accommodates 8-parameter APIs like MakeVTableOverlay laid out
      // one-arg-per-line LLVM-style.
      size_t BracePos = Impl.find('{', Pos);
      if (BracePos == std::string::npos || BracePos - Pos > 500) {
        Pos += Pattern.size();
        continue;
      }

      // Check the next ~200 chars after '{' for INTEROP_TRACE.
      std::string Body =
          Impl.substr(BracePos, std::min<size_t>(200, Impl.size() - BracePos));
      if (Body.find("INTEROP_TRACE") != std::string::npos) {
        Found = true;
        break;
      }
      Pos += Pattern.size();
    }
    if (!Found)
      Missing.push_back(Name);
  }

  if (!Missing.empty()) {
    std::string Msg = "Public API functions missing INTEROP_TRACE:\n";
    for (const auto& Name : Missing)
      Msg += "  - " + Name + "\n";
    Msg += "Add INTEROP_TRACE() to each function listed above.\n";
    FAIL() << Msg;
  }
}

#endif // !EMSCRIPTEN
