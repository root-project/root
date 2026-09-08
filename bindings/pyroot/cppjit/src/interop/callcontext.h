#ifndef CPPJIT_INTEROP_CALLCONTEXT_H
#define CPPJIT_INTEROP_CALLCONTEXT_H

// Standard
#include <cstddef>
#include <cstdint>

// convention to pass flag for direct calls (similar to Python's vector calls)
#define DIRECT_CALL ((size_t)1 << (8 * sizeof(size_t) - 1))

namespace cppjit::cpyrt {

// small number that allows use of stack for argument passing
const int SMALL_ARGS_N = 8;

// The shipped cpyrt/API.h carries an identical Parameter for JIT-side
// code, which cannot see this in-tree header; the shared CPYRT_PARAMETER
// guard keeps one definition per TU. Keep both copies identical.
#ifndef CPYRT_PARAMETER
#define CPYRT_PARAMETER
// general place holder for function parameters
struct Parameter {
  union Value {
    bool fBool;
    int8_t fInt8;
    uint8_t fUInt8;
    short fShort;
    unsigned short fUShort;
    int fInt;
    unsigned int fUInt;
    long fLong;
    intptr_t fIntPtr;
    unsigned long fULong;
    long long fLLong;
    unsigned long long fULLong;
    int64_t fInt64;
    uint64_t fUInt64;
    float fFloat;
    double fDouble;
    long double fLDouble;
    void* fVoidp;
  } fValue;
  void* fRef;
  char fTypeCode;
};
#endif // CPYRT_PARAMETER

} // namespace cppjit::cpyrt

#endif // !CPPJIT_INTEROP_CALLCONTEXT_H
