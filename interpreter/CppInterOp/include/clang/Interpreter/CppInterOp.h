#ifndef CLANG_CPPINTEROP_H
#define CLANG_CPPINTEROP_H

#include "CppInterOp/CppInterOp.h"
#if defined(_MSC_VER)
#pragma message(                                                               \
    "#include <clang/Interpreter/CppInterOp.h> is deprecated; use #include <CppInterOp/CppInterOp.h>")
#else
#warning                                                                       \
    "#include <clang/Interpreter/CppInterOp.h> is deprecated; use #include <CppInterOp/CppInterOp.h>"
#endif

#endif // CLANG_CPPINTEROP_H
