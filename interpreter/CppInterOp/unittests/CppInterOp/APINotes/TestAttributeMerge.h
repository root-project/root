#ifndef UNITTESTS_CPPINTEROP_TESTATTRIBUTEMERGE_H
#define UNITTESTS_CPPINTEROP_TESTATTRIBUTEMERGE_H

[[clang::annotate("cppAllocMalloc")]] void* mergeFunc();

#pragma clang attribute push([[clang::annotate("cppAllocNew")]],               \
                             apply_to = function)
int* overloadFunc();
int* overloadFunc(int n);
#pragma clang attribute pop

[[clang::annotate("cppAllocMalloc")]] void* func71_helper();

#endif
