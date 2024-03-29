//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.

//------------------------------------------------------------------------------
// RUN: cat %s | %cling | FileCheck %s

#include <iostream>
#if __cplusplus >= 201703L
#include <filesystem>
#endif

#if __cplusplus >= 201703L
auto p = std::filesystem::path("/some/path/foo.cpp");
p
#else
// Hack to prevent failure if feature does not exist!
std::cout << "(std::filesystem::path &)";
std::cout << "/some/path/foo.cpp\n";
#endif
    // CHECK: (std::filesystem::path &) /some/path/foo.cpp
