//------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//------------------------------------------------------------------------------

// RUN: cat %s | %cling 2>&1 | FileCheck %s

#include <memory>
#include <string>

extern "C" int printf(const char* fmt, ...);
.storeState "preUnload"
class MyClass{
private:
  double member;
public:
  MyClass() : member(42){}
  static int get12(){ return 12; }
  double getMember(){ return member; }
}; MyClass m; m.getMember(); MyClass::get12();
.undo
.compareState "preUnload"
//CHECK-NOT: Differences
float MyClass = 1.1
//CHECK: (float) 1.10000f

template <typename T>
struct MyStruct { T f(T x) { return x; } };
MyStruct<float> obj;
obj.f(42.0)
//CHECK: (float) 42.0000f
.undo
obj.f(42.0)
//CHECK: (float) 42.0000f

auto p = std::make_unique<std::string>("string");
(unsigned long)p.size() // expected-error{{no member named 'size' in 'std::unique_ptr<std::basic_string<char>>'; did you mean to use '->' instead of '.'?}}
(unsigned long)p->size()
//CHECK: (unsigned long) 6

// Test ROOT issue #23439: failure in an expression returning a template specialization
// (e.g. std::unique_ptr) should not corrupt the implicit template instantiation.
struct Target23439 { int val = 42; };
std::unique_ptr<Target23439> getTarget23439(int x) { return std::make_unique<Target23439>(); }
auto f_err = getTarget23439(); // expected-error{{no matching function for call to 'getTarget23439'}}
auto f_ok = getTarget23439(1);
f_ok->val
//CHECK: (int) 42

.q

