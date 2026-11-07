#ifndef ROOTTEST_NESTEDAUTOPARSE_H
#define ROOTTEST_NESTEDAUTOPARSE_H

namespace NestedAutoParse {
namespace Inner {

// Declaring a class template while parsing the payload calls
// Sema::EnterTemplatedContext, which walks Sema's function scopes.
template <int N>
struct Tmpl {
   static constexpr int value = N;
};

struct Payload {
   int fValue = 42;
   int get() const { return fValue + Tmpl<0>::value; }
};

} // namespace Inner
} // namespace NestedAutoParse

#endif
