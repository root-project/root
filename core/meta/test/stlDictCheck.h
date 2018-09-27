#ifndef _DICT_CHECK_TEST_
#define _DICT_CHECK_TEST_

#include <memory>

struct A {
};
struct B {
   std::unique_ptr<A> fA;
};

#endif
