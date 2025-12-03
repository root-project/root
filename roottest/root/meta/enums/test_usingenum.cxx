namespace A { enum E { kOne }; }
namespace B { using A::E; }
namespace C { using namespace A; }

#include <iostream>

int test_usingenum() {
  auto e = TEnum::GetEnum("A::E");
  if (!e) {
    std::cerr << "Can not find A::E declaration\n";
    return 1;
  }
  auto be = TEnum::GetEnum("B::E");
  if (!be) {
    std::cerr << "Can not find B::E\n";
    return 2;
  }
  if (e != be) {
    std::cerr << "The TEnum found for A::E and B::E is different\n";
    return 3;
  }
#if CLING_FIXED_15407
  // This is failing due to https://github.com/root-project/root/issues/15407
  auto ce = TEnum::GetEnum("C::E");
  if (!ce) {
    std::cerr << "Can not find C::E\n";
    return 4;
     
  }
  if (e != ce) {
    std::cerr << "The TEnum found for A::E and C::E is different\n";
    return 5;
  }
#endif
  return 0;
};
