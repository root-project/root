#include <iostream>
#include <TVectorD.h>

// ROOT-7739
int ops() {
  int ret = 0;
  double a = 2;
  TVectorD b(2);
  if (!std::is_same<decltype(a*b), TVectorD>::value) {
    std::cerr << "double * TVectorD is not TVectorD\n";
    ++ret;
  }
  if (!std::is_same<decltype(b*a), TVectorD>::value) {
    std::cerr << "TVectorD * double is not TVectorD\n";
    ++ret;
  }
  return ret;
}
