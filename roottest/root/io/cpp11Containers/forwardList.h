#include "TH1F.h"
#include <forward_list>
#include <list>
#include <vector>
#include <deque>
#include <complex>

#ifndef __COMPLEX__INSTANCES__
#define __COMPLEX__INSTANCES__
template<class T> class instantiator{
  std::forward_list<std::complex<T>> a1;
  std::list<std::complex<T>> a2;
  std::vector<std::complex<T>> a3;
  std::deque<std::complex<T>> a4;
};
instantiator<float> i1;
instantiator<double> i2;
#endif
