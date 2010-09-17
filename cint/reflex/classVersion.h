class WithClassVersion {
 public:
   int hasAClassVersion;
};

#ifndef __CINT__

#include <vector>
#include <string>

template <class T> class MyTemp {
   T value;
};

template class std::vector<std::string>;
template class std::vector<WithClassVersion>;
template class MyTemp<std::vector<std::string> >;

#endif
