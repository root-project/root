#include <string>

template <class T>
struct C {
  C() {}
  T& value() {return data;}
  T data;
};

C<std::string[2]> instance() {
  C<std::string[2]> c;
  return c;
}

C<std::string> instance2() {
  C<std::string> c;
  return c;
}

