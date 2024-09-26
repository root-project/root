#include <iostream>

struct Foo 
{
  Foo() {}
  ~Foo() {}
  void print() { std::cout << "Hello, I'm Foo" << std::endl; }
};
