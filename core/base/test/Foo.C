#include <iostream>

class Foo {
public:
   Foo() { std::cout << "Foo" << std::endl; }
   ~Foo() { std::cout << "~Foo" << std::endl; }
   void f() { std::cout << "f" << std::endl; }
};
