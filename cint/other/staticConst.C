#include <iostream>
using namespace std;

class X
{
 public:
  static const int i;
  static const int a[];

  void print();
};

const int X::a[]={0,1,2,3,4,5,6,7,8,9};

void X::print()
{
    std::cout << "Print static const int member:\n"
              << X::a[0] << "\t"
              << X::a[1] << "\t"
              << std::endl;
}

void staticConst() {
   X x;
   x.print();
}
