// -- file const.C
#include <iostream>

class MyClass4 {};

const MyClass4* g(const MyClass4* arg ) {
  std::cout << "with const MyClass4*" << std::endl;
  return arg;
}

MyClass4* g(MyClass4* arg  ) {
  std::cout << "with MyClass4*" << std::endl;
  return arg;
}

void runConst() {
  g( (const MyClass4*)0 );
  g( (MyClass4*)0 );
}
/*
>It seems (and you might already know :) ) that the function matching
>algorithm does not use the constness of the argument.  The following
>code acts differenty in CINT or C++:
>
>root [0] .L const.C 
>root [1] run()
>with const MyClass4*
>with const MyClass4*
>
>root [2] .L const.C+
>....
>root [3] run()
>with const MyClass4*
>with MyClass4*
*/
