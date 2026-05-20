#include <vector>
#include "MyTestClass.h"

void simpleVectorTest()
{
  std::vector<MyClass> mv;
  
  mv.push_back( MyClass() );
  std::cout << mv.back().var << std::endl;
  
  mv.pop_back();  
  if( mv.empty() )std::cout << 0 << std::endl;
  else std::cout << 1 << std::endl;
}
