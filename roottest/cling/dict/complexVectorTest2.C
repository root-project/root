#include <vector>
#include "MyTestClass.h"
#include "MyTemplateTestClass.h"

void complexVectorTest2()
{
  MyTemplateClass< MyClass > tmp;
  std::vector< MyTemplateClass<MyClass>  > vec;

  vec.push_back( tmp );
  std::cout << vec.back().var.var  << std::endl;
}
