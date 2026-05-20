#include <vector>
#include "MyTemplateTestClass.h"

void complexVectorTest()
{
  MyTemplateClass<int> mtc(2);
  std::vector< MyTemplateClass<int>  > mtcv;

  mtcv.push_back( mtc );
  std::cout << mtcv.back().var << std::endl;
}
