#include "MyTemplateTestClass.h"

void templateClassTest()
{
  MyTemplateClass<int> mtc(1);
  std::cout << mtc.var << std::endl;
}
