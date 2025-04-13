#include <iostream>
#include <TString.h>

class MyClass
{
   public:
      template<typename type>
      type GetScalar(TString alias)
      {
         std::cout << "Hello from GetScalar(" << alias << ")" << std::endl;
              return type(0);
                }
};

MyClass& GetMyClassReference();
