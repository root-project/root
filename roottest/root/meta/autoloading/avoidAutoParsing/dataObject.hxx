#pragma once

#ifdef TEST_RUNTIME
#error "This header should not be loaded at runtime"
#endif

#include <vector>

namespace testing 
{
   struct InnerContent {};
   template <typename T>
   struct FindUsingAdvance {};

   using Collection = std::vector<int>;

   template <typename T>
   class UserClass 
   {
   public:
      int fValue;
      Collection fCollValue;
   };
}

