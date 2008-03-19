// classinfo.h
#ifndef G__TEST_CLASSINFO_H
#define G__TEST_CLASSINFO_H

#include <stdio.h>

class A
{
public: // Types
   enum ETags {
      val1,
      val2,
      val3
   };

public: // Functions
   void with_enum(ETags tag)
   {
      printf("tag: %d\n", tag);
   }
   void with_int(int val)
   {
      printf("val: %d\n", val);
   }
};

#endif // G__TEST_CLASSINFO_H
