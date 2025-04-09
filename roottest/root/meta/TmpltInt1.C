#include "Template.h"
extern "C" int printf(const char*,...);

template <>
class Template<int, 1, TagClassWithoutDefinition> {
public:
   Template() { printf("Int1 template instance object created.\n"); }
   typedef int TheType_t;
   enum { kTheValue = 43 };
   void intInstance() { printf("intInstance1()\n"); }
};

