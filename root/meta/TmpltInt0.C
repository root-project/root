#include "Template.h"
extern "C" int printf(const char*,...);

template <>
class Template<int, 0, TagClassWithoutDefinition> {
public:
   Template() { printf("Int0 template instance object created.\n"); }
   typedef int TheType_t;
   enum { kTheValue = 42 };
   void intInstance() { printf("intInstance0()\n"); }
};

