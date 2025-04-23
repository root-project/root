#include "Template.h"

template <>
class Template<float, 0, TagClassWithoutDefinition> {
public:
   Template() { printf("Float template instance object created.\n"); }
   typedef float TheType_t;
   enum { kTheValue = 17 };
   void floatInstance() { printf("floatInstance()\n"); }
};

