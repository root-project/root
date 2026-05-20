#include "Template.h"
extern "C" int printf(const char*,...);
namespace ANamespace {

template <>
class Template<int, 0, TagClassWithoutDefinition> {
public:
   Template() { printf("Int0 ANamespace::Template object created.\n"); }
   typedef int TheType_t;
   enum { kTheValue = 42 };
   void intInstance() { printf("ANS::intInstance0()\n"); }
};

template <>
class Template<int, 1, TagClassWithoutDefinition> {
public:
   Template() { printf("Int1 ANamespace::Template object created.\n"); }
   typedef int TheType_t;
   enum { kTheValue = 43 };
   void intInstance() { printf("ANS::intInstance0()\n"); }
};

}
