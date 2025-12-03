#ifndef INCLUDE_TEMPLATE_H
#define INCLUDE_TEMPLATE_H

extern "C" int printf(const char*,...);
class TagClassWithoutDefinition;

template <typename T, int I, typename TAG, typename TDEF = T, int IDEF = I>
class Template {
public:
   Template() { printf("Generic Template object created.\n"); }
   typedef T TheType_t;
   enum { kTheValue = I };
};

namespace ANamespace {
   template <typename T, int I, typename TAG, typename TDEF = T, int IDEF = I>
   class Template {
   public:
      Template() { printf("Generic ANamespace::Template object created.\n"); }
      typedef T TheType_t;
      enum { kTheValue = I };
   };
}

#endif // INCLUDE_TEMPLATE_H
