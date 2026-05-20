#ifndef MyTemplateClass_h
#define MyTemplateClass_h

template <typename T>
class MyTemplateClass
{
 public:
  T var;

  MyTemplateClass()
  {
  }

  MyTemplateClass(T i)
  {
    var = i;
  }
};

#endif

