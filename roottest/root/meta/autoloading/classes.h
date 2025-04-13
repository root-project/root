#ifndef AA
#define AA

class myClass{};

typedef myClass myTypedef;

template <class T, int I> class myTemplateClass{};

typedef myTemplateClass<float***, 4> foo;

foo aa;

#endif