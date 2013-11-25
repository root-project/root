class A{};
class B{};

class myTypedefFake{};

typedef  A myTypedef;

typedef  B firstOrderTypedef;

typedef firstOrderTypedef secondOrderTypedef;

#ifdef __GXX_EXPERIMENTAL_CXX0X__
class C{};
using cpp11Alias = C;
#endif
