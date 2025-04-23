class A{};
class B{};
class C{};

class myTypedefFake{};

typedef  A myTypedef;
typedef  C myTypedef2;

typedef  B firstOrderTypedef;

typedef firstOrderTypedef secondOrderTypedef;

