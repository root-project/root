template <class T> class MyClass {};

MyClass<int> a;
MyClass<const double*> b;

namespace space {
template <class T> class Nested {};

Nested<const double*> c;
}

void t() {
MyClass<int> a;
MyClass<const double*> b;
(MyClass<int               >    * ) 0;
}

