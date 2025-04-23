namespace space {
   template <class T> class X {};
}

space::X<int> xi;

using namespace space;

X<double> xd;

class MyClass {
public:
   X<float> xd;

};

namespace other {
   template <class T> class Y {};
}
namespace application {
   using namespace other;
   class MyClass {
   public:
      Y<float> xd;
   };
}
