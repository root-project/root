namespace N {
class NoBase {
   int a;
   NoBase* b;
   float f;
};
class Base {
   int a;
   Base* b;
   float f;
};
}

class Derived0: N::NoBase {
   N::Base* b; // look, a second one!
};

class Derived1: N::Base {
   N::Base* b; // look, a second one!
};

class Derived2: N::Base {
   Derived1* d;
};
