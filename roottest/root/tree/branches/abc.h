class BaseABC {
public:
   BaseABC(int i = -1): abc(i) {}
   virtual ~BaseABC() {}
   virtual int pv() = 0;
   int abc;
};

class Derived: public BaseABC {
public:
   ~Derived() override {}
   Derived(int i = -2): BaseABC(i+1), derived(i) {}

   int pv() override { return abc + derived; }
   int derived;
};

class Holder {
public:
   Holder(): fABC(nullptr) {}
   ~Holder() { delete fABC; }

   void Set(int i) { delete fABC; fABC = new Derived(i); }

   BaseABC* fABC;
};

