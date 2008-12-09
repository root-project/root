class ABC {
public:
   ABC(int i = -1): abc(i) {}
   virtual ~ABC() {}
   virtual int pv() = 0;
   int abc;
};

class Derived: public ABC {
public:
   virtual ~Derived() {}
   Derived(int i = -2): ABC(i+1), derived(i) {}

   int pv() { return abc + derived; }
   int derived;
};

class Holder {
public:
   Holder(): fABC(0) {}
   ~Holder() { delete fABC; }

   void Set(int i) { delete fABC; fABC = new Derived(i); }

   ABC* fABC;
};

