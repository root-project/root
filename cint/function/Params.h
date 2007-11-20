class Base {
 public:
   Base(){};
   virtual ~Base(){};
   virtual int FunctionX(int b=0, int c=5) {printf("Base::FunctionX(int b=%d, int c=%d)\n", b, c);}
};

class Derived: public Base {
 public:
   Derived(){};
   virtual ~Derived(){};
   virtual int FunctionX(int b=1, int c=6) {printf("Derived::FunctionX(int b=%d, int c=%d)\n", b, c);}
};
