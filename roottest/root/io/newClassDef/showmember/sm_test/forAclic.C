class MyClass {
protected:
   int a;
   float b;
public:
   MyClass(int aa,float bb):  a(aa),b(bb) {};
   int GetA() { return a; };
   float GetB() { return b; };
   virtual void SetB(double bb) { b = bb; };
};


