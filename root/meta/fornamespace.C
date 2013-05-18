namespace MySpace {
   class MyClass {
   public:
      int a;
      MyClass() : a(0) {}
   };
   int funcInNS() { return 42; }
}

void fornamespace () {};
int globalFunc() { return 43; }
int globalFuncInline();
inline int globalFuncInline() { return 44; }
