int gCountingCalls = 0;

class Base {
 public:
   Base(const char* mode = ""): fMode(mode){};
   virtual ~Base(){};
   virtual void FunctionX(int b=0, int c=5) {printf("%s: Base::FunctionX(int b=%d, int c=%d)\n", fMode, b, c);}

   void CINTCannotHandleDefaultArgThatsNotYetParsed(int a = GetCountingCalls()) {};
   static int GetCountingCalls() {return ++fgCountingCalls;}
   virtual void FunctionY(int arg0 = ++gCountingCalls, float arg1 = Base::GetCountingCalls() ) {
      printf("%s: Base::FunctionY(int arg0=%d, float arg1=%d.)\n", fMode, arg0, (int)arg1);}


   const char* fMode;
   static int fgCountingCalls;
};

int Base::fgCountingCalls = 0;

class Derived: public Base {
 public:
   Derived(const char* mode = ""): Base(mode){};
   virtual ~Derived(){};
   virtual void FunctionX(int b=1, int c=6) {printf("%s: Derived::FunctionX(int b=%d, int c=%d)\n", fMode, b, c);}
   virtual void FunctionY(int arg0 = ++gCountingCalls, float arg1 = Base::GetCountingCalls() ) {
      printf("%s: Derived::FunctionY(int arg0=%d, float arg1=%d.)\n", fMode, arg0, (int)arg1);}
};
