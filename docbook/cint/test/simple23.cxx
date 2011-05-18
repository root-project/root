class A {
public:
   int a;
   A() : a(1) {};
   void disp_a() {
      printf("a=%d\n",a);
   }
};

class B {
public:
   int b;
   B() : b(2) {};
   void disp_b() {
      printf("b=%d\n",b);
   }
};

class C : public B, public A {
public:
   int c;
   C() : c(3) {};
   void disp_c() {
      disp_b();
      disp_a();
      printf("c=%d\n",c);
   }
};

int main() {
   C obj;
   obj.disp_c();
}

