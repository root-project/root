class A{};
class B{ public: virtual int getid() {return 0;}};
class C : public B {};
class D : public A {};

#include <iostream.h>
#include <map>
#include <string>
#include <typeinfo>
using namespace std;


void func(A *) { std::cerr << "A\n"; }
void func(B *) { std::cerr << "B\n"; }

typedef void (*funca)(A*);
typedef void (*funcb)(B*);

template <class T> void print_real_classname(const T *obj) {
  cerr << typeid(*obj).name() << endl;
}
int main() {

  A *a = 0;
  A *a2 = new D;
  B *b = new B;
  B *b2 = new C;
  cerr << "Expect A: "; func(a);
  cerr << "Expect B: "; func(b);

  funca pfunc;
  pfunc = &func;
  cerr << "Expect A: "; pfunc(a);

  funcb pfuncb;
  pfuncb = &func;
  cerr << "Expect B: "; pfuncb(b);

  string namea("A");
  string nameb("B");
  string namec("C");
  string named("D");

  map<const type_info*,string> m;
  m[&typeid(A)] = namea;
  m[&typeid(B)] = nameb;
  m[&typeid(C)] = namec;
  m[&typeid(D)] = named;

  cerr << "Expect A: "; 
  cerr << m[&typeid(*a)] << endl;  // A is not virtual, do not need a real object.
  cerr << "Expect A: "; 
  cerr << m[&typeid(*a2)] << endl;  // A is not virtual, real object ignored!
  cerr << "Expect B: "; 
  cerr << m[&typeid(*b)] << endl;
  cerr << "Expect C: "; 
  cerr << m[&typeid(*b2)] << endl;
 

  void * p = b2;
  p = p;
  cerr << "Expect B*: "; 
  cerr << typeid(b2).name() << endl;
  cerr << "Expect C: "; 
  cerr << typeid(*b2).name() << endl;
  cerr << "Expect void*: "; 
  cerr << typeid(p).name() << endl;
#ifdef __KCC
  cerr << "KCC does not support typeid(*(void*)p)" << endl;
#else
  cerr << "Expect void: "; 
  cerr << typeid(*p).name() << endl;
#endif

  cerr << "Expect A: "; print_real_classname(a);
  cerr << "Expect A: "; print_real_classname(a2);
  cerr << "Expect B: "; print_real_classname(b);
  cerr << "Expect C: "; print_real_classname(b2);
}
