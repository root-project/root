class A{};
class B{ 
public: 
  virtual ~B(){}
  virtual int getid() {return 0;}
};
class C : public B {};
class D : public A {};

#include "Riostream.h"
#include <map>
#include <string>
#include <typeinfo>
#include "TClass.h"

namespace std {}
using namespace std;

#if CONST_STRING
typedef map<const string,TClass*> IdMap0_t;
#else
typedef map<string,TClass*> IdMap0_t;
#endif

IdMap0_t *fIdMap       = new IdMap0_t;

void testAddClass(const type_info *info, TClass *cl)
{
   // Add a class to the list and map of classes.

   if (!cl) return;
   if (info) { (*fIdMap)[info->name()] = cl; }
}

TClass* testGetClass(const type_info &info) {
  
  TClass *cl = fIdMap->find(info.name())->second;
  return cl;

}

void func(A *) { cerr << "A\n"; }
void func(B *) { cerr << "B\n"; }

typedef void (*funca)(A*);
typedef void (*funcb)(B*);

template <class T> void print_real_classname(const T *obj) {
  cerr << typeid(*obj).name() << endl;
}

template <class T> bool print_root_classname(const T *obj, const char* expect) {
  TClass *cl = testGetClass(typeid(*obj));
  if (!cl || strcmp(cl->GetName(),expect)!=0 ) {
    cerr << "ERROR: in retrieving TClass";
    if (cl) { cerr << " found " << cl->GetName() << endl; }
    else { cerr << " NOT found!" << endl; };
    return 0;
  }
  cerr << cl->GetName() << endl;
  return 1;
}


bool typeidtest() {
  A *a = 0;
  A *a2 = new D;
  B *b = new B;
  B *b2 = new C;
  cerr << "Expect A: "; func(a);
  cerr << "Expect B: "; func(b);

  testAddClass(& ( typeid(A) ), TClass::GetClass("A") );
  testAddClass(& ( typeid(B) ), TClass::GetClass("B") );
  testAddClass(& ( typeid(C) ), TClass::GetClass("C") );
  testAddClass(& ( typeid(D) ), TClass::GetClass("D") );

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
  cerr << m[&typeid(*a)].c_str() << endl;  // A is not virtual, do not need a real object.
  cerr << "Expect A: "; 
  cerr << m[&typeid(*a2)].c_str() << endl;  // A is not virtual, real object ignored!
  cerr << "Expect B: "; 
  cerr << m[&typeid(*b)].c_str() << endl;
  cerr << "Expect C: "; 
  cerr << m[&typeid(*b2)].c_str() << endl;
 

  void * p = b2;
  // p = p;
  cerr << "Expect C++'s B*: "; 
  cerr << typeid(b2).name() << endl;
  cerr << "Expect C++'s C: "; 
  cerr << typeid(*b2).name() << endl;
  cerr << "Expect C++'s void*: "; 
  cerr << typeid(p).name() << endl;

  cerr << "Expect " << typeid(A).name() << " : "; print_real_classname(a);
  cerr << "Expect " << typeid(A).name() << " : "; print_real_classname(a2);
  cerr << "Expect " << typeid(B).name() << " : "; print_real_classname(b);
  cerr << "Expect " << typeid(C).name() << " : "; print_real_classname(b2);

  bool result = true;
  cerr << "Expect A: "; 
  result &= print_root_classname(a,"A");
  cerr << "Expect A: ";  
  result &= print_root_classname(a2,"A");
  cerr << "Expect B: ";  
  result &= print_root_classname(b,"B");
  cerr << "Expect C: ";  
  result &= print_root_classname(b2,"C");
  
  return result;
}
