class A { public: A() {}; virtual ~A() {}; int a; };
class B { public: B() {}; virtual ~B() {}; int a; };
class C : public A { public: virtual ~C() {}; int a; };
class D : public A, public B {public: virtual ~D() {}; int a; };

#include <stdio.h>

#include <TClass.h>
#include <TROOT.h>

class ReadCast {
public:
   char* fStartAdd;
   TClass* fActualClass;
   
   ReadCast(void* startAdd, TClass* clActual) : fStartAdd((char*)startAdd), fActualClass(clActual) {};
   
   template <class T> operator T*() {
      TClass *tcl = gROOT->GetClass(typeid(T));
      Int_t offset = fActualClass->GetBaseClassOffset(tcl);
      if (offset<0) return 0;
      return (T*)fStartAdd+offset;
   }
};

class WriteCast {
public:
   char *fStartAdd;
   TClass *fActualClass;
   int fOffset;

   WriteCast() {};
   template <class T> WriteCast(T* add) {
      TClass *tcl = gROOT->GetClass(typeid(T));
      fActualClass = tcl->GetActualClass(add);
      Int_t offset = fActualClass->GetBaseClassOffset(tcl);
      fOffset = offset;
      if (offset<0) fprintf(stderr,"problem in WriteCast\n");
      fStartAdd = ( (char*)add ) - offset;
   }
};

void castfeat() {
   A * a = new A;
   B * b = new B;
   C * c = new C;
   D * d = new D;

   A * ac = c;
   A * ad = d;
   B * bd = d;
   
   WriteCast * w;
   w = new WriteCast( a );
#if 1
   fprintf(stderr,"WriteCast of a (%p) gives %p and %s\n", a, w->fStartAdd, w->fActualClass->GetName());
   w = new WriteCast( b );
   fprintf(stderr,"WriteCast of b (%p) gives %p and %s\n", b, w->fStartAdd, w->fActualClass->GetName());

   w = new WriteCast( c );
   fprintf(stderr,"WriteCast of c (%p) gives %p and %s\n", c, w->fStartAdd, w->fActualClass->GetName());

   w = new WriteCast( d );
   fprintf(stderr,"WriteCast of d (%p) gives %p and %s\n", d, w->fStartAdd, w->fActualClass->GetName());

   w = new WriteCast( ac );
   fprintf(stderr,"WriteCast of ac (%p) gives %p and %s\n", ac, w->fStartAdd, w->fActualClass->GetName());

   w = new WriteCast( ad );
   fprintf(stderr,"WriteCast of ad (%p) gives %p and %s\n", ad, w->fStartAdd, w->fActualClass->GetName());

   w = new WriteCast( bd );
   fprintf(stderr,"WriteCast of bd (%p) gives %p and %s %d\n", bd, w->fStartAdd, w->fActualClass->GetName(), w->fOffset);
#endif
}
