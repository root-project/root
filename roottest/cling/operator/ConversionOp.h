#ifndef ConversionOp_h
#define ConversionOp_h
/* See ConversionOp.C for details. */

template <typename T>
class A {
 public:
   A(){}
   A(const A& a): fT(a.fT) {printf("A::A(const A&)");}
   A(const T& t): fT(t) {}
   operator T*() {printf("A::op T*()\n"); return &fT;}
   operator T() {printf("A::op T()\n"); return fT;}
   operator A() {printf("A::op A()\n"); return *this;}
   operator A*() {printf("A::op A*()\n"); return this;}
   bool operator <(const T&) const { printf("A::op <(T)\n"); return false; }
   bool operator <(const A&) { printf("A::op <(A)\n"); return true; }
 private:
   T fT;
};

class B {
 public:
   B() {}
   B(const B&) {}
};

namespace N {
   class B {
   public:
      B(const ::B& b): fB(b) {}
      // Make sure that CINT doesn't confuse
      // N::B::op ::B() with
      // N::B::op N::B() when writing the dictionary
      operator ::B() const {return fB;}
   private:
      ::B fB;
   };
}

class C {
 public:
   C(): fA(fB) {}

   operator A<C>() {printf("C::op A<C>()\n"); return A<C>(*this);}
   operator A<B>() {printf("C::op A<B>()\n"); return fA;}
   operator B() {printf("C::op B()\n"); return fB;}
   operator N::B() {printf("C::op N::B()\n"); return N::B(fB);}

   operator A<C>*() const {printf("C::op A<C>*()\n"); return new A<C>(*this);}
   operator const A<B>*() const {printf("C::op A<B>*()\n"); return &fA;}
   operator B*() const {printf("C::op B*()\n"); return const_cast<B*>(&fB);}

   operator const A<C>&() const {static A<C> a; a=A<C>(*this); printf("C::op A<C>&()\n"); return a;}
   operator A<B>&() {printf("C::op A<B>&()\n"); return fA;}
   operator const B&() const {printf("C::op B&()\n"); return fB;}

 private:
   A<B> fA;
   B    fB;
};

class D {
 public:
   A<B> operator+(A<B> a) {printf("D::op +(A<B>)\n"); return a;}
   A<B>* operator-(A<B> a) {printf("D::op -(A<B>)\n"); return new A<B>(a);}
   template <typename T>
   A<T>* operator*(A<T> &a) {printf("D::op *(A<T>)\n"); return &a;}
   template <typename T>
   A<T>& operator/(A<T> &a) {printf("D::op /(A<T>)\n"); return a;}
};

#ifdef __MAKECINT__
#pragma link C++ class A<B>+;
#pragma link C++ class A<float>+;
#pragma link C++ class A<int>+;
#pragma link C++ class B+;
#pragma link C++ class C+;
#pragma link C++ class D+;

#ifdef CINTFAILURE
#pragma link C++ function D::operator*(A<int> &a);
#pragma link C++ function D::operator/(A<float> &a);
#endif
#endif

#endif
