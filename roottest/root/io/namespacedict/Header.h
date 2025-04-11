namespace A {
  class Class1 {
  public:
     ClassDef(Class1,1) //
  };
};

namespace A {
   template <class T>
   class Class2 {
   public:
      Class2();
      virtual ~Class2();
   public:
#ifdef WIN32
      ClassDef(Class2,1) //
#else
      ClassDefT(A::Class2<T>,1) //
#endif
   };
   
#ifndef __MAKECINT__
  ClassDefT2(Class2,T)
#endif
}

namespace B {
   class Class3 : public A::Class2<A::Class1>
   {
#ifdef WIN32
	   ClassDef(Class3,1) //
#else
	   ClassDef(B::Class3,1) //
#endif
   };
};

TBuffer &operator>>(TBuffer &,A::Class2<A::Class1> *&);
TBuffer &operator<<(TBuffer &,A::Class2<A::Class1> *);
