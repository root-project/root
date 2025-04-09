#include "TNamed.h"

template< class T1 > 
class A { 
   // ... 
private: 
   template< class T2 > 
   class B  { 
      T2 fValue; 
   }; 
   
   template< class T2 > 
   class C : public TNamed { 
      T2 fValue; 
      ClassDef(C,1);
   }; 
   
   A<T1>::B< T1 > fMember3; 
   A::B< T1 > fMember2; 
   B< T1 > fMember1;
   C<T1> fCMember1;

};

template< class T1 > 
class TA : public TObject { 
   // ... 
private: 
   template< class T2 > 
   class B  { 
      T2 fValue; 
   }; 
   
   template< class T2 > 
   class C : public TNamed { 
      T2 fValue; 
      ClassDef(C,1);
   }; 
   
   TA<T1>::B< T1 > fMember3; 
   TA::B< T1 > fMember2; 
   B< T1 > fMember1;
   C<T1> fCMember1;
   
   ClassDef(TA,1);
};

A<short> a;
TA<short> ta;
