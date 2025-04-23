#ifndef _h1_
#define _h1_

template <class T>
class myTemplate;

template <class T>
class myClass0{
private:
   T dummy;
};

template <class T>
class myClass1{
private:
   T**** dummy;
};

template <class T>
class myClass2{
public:
   myClass2(T**& a):dummy(a){};
private:
   T**& dummy;
};

template <class T>
class myClass3{
public:
   void myMethod(T a){};
};

template <class T>
class myClass4{
public:
   void myMethod(const T*const**&& a){}; // Unluky signature but interesting for this test ;-)
};

//--------------------

template <class T>
class myClass5{
private:
   myTemplate<T> dummy;
};

template <class T>
class myClass6{
private:
   myTemplate<T>* dummy;
};

template <class T>
class myClass7{
private:
   const myTemplate<T***const>**const* dummy;
};

#include <queue>
//--------------------
template <class T>
class myClass8{
private:
   std::set<T> dummy;
};

#include <vector>
template <class T>
class myClass9{
private:
   std::set<std::vector<T>> dummy;
};


#endif
