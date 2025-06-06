#ifndef runtemplate32_h
#define runtemplate32_h

class WithDouble {
public:
   Double32_t d32;
   double regdouble;
public:
   inline WithDouble();
};

inline WithDouble::WithDouble()
      : d32(0.0)
      , regdouble(0.0)
{
}

template<class T>
class MyVector {
public:
   T d32;
   double regdouble;
public:
   MyVector();
};

template<class T>
MyVector<T>::MyVector()
      : d32(T())
      , regdouble(0.0)
{
}

class Contains {
public:
   MyVector<Double32_t> v1;
   MyVector<float> v2;
};

#endif
