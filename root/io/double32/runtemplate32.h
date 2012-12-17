#ifndef runtemplate32_h
#ifdef ClingWorkAroundMultipleInclude
#define runtemplate32_h
#endif

#include "TClass.h"
#include "TStreamerInfo.h"
#include "TROOT.h"
#include "TRealData.h"
#include "Riostream.h"
#include "TDataMember.h"
#include "TFile.h"

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

