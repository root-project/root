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
};

template <class T> class MyVector {
public:
   T d32;
   double regdouble;
};

class Contains {
public:
   MyVector<Double32_t> v1;
   MyVector<float> v2;
};

