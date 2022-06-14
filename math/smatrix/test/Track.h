// dummy track class for testing I/o of matric

#include "Math/SMatrix.h"


typedef double Double32_t;

typedef ROOT::Math::SMatrix<double,5,5,ROOT::Math::MatRepStd<double,5,5> >          SMatrix5D;
typedef ROOT::Math::SMatrix<Double32_t,5,5,ROOT::Math::MatRepStd<Double32_t,5,5> >  SMatrix5D32;

// track class based on SMatrix of double
class  TrackD {

public:
   TrackD() {}

   TrackD(const SMatrix5D & cov) : fCov(cov) {}

   SMatrix5D & CovMatrix() { return fCov; }

private:

   SMatrix5D  fCov;

};

// track class based on Smatrix of Double32

class  TrackD32 {

public:
   TrackD32() {}

   TrackD32(const SMatrix5D32 & cov) : fCov(cov) {}

   SMatrix5D & CovMatrix() { return fCov; }

private:

   SMatrix5D32  fCov;

};
