// dummy track class for testing I/o of matric

#include "Math/SMatrixDfwd.h"
#include "Math/SMatrixD32fwd.h"
#include "Math/SMatrix.h"



// track class based on SMatrix of double
class  TrackD { 

public:
   TrackD() {}

   TrackD(const ROOT::Math::SMatrix5D & cov) : fCov(cov) {}

   ROOT::Math::SMatrix5D & CovMatrix() { return fCov; }

private:
   
   ROOT::Math::SMatrix5D  fCov; 
   
}; 

// track class based on Smatrix of Double32

class  TrackD32 { 

public:
   TrackD32() {}

   TrackD32(const ROOT::Math::SMatrix5D32 & cov) : fCov(cov) {}

   ROOT::Math::SMatrix5D & CovMatrix() { return fCov; }

private:
   
   ROOT::Math::SMatrix5D32  fCov; 
   
}; 
