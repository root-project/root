#include <math.h>
#include "Riostream.h"
#include "Quad.h"

Quad::Quad(Float_t a,Float_t b,Float_t c)
{
   fA = a;
   fB = b;
   fC = c;
}

Quad::~Quad()
{
   cout << "deleting object with coeffts: "
        << fA << "," << fB << "," << fC << endl;
}

void Quad::Solve() const
{
   Float_t temp = fB*fB -4*fA*fC;
   if (temp > 0) {
      temp = sqrt(temp);
      cout << "There are two roots: "
           << ( -fB - temp ) / (2.*fA)
           << " and "
           << ( -fB + temp ) / (2.*fA)
           << endl;
   } else {
      if (temp == 0) {
         cout << "There are two equal roots: "
         << -fB / (2.*fA) << endl;
      } else {
         cout << "There are no roots" << endl;
      }
   }
}

Float_t Quad::Evaluate(Float_t x) const
{
  return fA*x*x + fB*x + fC;
  return 0;
}
