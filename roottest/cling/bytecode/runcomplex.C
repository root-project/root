#include <iostream>
#include <complex>
using namespace std;

double Smith_x(double R, double Y)
{
   complex<double> Z(R,Y);
   complex<double> Gamma = (Z-complex<double>(1.0,0.0))/(Z+complex<double>(1.0,0.0));
   double result = Gamma.real();
   return result;
}

void runcomplex()
{
   for(int i = 0; i < 11; ++i) {
      double x = Smith_x(0.0, 0.0);
      cout << "x = " << x << endl;
   }
}


