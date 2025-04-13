#include <TMath.h>
#include <complex>
#include <iostream>

using namespace std;

typedef double Double;
typedef complex<Double> cmp;
cmp i((Double)0.0, (Double)1.0);

// Surface resistance for copper
Double Rs = 0.5e-3; // Logbook 2008.05.06-2009.03.31 page 67

Double pi = 3.14159265358979323846;
Double c = 299792458;
Double mu0 = 12.566370614e-7;
Double epsilon0 = 8.854187817e-12;

Double f = 35e6;
Double w = 2*pi*f;
Double lambda = c/f;

const Double C1 = 350e-12; // endcap capacitor
const Double C2 = 6.3e-12; // endcap-to-ground: 34mm diam disk 5mm from ground

// Coax: inner diameter: d1, outer diameter: d2
// characteristic impedance: Z0
const Double d1 = 0.127e-3;
const Double d2 = 12e-3;
const Double Z0 = 1/(2*pi) * sqrt(mu0/epsilon0) * log(d2/d1);

int gBarnaDebug = 0;

Double U0 = 25e3;

// Z0 - transmission line impendance (HV-bias cable within the serpentine)
// L  - length of transmission line
// X  - Complex impedance of the filter at the end of the transmission line
//      ideally it is lossless: X = j*x

// Reflection factor on the coax line, at the endcap capacitance
// (transform the Load [i.e. the filter] at l=L to l=0
cmp Gamma_0(Double Z0, Double L, Double x)
{
    cmp Load((Double)0.0, x); // lossless filter
    cmp f1 = (Load/Z0-(Double)1.0)/(Load/Z0+(Double)1.0);
    cmp f2 = exp(-i*(Double)4.0*(Double)pi*L/lambda);
    return f1*f2;
}

bool compare(double x, double y) 
{
   double epsilon = 1e-14;
   double diff = x-y;
   if (x < epsilon ) return  TMath::Abs( diff ) < epsilon;
   else return TMath::Abs( diff/x ) < epsilon;
}

// The transformed impedance of the filter at the endcap capacitance
cmp Xprime(Double Z0, Double L, Double x)
{
    cmp G0 = Gamma_0(Z0,L,x);
    if (gBarnaDebug) cout<<"G0:   "<< G0.real()<<" "<<G0.imag()<<"\n";
    cmp f1 = ((Double)1.0+Gamma_0(Z0,L,x));
    cmp f2 = ((Double)1.0-Gamma_0(Z0,L,x));
    if (gBarnaDebug) cout<<"f1:   "<< f1.real()<<" "<<f1.imag()<<"\n";
    if (gBarnaDebug) cout<<"f2:   "<< f2.real()<<" "<<f2.imag()<<"\n";

    // This agrees with the compiled code's output
    cmp normalized_Z_1 = f1/f2;
    if (gBarnaDebug) cout<<"good: "<< normalized_Z_1.real()<<" "<<normalized_Z_1.imag()<<"\n";

    // Oups, this differs between compiled and interpreted...
    cmp normalized_Z_2 = ((Double)1.0+Gamma_0(Z0,L,x))/((Double)1.0-Gamma_0(Z0,L,x));
    if (gBarnaDebug) cout<<"bad:  "<< normalized_Z_2.real()<<" "<<normalized_Z_2.imag()<<"\n";

    if ( !compare( normalized_Z_1.real(), normalized_Z_2.real()) || 
         !compare( normalized_Z_1.imag(), normalized_Z_2.imag()) ) 
       {
          cout << "Error: Combined and step by step techinque do not agree!\n";
          cout << "Step by step: " << normalized_Z_1.real()<<" "<<normalized_Z_1.imag()<<"\n";
          cout << "Combined    : "  << normalized_Z_1.real()<<" "<<normalized_Z_1.imag()<<"\n";
       }

    cmp result =  Z0*normalized_Z_1;
    if (gBarnaDebug) cout<<"final "<< result.real()<<" "<<result.imag()<<endl;
    return result;
}


int daniel2()
{
    const Double L = 2;
    for(Double l=0; l<L; l += L/20)
    {
       if (gBarnaDebug) cout << "l=" << l << '\n';
       Xprime((Double)50, l, (Double)50);
    }
#ifdef ClingWorkAroundErracticValuePrinter
    printf("(int)0\n");
#endif
    return 0;
} 

int daniel3() {
cmp G; // some complex number
cmp f1 = 1.0+G;
cmp f2 = 1.0-G;
cmp Z1 = f1/f2;       // correct
cmp Z2 = (1.0+Gamma_0(1,2,3))/(1.0-Gamma_0(1,2,3)); // incorrect!
return 0;
}

cmp d4() {
   cmp bad = (((double)1.0)+Gamma_0(1,2,3))/(((double)1.0)-Gamma_0(4,5,6)); // incorrect!
   cmp result = bad;
   return result;
}

int daniel4() {
cmp G; // some complex number
cmp f1 = 1.0+G;
cmp f2 = 1.0-G;
cmp Z1 = f1/f2;       // correct
 cmp Z2 = d4();
return 0;
}

cmp Gamma_1(Double , Double, Double x)
{
    cmp Load((Double)0.0, x); // lossless filter
    return Load;
}

cmp Gamma_2(Double , Double, Double x)
{
    cmp Load((Double)0.0, x); // lossless filter
    return cmp(Load);
}

int daniel5() {
cmp Z2 = (Gamma_1(1,2,3)+Gamma_1(4,5,6)); // incorrect!
return 0;
}

cmp Gamma_3(Double, Double, Double x)
{
   cmp f1(0.0,x);
   return f1;
}

// The transformed impedance of the filter at the endcap capacitance
cmp Xprime2(Double Z0, Double L, Double x)
{
    // Oups, this differs between compiled and interpreted...
    cmp normalized_Z_2 = ((Double)1.0+Gamma_3(Z0,L,x))/((Double)1.0-Gamma_3(Z0,L,x));
    cout<<"bad:  "<< normalized_Z_2.real()<<" "<<normalized_Z_2.imag()<<"\n";

    return normalized_Z_2;
}

int daniel6() {
    const Double L = 2;
    Double l = 0.0;
    for(; l<L; l += L) // /20)
    {
       cout << "l=" << l << '\n';
       Xprime2((Double)50, l, (Double)50);
    }
    return 0;
}

int barna() { return daniel2(); }

int main() { return barna(); }
