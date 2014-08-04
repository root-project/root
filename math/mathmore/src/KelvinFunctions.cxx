// @(#)root/mathmore:$Id$
// The functions in this class have been imported by Jason Detwiler (jasondet@gmail.com) from
// CodeCogs GNU General Public License Agreement
// Copyright (C) 2004-2005 CodeCogs, Zyba Ltd, Broadwood, Holford, TA5 1DU,
// England.
//
// This program is free software; you can redistribute it and/or modify it
// under
// the terms of the GNU General Public License as published by CodeCogs.
// You must retain a copy of this licence in all copies.
//
// This program is distributed in the hope that it will be useful, but
// WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A
// PARTICULAR PURPOSE. See the Adapted GNU General Public License for more
// details.
//
// *** THIS SOFTWARE CAN NOT BE USED FOR COMMERCIAL GAIN. ***
// ---------------------------------------------------------------------------------

#include "Math/KelvinFunctions.h"
#include <math.h>


namespace ROOT {
namespace Math {

double KelvinFunctions::fgMin = 20;
double KelvinFunctions::fgEpsilon = 1.e-20;

double kSqrt2 = 1.4142135623730950488016887242097;
double kPi    = 3.14159265358979323846;
double kEulerGamma = 0.577215664901532860606512090082402431042;


/* Begin_Html
<center><h2>KelvinFunctions</h2></center>

<p>
This class calculates the Kelvin functions Ber(x), Bei(x), Ker(x),
Kei(x), and their first derivatives.
</p>

End_Html */

//______________________________________________________________________________
double KelvinFunctions::Ber(double x)
{
   // Begin_Latex
   // Ber(x) = Ber_{0}(x) = Re#left[J_{0}#left(x e^{3#pii/4}#right)#right]
   // End_Latex
   // where x is real, and Begin_Latex J_{0}(z) End_Latex is the zeroth-order Bessel
   // function of the first kind.
   //
   // If x < fgMin (=20), Ber(x) is computed according to its polynomial
   // approximation
   // Begin_Latex
   // Ber(x) = 1 + #sum_{n #geq 1}#frac{(-1)^{n}(x/2)^{4n}}{[(2n)!]^{2}}
   // End_Latex
   // For x > fgMin, Ber(x) is computed according to its asymptotic
   // expansion:
   // Begin_Latex
   // Ber(x) = #frac{e^{x/#sqrt{2}}}{#sqrt{2#pix}} [F1(x) cos#alpha + G1(x) sin#alpha] - #frac{1}{#pi}Kei(x)
   // End_Latex
   // where Begin_Latex #alpha = #frac{x}{#sqrt{2}} - #frac{#pi}{8} End_Latex.
   // See also F1(x) and G1(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fBer = new TF1("fBer","ROOT::Math::KelvinFunctions::Ber(x)",-10,10);
   //   fBer->Draw();
   //   return c;
   // }
   // End_Macro

   if (fabs(x) < fgEpsilon) return 1;

   if (fabs(x) < fgMin) {
      double sum, factorial = 1, n = 1;
      double term = 1, x_factor = x * x * x * x * 0.0625;

      sum = 1;

      do {
         factorial = 4 * n * n * (2 * n - 1) * (2 * n - 1);
         term *= (-1) / factorial * x_factor;
         sum += term;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term) > fgEpsilon * sum);

      return sum;
   } else {
      double alpha = x / kSqrt2 - kPi / 8;
      double value = F1(x) * cos(alpha) + G1(x) * sin(alpha);
      value *= exp(x / kSqrt2) / sqrt(2 * kPi * x);
      value -= Kei(x) / kPi;
      return value;
   }
}

//______________________________________________________________________________
double KelvinFunctions::Bei(double x)
{
   // Begin_Latex
   // Bei(x) = Bei_{0}(x) = Im#left[J_{0}#left(x e^{3#pii/4}#right)#right]
   // End_Latex
   // where x is real, and Begin_Latex J_{0}(z) End_Latex is the zeroth-order Bessel
   // function of the first kind.
   //
   // If x < fgMin (=20), Bei(x) is computed according to its polynomial
   // approximation
   // Begin_Latex
   // Bei(x) = #sum_{n #geq 0}#frac{(-1)^{n}(x/2)^{4n+2}}{[(2n+1)!]^{2}}
   // End_Latex
   // For x > fgMin, Bei(x) is computed according to its asymptotic
   // expansion:
   // Begin_Latex
   // Bei(x) = #frac{e^{x/#sqrt{2}}}{#sqrt{2#pix}} [F1(x) sin#alpha + G1(x) cos#alpha] - #frac{1}{#pi}Ker(x)
   // End_Latex
   // where Begin_Latex #alpha = #frac{x}{#sqrt{2}} - #frac{#pi}{8} End_Latex
   // See also F1(x) and G1(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fBei = new TF1("fBei","ROOT::Math::KelvinFunctions::Bei(x)",-10,10);
   //   fBei->Draw();
   //   return c;
   // }
   // End_Macro


   if (fabs(x) < fgEpsilon) return 0;

   if (fabs(x) < fgMin) {
      double sum, factorial = 1, n = 1;
      double term = x * x * 0.25, x_factor = term * term;

      sum = term;

      do {
         factorial = 4 * n * n * (2 * n + 1) * (2 * n + 1);
         term *= (-1) / factorial * x_factor;
         sum += term;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term) > fgEpsilon * sum);

      return sum;
   } else {
      double alpha = x / kSqrt2 - kPi / 8;
      double value = F1(x) * sin(alpha) + G1(x) * cos(alpha);
      value *= exp(x / kSqrt2) / sqrt(2 * kPi * x);
      value += Ker(x) / kPi;
      return value;
   }
}


//______________________________________________________________________________
double KelvinFunctions::Ker(double x)
{
   // Begin_Latex
   // Ker(x) = Ker_{0}(x) = Re#left[K_{0}#left(x e^{3#pii/4}#right)#right]
   // End_Latex
   // where x is real, and Begin_Latex K_{0}(z) End_Latex is the zeroth-order modified
   // Bessel function of the second kind.
   //
   // If x < fgMin (=20), Ker(x) is computed according to its polynomial
   // approximation
   // Begin_Latex
   // Ker(x) = -#left(ln #frac{|x|}{2} + #gamma#right) Ber(x) + #left(#frac{#pi}{4} - #delta#right) Bei(x) + #sum_{n #geq 0} #frac{(-1)^{n}}{[(2n)!]^{2}} H_{2n} #left(#frac{x}{2}#right)^{4n}
   // End_Latex
   // where Begin_Latex #gamma = 0.577215664... End_Latex is the Euler-Mascheroni constant,
   // Begin_Latex #delta = #pi End_Latex for x < 0 and is otherwise zero, and
   // Begin_Latex
   // H_{n} = #sum_{k = 1}^{n} #frac{1}{k}
   // End_Latex
   // For x > fgMin, Ker(x) is computed according to its asymptotic
   // expansion:
   // Begin_Latex
   // Ker(x) = #sqrt{#frac{#pi}{2x}} e^{-x/#sqrt{2}} [F2(x) cos#beta + G2(x) sin#beta]
   // End_Latex
   // where Begin_Latex #beta = #frac{x}{#sqrt{2}} + #frac{#pi}{8} End_Latex
   // See also F2(x) and G2(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fKer = new TF1("fKer","ROOT::Math::KelvinFunctions::Ker(x)",-10,10);
   //   fKer->Draw();
   //   return c;
   // }
   // End_Macro

   if (fabs(x) < fgEpsilon) return 1E+100;

   if (fabs(x) < fgMin) {
      double term = 1, x_factor = x * x * x * x * 0.0625;
      double factorial = 1, harmonic = 0, n = 1, sum;
      double delta = 0;
      if(x < 0) delta = kPi;

      sum  = - (log(fabs(x) * 0.5) + kEulerGamma) * Ber(x) + (kPi * 0.25 - delta) * Bei(x);

      do {
         factorial = 4 * n * n * (2 * n - 1) * (2 * n - 1);
         term *= (-1) / factorial * x_factor;
         harmonic += 1 / (2 * n - 1 ) + 1 / (2 * n);
         sum += term * harmonic;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term * harmonic) > fgEpsilon * sum);

      return sum;
   } else {
      double beta = x / kSqrt2 + kPi / 8;
      double value = F2(x) * cos(beta) - G2(x) * sin(beta);
      value *= sqrt(kPi / (2 * x)) * exp(- x / kSqrt2);
      return value;
   }
}



//______________________________________________________________________________
double KelvinFunctions::Kei(double x)
{
   // Begin_Latex
   // Kei(x) = Kei_{0}(x) = Im#left[K_{0}#left(x e^{3#pii/4}#right)#right]
   // End_Latex
   // where x is real, and Begin_Latex K_{0}(z) End_Latex is the zeroth-order modified
   // Bessel function of the second kind.
   //
   // If x < fgMin (=20), Kei(x) is computed according to its polynomial
   // approximation
   // Begin_Latex
   // Kei(x) = -#left(ln #frac{x}{2} + #gamma#right) Bei(x) - #left(#frac{#pi}{4} - #delta#right) Ber(x) + #sum_{n #geq 0} #frac{(-1)^{n}}{[(2n)!]^{2}} H_{2n} #left(#frac{x}{2}#right)^{4n+2}
   // End_Latex
   // where Begin_Latex #gamma = 0.577215664... End_Latex is the Euler-Mascheroni constant,
   // Begin_Latex #delta = #pi End_Latex for x < 0 and is otherwise zero, and
   // Begin_Latex
   // H_{n} = #sum_{k = 1}^{n} #frac{1}{k}
   // End_Latex
   // For x > fgMin, Kei(x) is computed according to its asymptotic
   // expansion:
   // Begin_Latex
   // Kei(x) = - #sqrt{#frac{#pi}{2x}} e^{-x/#sqrt{2}} [F2(x) sin#beta + G2(x) cos#beta]
   // End_Latex
   // where Begin_Latex #beta = #frac{x}{#sqrt{2}} + #frac{#pi}{8} End_Latex
   // See also F2(x) and G2(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fKei = new TF1("fKei","ROOT::Math::KelvinFunctions::Kei(x)",-10,10);
   //   fKei->Draw();
   //   return c;
   // }
   // End_Macro

   if (fabs(x) < fgEpsilon) return (-0.25 * kPi);

   if (fabs(x) < fgMin) {
      double term = x * x * 0.25, x_factor = term * term;
      double factorial = 1, harmonic = 1, n = 1, sum;
      double delta = 0;
      if(x < 0) delta = kPi;

      sum  = term - (log(fabs(x) * 0.5) + kEulerGamma) * Bei(x) - (kPi * 0.25 - delta) * Ber(x);

      do {
         factorial = 4 * n * n * (2 * n + 1) * (2 * n + 1);
         term *= (-1) / factorial * x_factor;
         harmonic += 1 / (2 * n) + 1 / (2 * n + 1);
         sum += term * harmonic;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term * harmonic) > fgEpsilon * sum);

      return sum;
   } else {
      double beta = x / kSqrt2 + kPi / 8;
      double value = - F2(x) * sin(beta) - G2(x) * cos(beta);
      value *= sqrt(kPi / (2 * x)) * exp(- x / kSqrt2);
      return value;
   }
}



//______________________________________________________________________________
double KelvinFunctions::DBer(double x)
{
   // Calculates the first derivative of Ber(x).
   //
   // If x < fgMin (=20), DBer(x) is computed according to the derivative of
   // the polynomial approximation of Ber(x). Otherwise it is computed
   // according to its asymptotic expansion
   // Begin_Latex
   // #frac{d}{dx} Ber(x) = M cos#left(#theta - #frac{#pi}{4}#right)
   // End_Latex
   // See also M(x) and Theta(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fDBer = new TF1("fDBer","ROOT::Math::KelvinFunctions::DBer(x)",-10,10);
   //   fDBer->Draw();
   //   return c;
   // }
   // End_Macro
   if (fabs(x) < fgEpsilon) return 0;

   if (fabs(x) < fgMin) {
      double sum, factorial = 1, n = 1;
      double term = - x * x * x * 0.0625, x_factor = - term * x;

      sum = term;

      do {
         factorial = 4 * n * (n + 1) * (2 * n + 1) * (2 * n + 1);
         term *= (-1) / factorial * x_factor;
         sum += term;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term) > fgEpsilon * sum);

      return sum;
   }
   else return (M(x) * sin(Theta(x) - kPi / 4));
}



//______________________________________________________________________________
double KelvinFunctions::DBei(double x)
{
   // Calculates the first derivative of Bei(x).
   //
   // If x < fgMin (=20), DBei(x) is computed according to the derivative of
   // the polynomial approximation of Bei(x). Otherwise it is computed
   // according to its asymptotic expansion
   // Begin_Latex
   // #frac{d}{dx} Bei(x) = M sin#left(#theta - #frac{#pi}{4}#right)
   // End_Latex
   // See also M(x) and Theta(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fDBei = new TF1("fDBei","ROOT::Math::KelvinFunctions::DBei(x)",-10,10);
   //   fDBei->Draw();
   //   return c;
   // }
   // End_Macro
   if (fabs(x) < fgEpsilon) return 0;

   if (fabs(x) < fgMin) {
      double sum, factorial = 1, n = 1;
      double term = x * 0.5, x_factor = x * x * x * x * 0.0625;

      sum = term;

      do {
         factorial = 4 * n * n * (2 * n - 1) * (2 * n + 1);
         term *= (-1) * x_factor / factorial;
         sum += term;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term) > fgEpsilon * sum);

      return sum;
   }
   else return (M(x) * cos(Theta(x) - kPi / 4));
}



//______________________________________________________________________________
double KelvinFunctions::DKer(double x)
{
   // Calculates the first derivative of Ker(x).
   //
   // If x < fgMin (=20), DKer(x) is computed according to the derivative of
   // the polynomial approximation of Ker(x). Otherwise it is computed
   // according to its asymptotic expansion
   // Begin_Latex
   // #frac{d}{dx} Ker(x) = N cos#left(#phi - #frac{#pi}{4}#right)
   // End_Latex
   // See also N(x) and Phi(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fDKer = new TF1("fDKer","ROOT::Math::KelvinFunctions::DKer(x)",-10,10);
   //   fDKer->Draw();
   //   return c;
   // }
   // End_Macro
   if (fabs(x) < fgEpsilon) return -1E+100;

   if (fabs(x) < fgMin) {
      double term = - x * x * x * 0.0625, x_factor = - term * x;
      double factorial = 1, harmonic = 1.5, n = 1, sum;
      double delta = 0;
      if(x < 0) delta = kPi;

      sum = 1.5 * term - Ber(x) / x - (log(fabs(x) * 0.5) + kEulerGamma) * DBer(x) + (0.25 * kPi - delta) * DBei(x);

      do {
         factorial = 4 * n * (n + 1) * (2 * n + 1) * (2 * n + 1);
         term *= (-1) / factorial * x_factor;
         harmonic += 1 / (2 * n + 1 ) + 1 / (2 * n + 2);
         sum += term * harmonic;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term * harmonic) > fgEpsilon * sum);

      return sum;
   }
   else return N(x) * sin(Phi(x) - kPi / 4);
}



//______________________________________________________________________________
double KelvinFunctions::DKei(double x)
{
   // Calculates the first derivative of Kei(x).
   //
   // If x < fgMin (=20), DKei(x) is computed according to the derivative of
   // the polynomial approximation of Kei(x). Otherwise it is computed
   // according to its asymptotic expansion
   // Begin_Latex
   // #frac{d}{dx} Kei(x) = N sin#left(#phi - #frac{#pi}{4}#right)
   // End_Latex
   // See also N(x) and Phi(x).
   //
   // Begin_Macro
   // {
   //   TCanvas *c = new TCanvas("c","c",0,0,500,300);
   //   TF1 *fDKei = new TF1("fDKei","ROOT::Math::KelvinFunctions::DKei(x)",-10,10);
   //   fDKei->Draw();
   //   return c;
   // }
   // End_Macro
   if (fabs(x) < fgEpsilon) return 0;

   if (fabs(x) < fgMin) {
      double term = 0.5 * x, x_factor = x * x * x * x * 0.0625;
      double factorial = 1, harmonic = 1, n = 1, sum;
      double delta = 0;
      if(x < 0) delta = kPi;

      sum  = term - Bei(x) / x - (log(fabs(x) * 0.5) + kEulerGamma) * DBei(x) - (kPi * 0.25 - delta) * DBer(x);

      do {
         factorial = 4 * n * n * (2 * n - 1) * (2 * n + 1);
         term *= (-1) / factorial * x_factor;
         harmonic += 1 / (2 * n) + 1 / (2 * n + 1);
         sum += term * harmonic;
         n += 1;
         if (n > 1000) break;
      } while (fabs(term * harmonic) > fgEpsilon * sum);

      return sum;
   }
   else return N(x) * cos(Phi(x) - kPi / 4);
}



//______________________________________________________________________________
double KelvinFunctions::F1(double x)
{
   // Utility function appearing in the calculations of the Kelvin
   // functions Bei(x) and Ber(x) (and their derivatives). F1(x) is given by
   // Begin_Latex
   // F1(x) = 1 + #sum_{n #geq 1} #frac{#prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} cos#left(#frac{n#pi}{4}#right)
   // End_Latex
   double sum, term;
   double prod = 1, x_factor = 8 * x, factorial = 1, n = 2;

   sum = kSqrt2 / (16 * x);

   do {
      factorial *= n;
      prod *= (2 * n - 1) * (2 * n - 1);
      x_factor *= 8 * x;
      term = prod / (factorial * x_factor) * cos(0.25 * n * kPi);
      sum += term;
      n += 1;
      if (n > 1000) break;
   } while (fabs(term) > fgEpsilon * sum);

   sum += 1;

   return sum;
}

//______________________________________________________________________________
double KelvinFunctions::F2(double x)
{
   // Utility function appearing in the calculations of the Kelvin
   // functions Kei(x) and Ker(x) (and their derivatives). F2(x) is given by
   // Begin_Latex
   // F2(x) = 1 + #sum_{n #geq 1} (-1)^{n} #frac{#prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} cos#left(#frac{n#pi}{4}#right)
   // End_Latex
   double sum, term;
   double prod = 1, x_factor = 8 * x, factorial = 1, n = 2;

   sum = kSqrt2 / (16 * x);

   do {
      factorial *= - n;
      prod *= (2 * n - 1) * (2 * n - 1);
      x_factor *= 8 * x;
      term = (prod / (factorial * x_factor)) * cos(0.25 * n * kPi);
      sum += term;
      n += 1;
      if (n > 1000) break;
   } while (fabs(term) > fgEpsilon * sum);

   sum += 1;

   return sum;
}



//______________________________________________________________________________
double KelvinFunctions::G1(double x)
{
   // Utility function appearing in the calculations of the Kelvin
   // functions Bei(x) and Ber(x) (and their derivatives). G1(x) is given by
   // Begin_Latex
   // G1(x) = #sum_{n #geq 1} #frac{#prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} sin#left(#frac{n#pi}{4}#right)
   // End_Latex
   double sum, term;
   double prod = 1, x_factor = 8 * x, factorial = 1, n = 2;

   sum = kSqrt2 / (16 * x);

   do {
      factorial *= n;
      prod *= (2 * n - 1) * (2 * n - 1);
      x_factor *= 8 * x;
      term = prod / (factorial * x_factor) * sin(0.25 * n * kPi);
      sum += term;
      n += 1;
      if (n > 1000) break;
   } while (fabs(term) > fgEpsilon * sum);

   return sum;
}

//______________________________________________________________________________
double KelvinFunctions::G2(double x)
{
   // Utility function appearing in the calculations of the Kelvin
   // functions Kei(x) and Ker(x) (and their derivatives). G2(x) is given by
   // Begin_Latex
   // G2(x) = #sum_{n #geq 1} (-1)^{n} #frac{#prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} sin#left(#frac{n#pi}{4}#right)
   // End_Latex
   double sum, term;
   double prod = 1, x_factor = 8 * x, factorial = 1, n = 2;

   sum = kSqrt2 / (16 * x);

   do {
      factorial *= - n;
      prod *= (2 * n - 1) * (2 * n - 1);
      x_factor *= 8 * x;
      term = prod / (factorial * x_factor) * sin(0.25 * n * kPi);
      sum += term;
      n += 1;
      if (n > 1000) break;
   } while (fabs(term) > fgEpsilon * sum);

   return sum;
}



//______________________________________________________________________________
double KelvinFunctions::M(double x)
{
   // Utility function appearing in the asymptotic expansions of DBer(x) and
   // DBei(x). M(x) is given by
   // Begin_Latex
   // M(x) = #frac{e^{x/#sqrt{2}}}{#sqrt{2#pix}}#left(1 + #frac{1}{8#sqrt{2} x} + #frac{1}{256 x^{2}} - #frac{399}{6144#sqrt{2} x^{3}} + O#left(#frac{1}{x^{4}}#right)#right)
   // End_Latex
   double value = 1 + 1 / (8 * kSqrt2 * x) + 1 / (256 * x * x) - 399 / (6144 * kSqrt2 * x * x * x);
   value *= exp(x / kSqrt2) / sqrt(2 * kPi * x);
   return value;
}



//______________________________________________________________________________
double KelvinFunctions::Theta(double x)
{
   // Utility function appearing in the asymptotic expansions of DBer(x) and
   // DBei(x). Begin_Latex #theta(x) #End_Latex is given by
   // Begin_Latex
   // #theta(x) = #frac{x}{#sqrt{2}} - #frac{#pi}{8} - #frac{1}{8#sqrt{2} x} - #frac{1}{16 x^{2}} - #frac{25}{384#sqrt{2} x^{3}} + O#left(#frac{1}{x^{5}}#right)
   // End_Latex
   double value = x / kSqrt2 - kPi / 8;
   value -= 1 / (8 * kSqrt2 * x) + 1 / (16 * x * x) + 25 / (384 * kSqrt2 * x * x * x);
   return value;
}



//______________________________________________________________________________
double KelvinFunctions::N(double x)
{
   // Utility function appearing in the asymptotic expansions of DKer(x) and
   // DKei(x). (x) is given by
   // Begin_Latex
   // N(x) = #sqrt{#frac{#pi}{2x}} e^{-x/#sqrt{2}} #left(1 - #frac{1}{8#sqrt{2} x} + #frac{1}{256 x^{2}} + #frac{399}{6144#sqrt{2} x^{3}} + O#left(#frac{1}{x^{4}}#right)#right)
   // End_Latex
   double value = 1 - 1 / (8 * kSqrt2 * x) + 1 / (256 * x * x) + 399 / (6144 * kSqrt2 * x * x * x);
   value *= exp(- x / kSqrt2) * sqrt(kPi / (2 * x));
   return value;
}



//______________________________________________________________________________
double KelvinFunctions::Phi(double x)
{
   // Utility function appearing in the asymptotic expansions of DKer(x) and
   // DKei(x). Begin_Latex #phi(x) #End_Latex is given by
   // Begin_Latex
   // #phi(x) = - #frac{x}{#sqrt{2}} - #frac{#pi}{8} + #frac{1}{8#sqrt{2} x} - #frac{1}{16 x^{2}} + #frac{25}{384#sqrt{2} x^{3}} + O#left(#frac{1}{x^{5}}#right)
   // End_Latex
   double value = - x / kSqrt2 - kPi / 8;
   value += 1 / (8 * kSqrt2 * x) - 1 / (16 * x * x) + 25 / (384 * kSqrt2 * x * x * x);
   return value;
}


} // namespace Math
} // namespace ROOT



