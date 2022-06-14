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


/** \class KelvinFunctions
This class calculates the Kelvin functions Ber(x), Bei(x), Ker(x),
Kei(x), and their first derivatives.
*/

////////////////////////////////////////////////////////////////////////////////
/// \f[
/// Ber(x) = Ber_{0}(x) = Re\left[J_{0}\left(x e^{3\pi i/4}\right)\right]
/// \f]
/// where x is real, and \f$J_{0}(z)\f$ is the zeroth-order Bessel
/// function of the first kind.
///
/// If x < fgMin (=20), Ber(x) is computed according to its polynomial
/// approximation
/// \f[
/// Ber(x) = 1 + \sum_{n \geq 1}\frac{(-1)^{n}(x/2)^{4n}}{[(2n)!]^{2}}
/// \f]
/// For x > fgMin, Ber(x) is computed according to its asymptotic
/// expansion:
/// \f[
/// Ber(x) = \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}} [F1(x) cos\alpha + G1(x) sin\alpha] - \frac{1}{\pi}Kei(x)
/// \f]
/// where \f$\alpha = \frac{x}{\sqrt{2}} - \frac{\pi}{8}\f$.
///
/// See also F1() and G1().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fBer = new TF1("fBer","ROOT::Math::KelvinFunctions::Ber(x)",-10,10);
///   fBer->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::Ber(double x)
{
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

////////////////////////////////////////////////////////////////////////////////
/// \f[
/// Bei(x) = Bei_{0}(x) = Im\left[J_{0}\left(x e^{3\pi i/4}\right)\right]
/// \f]
/// where x is real, and \f$J_{0}(z)\f$ is the zeroth-order Bessel
/// function of the first kind.
///
/// If x < fgMin (=20), Bei(x) is computed according to its polynomial
/// approximation
/// \f[
/// Bei(x) = \sum_{n \geq 0}\frac{(-1)^{n}(x/2)^{4n+2}}{[(2n+1)!]^{2}}
/// \f]
/// For x > fgMin, Bei(x) is computed according to its asymptotic
/// expansion:
/// \f[
/// Bei(x) = \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}} [F1(x) sin\alpha + G1(x) cos\alpha] - \frac{1}{\pi}Ker(x)
/// \f]
/// where \f$\alpha = \frac{x}{\sqrt{2}} - \frac{\pi}{8}\f$.
///
/// See also F1() and G1().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fBei = new TF1("fBei","ROOT::Math::KelvinFunctions::Bei(x)",-10,10);
///   fBei->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::Bei(double x)
{

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


////////////////////////////////////////////////////////////////////////////////
/// \f[
/// Ker(x) = Ker_{0}(x) = Re\left[K_{0}\left(x e^{3\pi i/4}\right)\right]
/// \f]
/// where x is real, and \f$K_{0}(z)\f$ is the zeroth-order modified
/// Bessel function of the second kind.
///
/// If x < fgMin (=20), Ker(x) is computed according to its polynomial
/// approximation
/// \f[
/// Ker(x) = -\left(ln \frac{|x|}{2} + \gamma\right) Ber(x) + \left(\frac{\pi}{4} - \delta\right) Bei(x) + \sum_{n \geq 0} \frac{(-1)^{n}}{[(2n)!]^{2}} H_{2n} \left(\frac{x}{2}\right)^{4n}
/// \f]
/// where \f$\gamma = 0.577215664...\f$ is the Euler-Mascheroni constant,
/// \f$\delta = \pi\f$ for x < 0 and is otherwise zero, and
/// \f[
/// H_{n} = \sum_{k = 1}^{n} \frac{1}{k}
/// \f]
/// For x > fgMin, Ker(x) is computed according to its asymptotic
/// expansion:
/// \f[
/// Ker(x) = \sqrt{\frac{\pi}{2x}} e^{-x/\sqrt{2}} [F2(x) cos\beta + G2(x) sin\beta]
/// \f]
/// where \f$\beta = \frac{x}{\sqrt{2}} + \frac{\pi}{8}\f$.
///
/// See also F2() and G2().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fKer = new TF1("fKer","ROOT::Math::KelvinFunctions::Ker(x)",-10,10);
///   fKer->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::Ker(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// \f[
/// Kei(x) = Kei_{0}(x) = Im\left[K_{0}\left(x e^{3\pi i/4}\right)\right]
/// \f]
/// where x is real, and \f$K_{0}(z)\f$ is the zeroth-order modified
/// Bessel function of the second kind.
///
/// If x < fgMin (=20), Kei(x) is computed according to its polynomial
/// approximation
/// \f[
/// Kei(x) = -\left(ln \frac{x}{2} + \gamma\right) Bei(x) - \left(\frac{\pi}{4} - \delta\right) Ber(x) + \sum_{n \geq 0} \frac{(-1)^{n}}{[(2n)!]^{2}} H_{2n} \left(\frac{x}{2}\right)^{4n+2}
/// \f]
/// where \f$\gamma = 0.577215664...\f$ is the Euler-Mascheroni constant,
/// \f$\delta = \pi\f$ for x < 0 and is otherwise zero, and
/// \f[
/// H_{n} = \sum_{k = 1}^{n} \frac{1}{k}
/// \f]
/// For x > fgMin, Kei(x) is computed according to its asymptotic
/// expansion:
/// \f[
/// Kei(x) = - \sqrt{\frac{\pi}{2x}} e^{-x/\sqrt{2}} [F2(x) sin\beta + G2(x) cos\beta]
/// \f]
/// where \f$\beta = \frac{x}{\sqrt{2}} + \frac{\pi}{8}\f$.
///
/// See also F2() and G2().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fKei = new TF1("fKei","ROOT::Math::KelvinFunctions::Kei(x)",-10,10);
///   fKei->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::Kei(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Calculates the first derivative of Ber(x).
///
/// If x < fgMin (=20), DBer(x) is computed according to the derivative of
/// the polynomial approximation of Ber(x). Otherwise it is computed
/// according to its asymptotic expansion
/// \f[
/// \frac{d}{dx} Ber(x) = M cos\left(\theta - \frac{\pi}{4}\right)
/// \f]
/// See also M() and Theta().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fDBer = new TF1("fDBer","ROOT::Math::KelvinFunctions::DBer(x)",-10,10);
///   fDBer->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::DBer(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Calculates the first derivative of Bei(x).
///
/// If x < fgMin (=20), DBei(x) is computed according to the derivative of
/// the polynomial approximation of Bei(x). Otherwise it is computed
/// according to its asymptotic expansion
/// \f[
/// \frac{d}{dx} Bei(x) = M sin\left(\theta - \frac{\pi}{4}\right)
/// \f]
/// See also M() and Theta().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fDBei = new TF1("fDBei","ROOT::Math::KelvinFunctions::DBei(x)",-10,10);
///   fDBei->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::DBei(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Calculates the first derivative of Ker(x).
///
/// If x < fgMin (=20), DKer(x) is computed according to the derivative of
/// the polynomial approximation of Ker(x). Otherwise it is computed
/// according to its asymptotic expansion
/// \f[
/// \frac{d}{dx} Ker(x) = N cos\left(\phi - \frac{\pi}{4}\right)
/// \f]
/// See also N() and Phi().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fDKer = new TF1("fDKer","ROOT::Math::KelvinFunctions::DKer(x)",-10,10);
///   fDKer->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::DKer(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Calculates the first derivative of Kei(x).
///
/// If x < fgMin (=20), DKei(x) is computed according to the derivative of
/// the polynomial approximation of Kei(x). Otherwise it is computed
/// according to its asymptotic expansion
/// \f[
/// \frac{d}{dx} Kei(x) = N sin\left(\phi - \frac{\pi}{4}\right)
/// \f]
/// See also N() and Phi().
///
/// Begin_Macro
/// {
///   TCanvas *c = new TCanvas("c","c",0,0,500,300);
///   TF1 *fDKei = new TF1("fDKei","ROOT::Math::KelvinFunctions::DKei(x)",-10,10);
///   fDKei->Draw();
///   return c;
/// }
/// End_Macro

double KelvinFunctions::DKei(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the calculations of the Kelvin
/// functions Bei(x) and Ber(x) (and their derivatives). F1(x) is given by
/// \f[
/// F1(x) = 1 + \sum_{n \geq 1} \frac{\prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} cos\left(\frac{n\pi}{4}\right)
/// \f]

double KelvinFunctions::F1(double x)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the calculations of the Kelvin
/// functions Kei(x) and Ker(x) (and their derivatives). F2(x) is given by
/// \f[
/// F2(x) = 1 + \sum_{n \geq 1} (-1)^{n} \frac{\prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} cos\left(\frac{n\pi}{4}\right)
/// \f]

double KelvinFunctions::F2(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the calculations of the Kelvin
/// functions Bei(x) and Ber(x) (and their derivatives). G1(x) is given by
/// \f[
/// G1(x) = \sum_{n \geq 1} \frac{\prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} sin\left(\frac{n\pi}{4}\right)
/// \f]

double KelvinFunctions::G1(double x)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the calculations of the Kelvin
/// functions Kei(x) and Ker(x) (and their derivatives). G2(x) is given by
/// \f[
/// G2(x) = \sum_{n \geq 1} (-1)^{n} \frac{\prod_{m=1}^{n}(2m - 1)^{2}}{n! (8x)^{n}} sin\left(\frac{n\pi}{4}\right)
/// \f]

double KelvinFunctions::G2(double x)
{
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



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the asymptotic expansions of DBer(x) and
/// DBei(x). M(x) is given by
/// \f[
/// M(x) = \frac{e^{x/\sqrt{2}}}{\sqrt{2\pi x}}\left(1 + \frac{1}{8\sqrt{2} x} + \frac{1}{256 x^{2}} - \frac{399}{6144\sqrt{2} x^{3}} + O\left(\frac{1}{x^{4}}\right)\right)
/// \f]

double KelvinFunctions::M(double x)
{
   double value = 1 + 1 / (8 * kSqrt2 * x) + 1 / (256 * x * x) - 399 / (6144 * kSqrt2 * x * x * x);
   value *= exp(x / kSqrt2) / sqrt(2 * kPi * x);
   return value;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the asymptotic expansions of DBer(x) and
/// DBei(x). \f$\theta(x)\f$ is given by
/// \f[
/// \theta(x) = \frac{x}{\sqrt{2}} - \frac{\pi}{8} - \frac{1}{8\sqrt{2} x} - \frac{1}{16 x^{2}} - \frac{25}{384\sqrt{2} x^{3}} + O\left(\frac{1}{x^{5}}\right)
/// \f]

double KelvinFunctions::Theta(double x)
{
   double value = x / kSqrt2 - kPi / 8;
   value -= 1 / (8 * kSqrt2 * x) + 1 / (16 * x * x) + 25 / (384 * kSqrt2 * x * x * x);
   return value;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the asymptotic expansions of DKer(x) and
/// DKei(x). N(x) is given by
/// \f[
/// N(x) = \sqrt{\frac{\pi}{2x}} e^{-x/\sqrt{2}} \left(1 - \frac{1}{8\sqrt{2} x} + \frac{1}{256 x^{2}} + \frac{399}{6144\sqrt{2} x^{3}} + O\left(\frac{1}{x^{4}}\right)\right)
/// \f]

double KelvinFunctions::N(double x)
{
   double value = 1 - 1 / (8 * kSqrt2 * x) + 1 / (256 * x * x) + 399 / (6144 * kSqrt2 * x * x * x);
   value *= exp(- x / kSqrt2) * sqrt(kPi / (2 * x));
   return value;
}



////////////////////////////////////////////////////////////////////////////////
/// Utility function appearing in the asymptotic expansions of DKer(x) and
/// DKei(x). \f$\phi(x)\f$ is given by
/// \f[
/// \phi(x) = - \frac{x}{\sqrt{2}} - \frac{\pi}{8} + \frac{1}{8\sqrt{2} x} - \frac{1}{16 x^{2}} + \frac{25}{384\sqrt{2} x^{3}} + O\left(\frac{1}{x^{5}}\right)
/// \f]

double KelvinFunctions::Phi(double x)
{
   double value = - x / kSqrt2 - kPi / 8;
   value += 1 / (8 * kSqrt2 * x) - 1 / (16 * x * x) + 25 / (384 * kSqrt2 * x * x * x);
   return value;
}


} // namespace Math
} // namespace ROOT



