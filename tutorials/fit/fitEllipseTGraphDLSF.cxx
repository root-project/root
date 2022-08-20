//
// The "fitEllipseTGraphDLSF" macro uses the "Direct Least Squares Fitting"
// algorithm for fitting an ellipse to a set of data points from a TGraph
//
// To try this macro, in a ROOT prompt, do:
// .L fitEllipseTGraphDLSF.cxx // or ".L fitEllipseTGraphDLSF.cxx++"
// fitEllipseTGraphDLSF(TestGraphDLSF());
// for (int i=0; i<10; i++) { fitEllipseTGraphDLSF(); gSystem->Sleep(333); }
//
// Last update: Thu Jul 31 18:00:00 UTC 2014
//
// Changes:
// 2014.07.31 - (initial version)
//

#include "TROOT.h"
#include "TMath.h"
#include "TRandom.h"
#include "TGraph.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDEigen.h"
#include "TCanvas.h"
#include "TEllipse.h"

#include <cmath>
#include <iostream>

//
// "NUMERICALLY STABLE DIRECT LEAST SQUARES FITTING OF ELLIPSES"
// Radim Halir, Jan Flusser
// http://autotrace.sourceforge.net/WSCG98.pdf
//
// http://en.wikipedia.org/wiki/Ellipse
//
// An "algebraic distance" of a point "(x, y)" to a given conic:
//   F(x, y) = A * (x - X0)^2 + B * (x - X0) * (y - Y0) + C * (y - Y0)^2
//           + D * (x - X0) + E * (y - Y0) + F
//
// Ellipse-specific constraints:
//   F(x, y) = 0
//   B^2 - 4 * A * C < 0
//
// input parameter is a pointer to a "TGraph" with at least 6 points
//
// returns a "TVectorD" ("empty" in case any problems encountered):
// ellipse[0] = "X0"
// ellipse[1] = "Y0"
// ellipse[2] = "A"
// ellipse[3] = "B"
// ellipse[4] = "C"
// ellipse[5] = "D"
// ellipse[6] = "E"
// ellipse[7] = "F"
//
TVectorD fit_ellipse(TGraph *g)
{
  TVectorD ellipse;

  if (!g) return ellipse; // just a precaution
  if (g->GetN() < 6) return ellipse; // just a precaution

  int i;
  double tmp;

  int N = g->GetN();
  double xmin, xmax, ymin, ymax, X0, Y0;
  g->ComputeRange(xmin, ymin, xmax, ymax);
#if 1 /* 0 or 1 */
  X0 = (xmax + xmin) / 2.0;
  Y0 = (ymax + ymin) / 2.0;
#else /* 0 or 1 */
  X0 = Y0 = 0.0;
#endif /* 0 or 1 */

  TMatrixD D1(N, 3); // quadratic part of the design matrix
  TMatrixD D2(N, 3); // linear part of the design matrix

  for (i = 0; i < N; i++) {
    double x = (g->GetX())[i] - X0;
    double y = (g->GetY())[i] - Y0;
    D1[i][0] = x * x;
    D1[i][1] = x * y;
    D1[i][2] = y * y;
    D2[i][0] = x;
    D2[i][1] = y;
    D2[i][2] = 1.0;
  }

  // quadratic part of the scatter matrix
  TMatrixD S1(TMatrixD::kAtA, D1);
  // combined part of the scatter matrix
  TMatrixD S2(D1, TMatrixD::kTransposeMult, D2);
  // linear part of the scatter matrix
  TMatrixD S3(TMatrixD::kAtA, D2);
  S3.Invert(&tmp); S3 *= -1.0;
  if (tmp == 0.0) {
    std::cout << "fit_ellipse : linear part of the scatter matrix is singular!" << std::endl;
    return ellipse;
  }
  // for getting a2 from a1
  TMatrixD T(S3, TMatrixD::kMultTranspose, S2);
  // reduced scatter matrix
  TMatrixD M(S2, TMatrixD::kMult, T); M += S1;
  // premultiply by inv(C1)
  for (i = 0; i < 3; i++) {
    tmp = M[0][i] / 2.0;
    M[0][i] = M[2][i] / 2.0;
    M[2][i] = tmp;
    M[1][i] *= -1.0;
  }
  // solve eigensystem
  TMatrixDEigen eig(M); // note: eigenvectors are not normalized
  const TMatrixD &evec = eig.GetEigenVectors();
  // const TVectorD &eval = eig.GetEigenValuesRe();
  if ((eig.GetEigenValuesIm()).Norm2Sqr() != 0.0) {
    std::cout << "fit_ellipse : eigenvalues have nonzero imaginary parts!" << std::endl;
    return ellipse;
  }
  // evaluate a’Ca (in order to find the eigenvector for min. pos. eigenvalue)
  for (i = 0; i < 3; i++) {
    tmp = 4.0 * evec[0][i] * evec[2][i] - evec[1][i] * evec[1][i];
    if (tmp > 0.0) break;
  }
  if (i > 2) {
    std::cout << "fit_ellipse : no min. pos. eigenvalue found!" << std::endl;
    // i = 2;
    return ellipse;
  }
  // eigenvector for min. pos. eigenvalue
  TVectorD a1(TMatrixDColumn_const(evec, i));
  tmp = a1.Norm2Sqr();
  if (tmp > 0.0) {
    a1 *= 1.0 / std::sqrt(tmp); // normalize this eigenvector
  } else {
    std::cout << "fit_ellipse : eigenvector for min. pos. eigenvalue is NULL!" << std::endl;
    return ellipse;
  }
  TVectorD a2(T * a1);

  // ellipse coefficients
  ellipse.ResizeTo(8);
  ellipse[0] = X0; // "X0"
  ellipse[1] = Y0; // "Y0"
  ellipse[2] = a1[0]; // "A"
  ellipse[3] = a1[1]; // "B"
  ellipse[4] = a1[2]; // "C"
  ellipse[5] = a2[0]; // "D"
  ellipse[6] = a2[1]; // "E"
  ellipse[7] = a2[2]; // "F"

  return ellipse;
}

//
// http://mathworld.wolfram.com/Ellipse.html
// http://mathworld.wolfram.com/QuadraticCurve.html
// http://mathworld.wolfram.com/ConicSection.html
//
// "Using the Ellipse to Fit and Enclose Data Points"
// Charles F. Van Loan
// http://www.cs.cornell.edu/cv/OtherPdf/Ellipse.pdf
//
// input parameter is a reference to a "TVectorD" which describes
// an ellipse according to the equation:
//   0 = A * (x - X0)^2 + B * (x - X0) * (y - Y0) + C * (y - Y0)^2
//     + D * (x - X0) + E * (y - Y0) + F
// conic[0] = "X0"
// conic[1] = "Y0"
// conic[2] = "A"
// conic[3] = "B"
// conic[4] = "C"
// conic[5] = "D"
// conic[6] = "E"
// conic[7] = "F"
//
// returns a "TVectorD" ("empty" in case any problems encountered):
// ellipse[0] = ellipse's "x" center ("x0")
// ellipse[1] = ellipse's "y" center ("y0")
// ellipse[2] = ellipse's "semimajor" axis along "x" ("a" > 0)
// ellipse[3] = ellipse's "semiminor" axis along "y" ("b" > 0)
// ellipse[4] = ellipse's axes rotation angle ("theta" = -45 ... 135 degrees)
//
TVectorD ConicToParametric(const TVectorD &conic)
{
  TVectorD ellipse;

  if (conic.GetNrows() != 8) {
    std::cout << "ConicToParametric : improper input vector length!" << std::endl;
    return ellipse;
  }

  double a, b, theta;
  double x0 = conic[0]; // = X0
  double y0 = conic[1]; // = Y0

  // http://mathworld.wolfram.com/Ellipse.html
  double A = conic[2];
  double B = conic[3] / 2.0;
  double C = conic[4];
  double D = conic[5] / 2.0;
  double F = conic[6] / 2.0;
  double G = conic[7];

  double J = B * B - A * C;
  double Delta = A * F * F + C * D * D + J * G - 2.0 * B * D * F;
  double I = - (A + C);

  // http://mathworld.wolfram.com/QuadraticCurve.html
  if (!( (Delta != 0.0) && (J < 0.0) && (I != 0.0) && (Delta / I < 0.0) )) {
    std::cout << "ConicToParametric : ellipse (real) specific constraints not met!" << std::endl;
    return ellipse;
  }

  x0 += (C * D - B * F) / J;
  y0 += (A * F - B * D) / J;

  double tmp = std::sqrt((A - C) * (A - C) + 4.0 * B * B);
  a = std::sqrt(2.0 * Delta / J / (I + tmp));
  b = std::sqrt(2.0 * Delta / J / (I - tmp));

  theta = 0.0;
  if (B != 0.0) {
    tmp = (A - C) / 2.0 / B;
    theta = -45.0 * (std::atan(tmp) / TMath::PiOver2());
    if (tmp < 0.0) { theta -= 45.0; } else { theta += 45.0; }
    if (A > C) theta += 90.0;
  } else if (A > C) theta = 90.0;

  // try to keep "a" > "b"
  if (a < b) { tmp = a; a = b; b = tmp; theta -= 90.0; }
  // try to keep "theta" = -45 ... 135 degrees
  if (theta < -45.0) theta += 180.0;
  if (theta > 135.0) theta -= 180.0;

  // ellipse coefficients
  ellipse.ResizeTo(5);
  ellipse[0] = x0; // ellipse's "x" center
  ellipse[1] = y0; // ellipse's "y" center
  ellipse[2] = a; // ellipse's "semimajor" axis along "x"
  ellipse[3] = b; // ellipse's "semiminor" axis along "y"
  ellipse[4] = theta; // ellipse's axes rotation angle (in degrees)

  return ellipse;
}

//
// creates a test TGraph with an ellipse
//
TGraph *TestGraphDLSF(bool randomize = false) {
  int i;

  // define the test ellipse
  double x0 = 4; // ellipse's "x" center
  double y0 = 3; // ellipse's "y" center
  double a = 2; // ellipse's "semimajor" axis along "x" (> 0)
  double b = 1; // ellipse's "semiminor" axis along "y" (> 0)
  double theta = 100; // ellipse's axes rotation angle (-45 ... 135 degrees)

  // gRandom->SetSeed(0);
  if (randomize) {
    x0 = 10.0 - 20.0 * gRandom->Rndm();
    y0 = 10.0 - 20.0 * gRandom->Rndm();
    a = 0.5 + 4.5 * gRandom->Rndm();
    b = 0.5 + 4.5 * gRandom->Rndm();
    theta = 180.0 - 360.0 * gRandom->Rndm();
  }

  const int n = 100; // number of points
  double x[n], y[n];
  double dt = TMath::TwoPi() / double(n);
  double tmp;
  theta *= TMath::PiOver2() / 90.0; // degrees -> radians
  for (i = 0; i < n; i++) {
    x[i] = a * (std::cos(dt * double(i)) + 0.1 * gRandom->Rndm() - 0.05);
    y[i] = b * (std::sin(dt * double(i)) + 0.1 * gRandom->Rndm() - 0.05);
    // rotate the axes
    tmp = x[i];
    x[i] = x[i] * std::cos(theta) - y[i] * std::sin(theta);
    y[i] = y[i] * std::cos(theta) + tmp * std::sin(theta);
    // shift the center
    x[i] += x0;
    y[i] += y0;
  }

  // create the test TGraph
  TGraph *g = ((TGraph *)(gROOT->FindObject("g")));
  if (g) delete g;
  g = new TGraph(n, x, y);
  g->SetNameTitle("g", "test ellipse");

  return g;
}

//
// "ROOT Script" entry point (the same name as the "filename's base")
//
void fitEllipseTGraphDLSF(TGraph *g = ((TGraph *)0))
{
  if (!g) g = TestGraphDLSF(true); // create a "random" ellipse

  // fit the TGraph
  TVectorD conic = fit_ellipse(g);
  TVectorD ellipse = ConicToParametric(conic);

#if 1 /* 0 or 1 */
  if ( ellipse.GetNrows() == 5 ) {
    std::cout << std::endl;
    std::cout << "x0 = " << ellipse[0] << std::endl;
    std::cout << "y0 = " << ellipse[1] << std::endl;
    std::cout << "a = " << ellipse[2] << std::endl;
    std::cout << "b = " << ellipse[3] << std::endl;
    std::cout << "theta = " << ellipse[4] << std::endl;
    std::cout << std::endl;
  }
#endif /* 0 or 1 */

#if 1 /* 0 or 1 */
  // draw everything
  TCanvas *c = ((TCanvas *)(gROOT->GetListOfCanvases()->FindObject("c")));
  if (c) { c->Clear(); } else { c = new TCanvas("c", "c"); }
  c->SetGrid(1, 1);
  g->Draw("A*");
  if ( ellipse.GetNrows() == 5 ) {
    TEllipse *e = new TEllipse(ellipse[0], ellipse[1], // "x0", "y0"
                               ellipse[2], ellipse[3], // "a", "b"
                               0, 360,
                               ellipse[4]); // "theta" (in degrees)
    e->SetFillStyle(0); // hollow
    e->Draw();
  }
  c->Modified(); c->Update(); // make sure it's really drawn
#endif /* 0 or 1 */

  return;
}

// end of file fitEllipseTGraphDLSF.cxx by Silesius Anonymus
