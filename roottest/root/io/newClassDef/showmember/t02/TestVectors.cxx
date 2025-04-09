// @(#)root/test:$Id$
// Author: Peter Malzacher   19/06/99

#ifndef __CINT__
#include <Riostream.h>
#include <TMath.h>
#include <TVector3.h>
#include <TLorentzVector.h>
#include <TRotation.h>
#include <TLorentzRotation.h>
#include <assert.h>
#endif

Double_t DEPS=1.0e-14;
Double_t FEPS=1.0e-6;

Bool_t approx(Double_t a, Double_t b, Double_t eps) {
  Double_t diff = TMath::Abs(a-b);
  Bool_t OK = Bool_t(diff < eps);
  if (OK) return OK;
  printf(" a = %.18g, b= %.18g, diff = %.18g\n",a,b,diff);
  return kFALSE;
}

Bool_t test(const TVector3 &p, Double_t x, Double_t y, Double_t z, Double_t eps) {
  Bool_t OK = Bool_t( approx(p.X(), x, eps) &&
                      approx(p.Y(), y, eps) &&
                      approx(p.Z(), z, eps) );
  if (OK) return OK;
  //p.Dump();
  printf("px = %.18g, py= %.18g, pz = %.18g, eps = %.18g\n",p.X(),p.Y(),p.Z(),eps);
  printf("x = %.18g, y= %.18g, z = %.18g, eps = %.18g\n",x,y,z,eps);
  return kFALSE;

}

Bool_t test(const TLorentzVector & p, Double_t x, Double_t y, Double_t z, Double_t e, Double_t eps) {
  Bool_t OK = Bool_t( approx(p.X(), x, eps) &&
                      approx(p.Y(), y, eps) &&
                      approx(p.Z(), z, eps) &&
                      approx(p.T(), e, eps));
  if (OK) return OK;
  //p.Dump();
  printf("px = %.18g, py= %.18g, pz = %.18g, pe = %.18g, eps = %.18g\n",p.X(),p.Y(),p.Z(),p.E(),eps);
  printf("x = %.18g, y= %.18g, z = %.18g, e = %.18g, eps = %.18g\n",x,y,z,e,eps);
  return kFALSE;
}

Bool_t test(const TLorentzVector & p, const TLorentzVector & q, Double_t eps) {
  Bool_t OK =Bool_t( approx(p.X(), q.X(), eps) &&
                     approx(p.Y(), q.Y(), eps) &&
                     approx(p.Z(), q.Z(), eps) &&
                     approx(p.T(), q.T(), eps));
  if (OK) return OK;
  //p.Dump();
  //q.Dump();
  printf("px = %.18g, py= %.18g, pz = %.18g, pe = %.18g, eps = %.18g\n",p.X(),p.Y(),p.Z(),p.E(),eps);
  printf("qx = %.18g, qy= %.18g, qz = %.18g, qe = %.18g, eps = %.18g\n",q.X(),q.Y(),q.Z(),q.E(),eps);
  printf("eps = %.18g\n",eps);
  return kFALSE;
}

Double_t SQR(Double_t x) {return x*x;}

int TestVector3();
int TestLorentzVector();
int TestRotation();

int TestVectors()
{
   int t1 = TestVector3();
   int t2 = TestLorentzVector();
   int t3 = TestRotation();
   return t1+t2+t3;
}


int TestVector3() {

// test constructors:

  TVector3 d0; assert( test(d0, 0.0, 0.0, 0.0, DEPS) );
  TVector3 f0; assert( test(f0, 0.0, 0.0, 0.0, FEPS) );

  TVector3 d1(1.0); assert( test(d1, 1.0, 0.0, 0.0, DEPS) ) ;
  TVector3 f1(1.0); assert( test(f1, 1.0, 0.0, 0.0, FEPS) ) ;

  TVector3 d2(1.0, 1.0); assert( test(d2, 1.0, 1.0, 0.0, DEPS) ) ;
  TVector3 f2(1.0, 1.0); assert( test(f2, 1.0, 1.0, 0.0, FEPS) ) ;
  TVector3 d3(1.0, 1.0, 1.0); assert( test(d3, 1.0, 1.0, 1.0, DEPS) ) ;
  TVector3 f3(1.0, 1.0, 1.0); assert( test(f3, 1.0, 1.0, 1.0, FEPS) ) ;


  TVector3 d4(f3); assert( test(d4, 1.0, 1.0, 1.0, DEPS) ) ;
  TVector3 f4(d3); assert( test(f4, 1.0, 1.0, 1.0, FEPS) ) ;


// test assignment:

  d4 = d1;  assert( test(d4, 1.0, 0.0, 0.0, DEPS) ) ;
  f4 = f1;  assert( test(f4, 1.0, 0.0, 0.0, FEPS) ) ;
  d4 = f1;  assert( test(d4, 1.0, 0.0, 0.0, DEPS) ) ;
  f4 = d1;  assert( test(f4, 1.0, 0.0, 0.0, FEPS) ) ;


// test addition:

  d4 = d1 + d2; assert( test(d4, 2.0, 1.0, 0.0, DEPS) ) ;
  d4 = f1 + d2; assert( test(d4, 2.0, 1.0, 0.0, FEPS) ) ;
  d4 = d1 + f2; assert( test(d4, 2.0, 1.0, 0.0, FEPS) ) ;
  d4 = f1 + f2; assert( test(d4, 2.0, 1.0, 0.0, FEPS) ) ;
  d4 += d3; assert( test(d4, 3.0, 2.0, 1.0, FEPS) ) ;
  d4 += f3; assert( test(d4, 4.0, 3.0, 2.0, FEPS) ) ;
  f4 = d1 + d2; assert( test(f4, 2.0, 1.0, 0.0, FEPS) ) ;
  f4 = f1 + d2; assert( test(f4, 2.0, 1.0, 0.0, FEPS) ) ;
  f4 = d1 + f2; assert( test(f4, 2.0, 1.0, 0.0, FEPS) ) ;
  f4 = f1 + f2; assert( test(f4, 2.0, 1.0, 0.0, FEPS) ) ;
  f4 += d3; assert( test(f4, 3.0, 2.0, 1.0, FEPS) ) ;
  f4 += f3; assert( test(f4, 4.0, 3.0, 2.0, FEPS) ) ;

// test subtraction

  d4 -= d3; assert( test(d4, 3.0, 2.0, 1.0, FEPS) ) ;
  d4 -= f3; assert( test(d4, 2.0, 1.0, 0.0, FEPS) ) ;
  f4 -= d3; assert( test(f4, 3.0, 2.0, 1.0, FEPS) ) ;
  f4 -= f3; assert( test(f4, 2.0, 1.0, 0.0, FEPS) ) ;
  d4 = d1 - d2; assert( test(d4, 0.0, -1.0, 0.0, DEPS) ) ;
  d4 = f1 - d2; assert( test(d4, 0.0, -1.0, 0.0, FEPS) ) ;
  d4 = d1 - f2; assert( test(d4, 0.0, -1.0, 0.0, FEPS) ) ;
  d4 = f1 - f2; assert( test(d4, 0.0, -1.0, 0.0, FEPS) ) ;
  f4 = d1 - d2; assert( test(f4, 0.0, -1.0, 0.0, FEPS) ) ;
  f4 = f1 - d2; assert( test(f4, 0.0, -1.0, 0.0, FEPS) ) ;
  f4 = d1 - f2; assert( test(f4, 0.0, -1.0, 0.0, FEPS) ) ;
  f4 = f1 - f2; assert( test(f4, 0.0, -1.0, 0.0, FEPS) ) ;

// test unary minus:

  assert( test(-d3, -1.0, -1.0, -1.0, DEPS) ) ;
  assert( test(-f3, -1.0, -1.0, -1.0, FEPS) ) ;
  assert( test(-d1, -1.0, 0.0, 0.0, DEPS) ) ;
  assert( test(-f1, -1.0, 0.0, 0.0, FEPS) ) ;

// test scaling:

  assert( test(d3*2.0, 2.0, 2.0, 2.0, DEPS) ) ;
  assert( test(2.0*d3, 2.0, 2.0, 2.0, DEPS) ) ;
  assert( test(d1*2.0, 2.0, 0.0, 0.0, DEPS) ) ;
  assert( test(2.0*d1, 2.0, 0.0, 0.0, DEPS) ) ;
  assert( test(f3*2.0f, 2.0, 2.0, 2.0, FEPS) ) ;
  assert( test(2.0f*f3, 2.0, 2.0, 2.0, FEPS) ) ;
  assert( test(f1*2.0f, 2.0, 0.0, 0.0, FEPS) ) ;
  assert( test(2.0f*f1, 2.0, 0.0, 0.0, FEPS) ) ;
  assert( test(d4*=2.0, 0.0, -2.0, 0.0, FEPS) ) ;
  assert( test(f4*=2.0, 0.0, -2.0, 0.0, FEPS) ) ;

// testing scalar and vector product:

  assert( approx(d4*d1, 0.0, DEPS) ) ;
  assert( approx(d4*f1, 0.0, FEPS) ) ;
  assert( approx(f4*d1, 0.0, FEPS) ) ;
  assert( approx(f4*f1, 0.0, FEPS) ) ;
  assert( approx(d4.Dot(d1), 0.0, DEPS) ) ;
  assert( approx(d4.Dot(f1), 0.0, FEPS) ) ;
  assert( approx(f4.Dot(d1), 0.0, FEPS) ) ;
  assert( approx(f4.Dot(f1), 0.0, FEPS) ) ;
  assert( approx(d4*d2, -2.0, DEPS) ) ;
  assert( approx(d4*f2, -2.0, FEPS) ) ;
  assert( approx(f4*d2, -2.0, FEPS) ) ;
  assert( approx(f4*f2, -2.0, FEPS) ) ;
  assert( approx(d4.Dot(d2), -2.0, DEPS) ) ;
  assert( approx(d4.Dot(f2), -2.0, FEPS) ) ;
  assert( approx(f4.Dot(d2), -2.0, FEPS) ) ;
  assert( approx(f4.Dot(f2), -2.0, FEPS) ) ;
  d4 = d1.Cross(d2); assert( test(d4, 0.0, 0.0, 1.0, DEPS) ) ;
  d4 = d2.Cross(d1); assert( test(d4, 0.0, 0.0, -1.0, DEPS) ) ;
  f4 = f1.Cross(d2); assert( test(f4, 0.0, 0.0, 1.0, FEPS) ) ;
  f4 = d2.Cross(f1); assert( test(f4, 0.0, 0.0, -1.0, FEPS) ) ;

// testing ptot and pt:

  d4 = d1 + f2 + d3;
  f4 = d1 + f2 + d3;
  assert( approx(d4.Mag2(), 14.0, FEPS) ) ;
  assert( approx(d4.Mag(), TMath::Sqrt(14.0), FEPS) ) ;
  assert( approx(d4.Perp2(), 13.0, FEPS) ) ;
  assert( approx(d4.Perp(), TMath::Sqrt(13.0), FEPS) ) ;
  assert( approx(f4.Mag2(), 14.0, FEPS) ) ;
  assert( approx(f4.Mag(), TMath::Sqrt(14.0), FEPS) ) ;
  assert( approx(f4.Perp2(), 13.0, FEPS) ) ;
  assert( approx(f4.Perp(), TMath::Sqrt(13.0), FEPS) ) ;

// testing angles:

  d4 = d2 - 2.0 * d1;
  f4 = d2 - 2.0f * f1;
  assert( approx(d1.Phi(), 0.0, DEPS) ) ;
  assert( approx(d1.Theta(), TMath::Pi()/2., DEPS) ) ;
  assert( approx(d1.CosTheta(), 0.0, DEPS) ) ;
  assert( approx(d2.Phi(), TMath::Pi()/2.*0.5, DEPS) ) ;
  assert( approx(d2.Theta(), TMath::Pi()/2., DEPS) ) ;
  assert( approx(d2.CosTheta(), 0.0, DEPS) ) ;
  assert( approx(((-d2)).Phi(), -3.0*TMath::Pi()/2.*0.5, DEPS) ) ;
  assert( approx(d4.Phi(), 3.0*TMath::Pi()/2.*0.5, DEPS) ) ;

  assert( approx(f1.Phi(), 0.0, FEPS) ) ;
  assert( approx(f1.Theta(), TMath::Pi()/2., FEPS) ) ;
  assert( approx(f1.CosTheta(), 0.0, FEPS) ) ;
  assert( approx(f2.Phi(), TMath::Pi()/2.*0.5, FEPS) ) ;
  assert( approx(f2.Theta(), TMath::Pi()/2., FEPS) ) ;
  assert( approx(f2.CosTheta(), 0.0, FEPS) ) ;
  assert( approx(((-f2)).Phi(), -3.0*TMath::Pi()/2.*0.5, FEPS) ) ;
  assert( approx(f4.Phi(), 3.0*TMath::Pi()/2.*0.5, FEPS) ) ;

  d4 = d3 - d1; assert( approx(d4.Theta(), TMath::Pi()/2.*0.5, DEPS) ) ;
  assert( approx(((-d4)).Theta(), 3.0*TMath::Pi()/2.*0.5, DEPS) ) ;
  assert( approx(((-d4)).CosTheta(), -TMath::Sqrt(0.5), DEPS) ) ;
  d4 = d3 - d2; assert( approx(d4.Theta(), 0.0, DEPS) ) ;
  assert( approx(d4.CosTheta(), 1.0, DEPS) ) ;
  assert( approx(((-d4)).Theta(), TMath::Pi(), DEPS) ) ;
  assert( approx(((-d4)).CosTheta(), -1.0, DEPS) ) ;
  f4 = d3 - d1; assert( approx(f4.Theta(), TMath::Pi()/2.*0.5, FEPS) ) ;
  assert( approx(((-f4)).Theta(), 3.0*TMath::Pi()/2.*0.5, FEPS) ) ;
  assert( approx(((-f4)).CosTheta(), -TMath::Sqrt(0.5), FEPS) ) ;
  f4 = d3 - d2; assert( approx(f4.Theta(), 0.0, FEPS) ) ;
  assert( approx(f4.CosTheta(), 1.0, FEPS) ) ;
  assert( approx(((-f4)).Theta(), TMath::Pi(), FEPS) ) ;
  assert( approx(((-f4)).CosTheta(), -1.0, FEPS) ) ;

  d4 = d2 - 2.0*d1; assert( approx(d4.Angle(d2), TMath::Pi()/2., DEPS) ) ;
  f4 = d2 - 2.0*d1; assert( approx(f4.Angle(f2), TMath::Pi()/2., FEPS) ) ;

// testing rotations

  d4 = d1;
  d4.RotateZ(TMath::Pi()/2.); assert( test(d4, 0.0, 1.0, 0.0, DEPS) ) ;
  d4.RotateY(25.3); assert( test(d4, 0.0, 1.0, 0.0, DEPS) ) ;
  d4.RotateZ(TMath::Pi()/2.); assert( test(d4, -1.0, 0.0, 0.0, DEPS) ) ;
  d4.RotateY(TMath::Pi()/2.); assert( test(d4, 0.0, 0.0, 1.0, DEPS) ) ;
  d4.RotateZ(2.6); assert( test(d4, 0.0, 0.0, 1.0, DEPS) ) ;
  d4.RotateY(TMath::Pi()*0.25);
  assert( test(d4, TMath::Sqrt(0.5), 0.0, TMath::Sqrt(0.5), DEPS) ) ;
  f4 = f1;
  f4.RotateZ(TMath::Pi()/2.); assert( test(f4, 0.0, 1.0, 0.0, FEPS) ) ;
  f4.RotateY(25.3); assert( test(f4, 0.0, 1.0, 0.0, FEPS) ) ;
  f4.RotateZ(TMath::Pi()/2.); assert( test(f4, -1.0, 0.0, 0.0, FEPS) ) ;
  f4.RotateY(TMath::Pi()/2.); assert( test(f4, 0.0, 0.0, 1.0, FEPS) ) ;
  f4.RotateZ(2.6); assert( test(f4, 0.0, 0.0, 1.0, FEPS) ) ;
  f4.RotateY(TMath::Pi()*0.25);
  assert( test(f4, TMath::Sqrt(0.5), 0.0, TMath::Sqrt(0.5), FEPS) ) ;

  d4 = d1;

  d4.Rotate(d4.Angle(d3), d4.Cross(d3));

  d4 *= d3.Mag();

  assert( test(d4, 1.0, 1.0, 1.0, DEPS) ) ;
  d4 = d1;
  d4.Rotate(0.23, d4.Cross(d3));
  assert( approx(d4.Angle(d1), 0.23, DEPS) ) ;
  f4 = d1;
  f4.Rotate(f4.Angle(d3), f4.Cross(d3));
  f4 *= f3.Mag();
  assert( test(f4, 1.0, 1.0, 1.0, FEPS) ) ;
  f4 = f1;
  f4.Rotate(0.23, f4.Cross(d3));
  assert( approx(f4.Angle(f1), 0.23, FEPS) ) ;
  assert( approx(f4.Angle(d3), f1.Angle(d3) - 0.23, FEPS) ) ;

// test rotation maticies:

  d4 = d1;

  TRotation r0, r1, r2, r3, r4, r5;
  r1.RotateZ(TMath::Pi()/2.);
  r2.RotateY(TMath::Pi()/2.);
  r4.Rotate(d4.Angle(d3), d4.Cross(d3));
  r5.Rotate(0.23, d4.Cross(d3));
  d4 = r4.Inverse() * d3;
  assert( test(d4, d3.Mag(), 0.0, 0.0, DEPS) ) ;
  d4 = r5 * d3;
  assert( approx(d1.Angle(d4), d1.Angle(d3)+0.23, DEPS) ) ;
  f4 = r4.Inverse() * f3;
  assert( test(f4, f3.Mag(), 0.0, 0.0, FEPS) ) ;
  f4 = r5 * d3;
  assert( approx(d1.Angle(f4), f1.Angle(f3)+0.23, FEPS) ) ;
  r5 = r2 * r1 * r3.Inverse() * r0 * r0.Inverse();
  d4 = d3;
  d4 *= r3.Inverse();
  d4 *= r1;
  d4 *= r2;
  assert( test(d4, 1.0, 1.0, 1.0, DEPS) ) ;
  r5.Invert();
  d4 = r5 * d4;
  assert( test(d4, 1.0, 1.0, 1.0, DEPS) ) ;
  d1 = d2 = TVector3(1.0, -0.5, 2.1);
  d3 = TVector3(-0.3, 1.1, 1.5);
  d4 = d3.Unit();
  d4 *= d3.Mag();

  assert( test(d4, d3.X(), d3.Y(), d3.Z(), DEPS) ) ;
  r0.Rotate(0.10, d1.Cross(d3));
  d1 *= r0;
  assert( approx(d1.Angle(d3), d2.Angle(d3)-0.1, DEPS) ) ;
  assert( approx(d1.Angle(d2), 0.1, DEPS) ) ;

  return 0;

}


int TestLorentzVector() {

  TVector3 f3x(1.0), f3y(0.0, 1.0), f3z(0.0, 0.0, 1.0);
  TVector3 d30, d3x(1.0), d3y(0.0, 1.0), d3z(0.0, 0.0, 1.0);

// test constructors:

  TLorentzVector d0;
  assert( test(d0, 0.0, 0.0, 0.0, 0.0, DEPS) ) ;
  TLorentzVector d1(d3x, 1.0);
  assert( test(d1, 1.0, 0.0, 0.0, 1.0, DEPS) ) ;
  TLorentzVector d2(d3x + d3y, TMath::Sqrt(2.0));
  assert( test(d2, 1.0, 1.0, 0.0, TMath::Sqrt(2.0), DEPS) ) ;
  TLorentzVector d3(d3z + d2.Vect(), TMath::Sqrt(3.0));
  assert( test(d3, 1.0, 1.0, 1.0, TMath::Sqrt(Double_t(3.0)), DEPS) ) ;
  TLorentzVector d4(0.0, 0.0, 0.0, 1.0);
  assert( test(d4,0.0, 0.0, 0.0, 1.0, DEPS) ) ;
  TLorentzVector d5(f3x, f3x.Mag()); assert( test(d5, d1, FEPS) ) ;
  TLorentzVector d6(d3x+f3y, ((d3x+f3y)).Mag());
  assert( test(d6, d2, FEPS) ) ;
  TLorentzVector d7(f3x+f3y+f3z, ((f3x+f3y+f3z)).Mag());
  assert( test(d7, d3, FEPS) ) ;

  TLorentzVector f0; assert( test(f0, 0.0, 0.0, 0.0, 0.0, FEPS) ) ;
  TLorentzVector f1(f3x, 1.0);
  assert( test(f1, 1.0, 0.0, 0.0, 1.0, FEPS) ) ;
  TLorentzVector f2(f3x + f3y, TMath::Sqrt(2.0));
  assert( test(f2, 1.0, 1.0, 0.0, TMath::Sqrt(2.0), FEPS) ) ;
  TLorentzVector f3(f3z + f2.Vect(), TMath::Sqrt(3.0));
  assert( test(f3, 1.0, 1.0, 1.0, TMath::Sqrt(3.0), FEPS) ) ;
  TLorentzVector f4(0.0, 0.0, 0.0, 1.0);
  assert( test(f4,0.0, 0.0, 0.0, 1.0, FEPS) ) ;
  TLorentzVector f5(d3x, d3x.Mag()); assert( test(f5, f1, FEPS) ) ;
  TLorentzVector f6(f3x+d3y, ((f3x+d3y)).Mag());
  assert( test(f6, f2, FEPS) ) ;
  TLorentzVector f7(d3x+d3y+d3z, ((d3x+d3y+d3z)).Mag());
  assert( test(f7, f3, FEPS) ) ;

  TLorentzVector d8(f7); assert( test(d8, d7, FEPS) ) ;
  TLorentzVector d9(d7); assert( test(d9, d7, DEPS) ) ;
  TLorentzVector f8(f7); assert( test(f8, d7, FEPS) ) ;
  TLorentzVector f9(d7); assert( test(f9, d7, FEPS) ) ;

  TLorentzVector d10(1.0, 1.0, 1.0, TMath::Sqrt(3.0));
  assert( test(d10, d7, FEPS) ) ;
  TLorentzVector f10(1.0, 1.0, 1.0, TMath::Sqrt(3.0));
  assert( test(f10, f7, FEPS) ) ;

  TLorentzVector d11(d3x+d3y+d3z, 1.0);
  assert( test(d11, 1.0, 1.0, 1.0, 1.0, DEPS) ) ;
  TLorentzVector f11(d3x+d3y+d3z, 1.0);
  assert( test(f11, 1.0, 1.0, 1.0, 1.0, FEPS) ) ;

// testing assignment

  d6 = d7; assert( test(d6, d7, DEPS) ) ;
  d6 = f7; assert( test(d6, d7, FEPS) ) ;
  f6 = d7; assert( test(f6, f7, FEPS) ) ;
  f6 = f7; assert( test(f6, f7, FEPS) ) ;

  //testing addition and subtraction:

  d11 = d3 + d7 + f3;
  assert( test(d11, 3.0, 3.0, 3.0, TMath::Sqrt(27.0), FEPS) ) ;
  f11 = d3 + d7 + f3;
  assert( test(f11, 3.0, 3.0, 3.0, TMath::Sqrt(27.0), FEPS) ) ;
  d11 += d3;
  assert( test(d11, 4.0, 4.0, 4.0, TMath::Sqrt(48.0), FEPS) ) ;
  f11 += f3;
  assert( test(f11, 4.0, 4.0, 4.0, TMath::Sqrt(48.0), FEPS) ) ;
  d11 = d3 + d7 - f3;
  assert( test(d11, 1.0, 1.0, 1.0, TMath::Sqrt(3.0), FEPS) ) ;
  assert( test(-d11, -1.0, -1.0, -1.0, -TMath::Sqrt(3.0), FEPS) ) ;
  f11 = d3 + f7 - d3;
  assert( test(f11, 1.0, 1.0, 1.0, TMath::Sqrt(3.0), FEPS) ) ;
  assert( test(-f11, -1.0, -1.0, -1.0, -TMath::Sqrt(3.0), FEPS) ) ;
  d11 -= d3;
  assert( test(d11, 0.0, 0.0, 0.0, 0.0, FEPS) ) ;
  f11 -= f3;
  assert( test(f11, 0.0, 0.0, 0.0, 0.0, FEPS) ) ;

  d11 = TLorentzVector(1.0, 2.0, 3.0, 4.0);
  d11 *= 2.;
  assert( test(d11, 2.0, 4.0, 6.0, 8.0, DEPS) ) ;
  d11 = 2.*TLorentzVector(1.0, 2.0, 3.0, 4.0);
  assert( test(d11, 2.0, 4.0, 6.0, 8.0, DEPS) ) ;
  d11 = TLorentzVector(1.0, 2.0, 3.0, 4.0)*2.;
  assert( test(d11, 2.0, 4.0, 6.0, 8.0, DEPS) ) ;

// testing scalar products:

  assert( approx(d1 * d2, TMath::Sqrt(2.0)-1.0, DEPS) ) ;
  assert( approx(d3.Dot(d7), 0.0, FEPS) ) ;
  assert( approx(d2 * f1, TMath::Sqrt(2.0)-1.0, FEPS) ) ;
  assert( approx(f3.Dot(d7), 0.0, FEPS) ) ;

// testing components:

  d11 = TLorentzVector(1.0, 1.0, 1.0, TMath::Sqrt(7.0));
  assert( approx(d11.Mag2(), 4.0, DEPS) ) ;
  assert( approx(d11.Mag(), 2.0, DEPS) ) ;
  assert( approx(TVector3(d11.Vect()).Mag2(), 3.0, DEPS) ) ;
  assert( approx(TVector3(d11.Vect()).Mag(), TMath::Sqrt(3.0), DEPS) ) ;
  assert( approx(d11.Perp2(), 2.0, DEPS) ) ;
  assert( approx(d11.Perp(), TMath::Sqrt(2.0), DEPS) ) ;
  f11 = TLorentzVector(1.0, 1.0, 1.0, TMath::Sqrt(7.0));
  assert( approx(f11.Mag2(), 4.0, FEPS) ) ;
  assert( approx(f11.Mag(), 2.0, FEPS) ) ;
  assert( approx(f11.Vect().Mag2(), 3.0, FEPS) ) ;
  assert( approx(f11.Vect().Mag(), TMath::Sqrt(3.0), FEPS) ) ;
  assert( approx(f11.Perp2(), 2.0, FEPS) ) ;
  assert( approx(f11.Perp(), TMath::Sqrt(2.0), FEPS) ) ;

// testing boosts:

  d5 = d3 = d1 = TLorentzVector(1.0, 2.0, -1.0, 3.0);
  d6 = d4 = d2 = TLorentzVector(-1.0, 1.0, 2.0, 4.0);
  Double_t M = ((d1 + d2)).Mag();
  Double_t m1 = d1.Mag();
  Double_t m2 = d2.Mag();
  Double_t p2 = (SQR(M)-SQR(m1+m2))*(SQR(M)-SQR(m1-m2))/(4.0*SQR(M));
  d30 = -((d1 + d2)).BoostVector();
  d1.Boost(d30);
  Double_t phi = d1.Phi();
  Double_t theta = d1.Theta();
  d1.RotateZ(-phi);
  d1.RotateY(-theta);
  TRotation r;
  r.RotateZ(-phi);
  TLorentzRotation r1(d30), r2(r), r3, r4, r5;
  r3.RotateY(-theta);
  r4 = r3  * r2 * r1;
  d2 *= r4;
  assert( test(d1, 0.0, 0.0, TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m1)), DEPS) ) ;
  assert( test(d2, 0.0, 0.0, -TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m2)), DEPS) ) ;
  d1.Transform(r4.Inverse());
  assert( test(d1, d3, DEPS) ) ;
  r5 *= r3;
  r5 *= r;
  r5 *= r1;
  r5.Invert();
  d2 *= r5;
  assert( test(d2, d4, DEPS) ) ;
  r4 = r1;
  r4.RotateZ(-phi);
  r4.RotateY(-theta);
  d3 *= r4;
  d4 = r4 * d6;
  assert( test(d3, 0.0, 0.0, TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m1)), DEPS) ) ;
  assert( test(d4, 0.0, 0.0, -TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m2)), DEPS) ) ;
  r5 = r1.Inverse();
  r5 *= r.Inverse();
  r5 *= r3.Inverse();
  d4.Transform(r5);
  d3.Transform(r5);

  assert( test(d4, d6, DEPS) ) ;
  assert( test(d3, d5, DEPS) ) ;

  r5 = r1;
  r5.Transform(r);
  r5.Transform(r3);
  d4.Transform(r5);
  d3.Transform(r5);
  assert( test(d3, 0.0, 0.0, TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m1)), DEPS) ) ;
  assert( test(d4, 0.0, 0.0, -TMath::Sqrt(p2), TMath::Sqrt(p2 + SQR(m2)), DEPS) ) ;

  // beta and gamma

  assert( approx(d3.BoostVector().Mag(), d3.Beta(), DEPS) );
  assert( approx(d4.BoostVector().Mag(), d4.Beta(), DEPS) );

  assert( approx(d3.Gamma(), 1./TMath::Sqrt(1-d3.Beta()*d3.Beta()), DEPS) );
  assert( approx(d4.Gamma(), 1./TMath::Sqrt(1-d4.Beta()*d4.Beta()), DEPS) );

  return 0;
}

//typedef TRotation Rotation;
//typedef TVector3  Vector;

int TestRotation() {

  int i,k;
  double angA=TMath::Pi()/3, angB=TMath::Pi()/4, angC=TMath::Pi()/6;
  double cosA=TMath::Cos(angA), sinA=TMath::Sin(angA);
  double cosB=TMath::Cos(angB), sinB=TMath::Sin(angB);
  double cosC=TMath::Cos(angC), sinC=TMath::Sin(angC);

  TRotation R;                   // default constructor
  assert ( R.XX() == 1 );
  assert ( R.XY() == 0 );
  assert ( R.XZ() == 0 );
  assert ( R.YX() == 0 );
  assert ( R.YY() == 1 );
  assert ( R.YZ() == 0 );
  assert ( R.ZX() == 0 );
  assert ( R.ZY() == 0 );
  assert ( R.ZZ() == 1 );

  assert( R.IsIdentity() );     // isIdentity()

  R = TRotation();               // rotateX()
  R.RotateX(angA);
  assert ( approx(R.XX(), 1,    DEPS) );
  assert ( approx(R.XY(), 0,    DEPS) );
  assert ( approx(R.XZ(), 0,    DEPS) );
  assert ( approx(R.YX(), 0,    DEPS) );
  assert ( approx(R.YY(), cosA, DEPS) );
  assert ( approx(R.YZ(),-sinA, DEPS) );
  assert ( approx(R.ZX(), 0,    DEPS) );
  assert ( approx(R.ZY(), sinA, DEPS) );
  assert ( approx(R.ZZ(), cosA, DEPS) );

  R = TRotation();               // rotateY()
  R.RotateY(angB);
  assert ( approx(R.XX(), cosB, DEPS) );
  assert ( approx(R.XY(), 0,    DEPS) );
  assert ( approx(R.XZ(), sinB, DEPS) );
  assert ( approx(R.YX(), 0,    DEPS) );
  assert ( approx(R.YY(), 1,    DEPS) );
  assert ( approx(R.YZ(), 0,    DEPS) );
  assert ( approx(R.ZX(),-sinB, DEPS) );
  assert ( approx(R.ZY(), 0,    DEPS) );
  assert ( approx(R.ZZ(), cosB, DEPS) );

  R = TRotation();               // rotateZ()
  R.RotateZ(angC);
  assert ( approx(R.XX(), cosC, DEPS) );
  assert ( approx(R.XY(),-sinC, DEPS) );
  assert ( approx(R.XZ(), 0,    DEPS) );
  assert ( approx(R.YX(), sinC, DEPS) );
  assert ( approx(R.YY(), cosC, DEPS) );
  assert ( approx(R.YZ(), 0,    DEPS) );
  assert ( approx(R.ZX(), 0,    DEPS) );
  assert ( approx(R.ZY(), 0,    DEPS) );
  assert ( approx(R.ZZ(), 1,    DEPS) );

  R = TRotation();               // copy constructor
  R.RotateZ(angC);
  R.RotateY(angB);
  R.RotateZ(angA);
  TRotation RR(R);

  assert ( TMath::Abs(RR.XX() - cosA*cosB*cosC + sinA*sinC) < DEPS );
  assert ( TMath::Abs(RR.XY() + cosA*cosB*sinC + sinA*cosC) < DEPS );
  assert ( TMath::Abs(RR.XZ() - cosA*sinB)                  < DEPS );
  assert ( TMath::Abs(RR.YX() - sinA*cosB*cosC - cosA*sinC) < DEPS );
  assert ( TMath::Abs(RR.YY() + sinA*cosB*sinC - cosA*cosC) < DEPS );
  assert ( TMath::Abs(RR.YZ() - sinA*sinB)                  < DEPS );
  assert ( TMath::Abs(RR.ZX() + sinB*cosC)                  < DEPS );
  assert ( TMath::Abs(RR.ZY() - sinB*sinC)                  < DEPS );
  assert ( TMath::Abs(RR.ZZ() - cosB)                       < DEPS );

  RR = TRotation();              // operator=, operator!=, operator==
  assert ( RR != R );
  RR = R;
  assert ( RR == R );

  assert ( R(0,0) == R.XX() );  // operator(i,j)
  assert ( R(0,1) == R.XY() );
  assert ( R(0,2) == R.XZ() );
  assert ( R(1,0) == R.YX() );
  assert ( R(1,1) == R.YY() );
  assert ( R(1,2) == R.YZ() );
  assert ( R(2,0) == R.ZX() );
  assert ( R(2,1) == R.ZY() );
  assert ( R(2,2) == R.ZZ() );

  for(i=0; i<3; i++) {
    for(k=0; k<3; k++) {
      assert ( RR(i,k) == R(i,k) );
    }
  }

  TRotation A, B ,C;                                // operator*=
  A.RotateZ(angA);
  B.RotateY(angB);
  C.RotateZ(angC);
  R  = A; R *= B; R *= C;

  TVector3 V(1,2,3);                                 // operator* (Vector)
  V = R * V;
  assert ( TMath::Abs(V.X()-R.XX()-2.*R.XY()-3.*R.XZ()) < DEPS );
  assert ( TMath::Abs(V.Y()-R.YX()-2.*R.YY()-3.*R.YZ()) < DEPS );
  assert ( TMath::Abs(V.Z()-R.ZX()-2.*R.ZY()-3.*R.ZZ()) < DEPS );

  R = A * B * C;                                  // operator*(Matrix)
  assert ( TMath::Abs(RR.XX() - R.XX()) < DEPS );
  assert ( TMath::Abs(RR.XY() - R.XY()) < DEPS );
  assert ( TMath::Abs(RR.XZ() - R.XZ()) < DEPS );
  assert ( TMath::Abs(RR.YX() - R.YX()) < DEPS );
  assert ( TMath::Abs(RR.YY() - R.YY()) < DEPS );
  assert ( TMath::Abs(RR.YZ() - R.YZ()) < DEPS );
  assert ( TMath::Abs(RR.ZX() - R.ZX()) < DEPS );
  assert ( TMath::Abs(RR.ZY() - R.ZY()) < DEPS );
  assert ( TMath::Abs(RR.ZZ() - R.ZZ()) < DEPS );

  R = C;                                           // transform()
  R.Transform(B);
  R.Transform(A);
  assert ( TMath::Abs(RR.XX() - R.XX()) < DEPS );
  assert ( TMath::Abs(RR.XY() - R.XY()) < DEPS );
  assert ( TMath::Abs(RR.XZ() - R.XZ()) < DEPS );
  assert ( TMath::Abs(RR.YX() - R.YX()) < DEPS );
  assert ( TMath::Abs(RR.YY() - R.YY()) < DEPS );
  assert ( TMath::Abs(RR.YZ() - R.YZ()) < DEPS );
  assert ( TMath::Abs(RR.ZX() - R.ZX()) < DEPS );
  assert ( TMath::Abs(RR.ZY() - R.ZY()) < DEPS );
  assert ( TMath::Abs(RR.ZZ() - R.ZZ()) < DEPS );

  R = RR.Inverse();                                // inverse()
  for(i=0; i<3; i++) {
    for(k=0; k<3; k++) {
      assert ( RR(i,k) == R(k,i) );
    }
  }

  R.Invert();                                      // invert()
  assert ( RR == R );

  R = TRotation();                                  // rotateAxes()
  R.RotateAxes( TVector3(RR.XX(), RR.YX(), RR.ZX()),
                TVector3(RR.XY(), RR.YY(), RR.ZY()),
                TVector3(RR.XZ(), RR.YZ(), RR.ZZ()) );
  assert ( RR == R );

  double ang=2.*TMath::Pi()/9.;                           // rotate()
  R = TRotation();
  R.Rotate(ang, V);

  RR = TRotation();
  RR.RotateZ(-(V.Phi()));
  RR.RotateY(-(V.Theta()));
  RR.RotateZ(ang);
  RR.RotateY(V.Theta());
  RR.RotateZ(V.Phi());

  assert ( TMath::Abs(RR.XX() - R.XX()) < DEPS );
  assert ( TMath::Abs(RR.XY() - R.XY()) < DEPS );
  assert ( TMath::Abs(RR.XZ() - R.XZ()) < DEPS );
  assert ( TMath::Abs(RR.YX() - R.YX()) < DEPS );
  assert ( TMath::Abs(RR.YY() - R.YY()) < DEPS );
  assert ( TMath::Abs(RR.YZ() - R.YZ()) < DEPS );
  assert ( TMath::Abs(RR.ZX() - R.ZX()) < DEPS );
  assert ( TMath::Abs(RR.ZY() - R.ZY()) < DEPS );
  assert ( TMath::Abs(RR.ZZ() - R.ZZ()) < DEPS );

  TVector3 Vu = V.Unit();                           // getAngleAxis
  R.AngleAxis(ang, V);
  assert ( TMath::Abs(ang   - 2.*TMath::Pi()/9.) < DEPS );
  assert ( TMath::Abs(V.X() - Vu.X())     < DEPS );
  assert ( TMath::Abs(V.Y() - Vu.Y())     < DEPS );
  assert ( TMath::Abs(V.Z() - Vu.Z())     < DEPS );

  assert ( TMath::Abs(RR.PhiX()-TMath::ATan2(RR.YX(),RR.XX())) < DEPS ); // phiX()
  assert ( TMath::Abs(RR.PhiY()-TMath::ATan2(RR.YY(),RR.XY())) < DEPS ); // phiY()
  assert ( TMath::Abs(RR.PhiZ()-TMath::ATan2(RR.YZ(),RR.XZ())) < DEPS ); // phiZ()

  assert ( TMath::Abs(RR.ThetaX()-TMath::ACos(RR.ZX())) < DEPS );        // thetaX()
  assert ( TMath::Abs(RR.ThetaY()-TMath::ACos(RR.ZY())) < DEPS );        // thetaY()
  assert ( TMath::Abs(RR.ThetaZ()-TMath::ACos(RR.ZZ())) < DEPS );        // thetaZ()

  return 0;
}
