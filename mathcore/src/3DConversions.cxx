// @(#)root/mathcore:$Name:  $:$Id: 3DConversions.cxx,v 1.1 2005/09/18 17:33:47 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005, LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Source file for something else
//
// Created by: Mark Fischler and Walter Brown Thurs July 7, 2005
//
// Last update: $Id: 3DConversions.cxx,v 1.1 2005/09/18 17:33:47 brun Exp $
//

// TODO - For now, all conversions are grouped in this one compilation unit.
//        The intention is to seraparte them into a few .cpp files instead,
//        so that users needing one form need not incorporate code for them all.


#include "Math/GenVector/3DConversions.h"

#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

#include <cmath>


namespace ROOT {
namespace Math {
namespace gv_detail {

enum Rotation3DMatrixIndex
{ XX = Rotation3D::XX, XY = Rotation3D::XY, XZ = Rotation3D::XZ
, YX = Rotation3D::YX, YY = Rotation3D::YY, YZ = Rotation3D::YZ
, ZX = Rotation3D::ZX, ZY = Rotation3D::ZY, ZZ = Rotation3D::ZZ
};

// ----------------------------------------------------------------------
// conversions from Rotation3D

void convert( Rotation3D const & from, AxisAngle   & to)
{
  double m[9];
  from.GetComponents(m, m+9);

  const double  Uz = m[YX] - m[XY];
  const double  Uy = m[XZ] - m[ZX];
  const double  Ux = m[ZY] - m[YZ];

  AxisAngle::AxisVector u;

  if ( (Uz==0) && (Uy==0) && (Ux==0) ) {
    if        ( m[ZZ]>0 ) {
      u.SetCoordinates(0,0,1);
    } else if ( m[YY]>0 ) {
      u.SetCoordinates(0,1,0);
    } else {
      u.SetCoordinates(1,0,0);
    }
  } else {
    u.SetCoordinates( Ux, Uy, Uz );
  }
  //to.SetAxis(u); // Note:  SetAxis does normalize

  static const double pi=3.14159265358979323;

  double angle;
  const double cosdelta = (m[XX] + m[YY] + m[ZZ] - 1.0) / 2.0;
  if (cosdelta > 1.0) {
    angle = 0;
  } else if (cosdelta < -1.0) {
    angle = pi;
  } else {
    angle = std::acos( cosdelta );
  }
  //to.SetAngle(angle);
  to.SetComponents(u, angle);
  to.Rectify();

} // convert to AxisAngle

static void correctByPi ( double& psi, double& phi ) {
  static const double pi=3.14159265358979323;
  if (psi > 0) {
    psi -= pi;
  } else {
    psi += pi;
  }
  if (phi > 0) {
    phi -= pi;
  } else {
    phi += pi;
  }  
}

void convert( Rotation3D const & from, EulerAngles & to)
{

  // Mathematical justification appears in eulerAngleComputations.ps

  double r[9];
  from.GetComponents(r,r+9);
    
  double phi, theta, psi;
  double psiPlusPhi, psiMinusPhi;
  static const double pi=3.14159265358979323;
  
  theta = (std::fabs(r[ZZ]) <= 1.0) ? std::acos(r[ZZ]) :
  	            (r[ZZ]  >  0.0) ?     0            : pi;
  
  double cosTheta = r[ZZ];
  if (cosTheta > 1)  cosTheta = 1;
  if (cosTheta < -1) cosTheta = -1;

  // Compute psi +/- phi:  
  // Depending on whether cosTheta is positive or negative and whether it
  // is less than 1 in absolute value, different mathematically equivalent
  // expressions are numerically stable.
  if (cosTheta == 1) {
    psiPlusPhi = atan2 ( r[XY] - r[YX], r[XX] + r[YY] );
    psiMinusPhi = 0;     
  } else if (cosTheta >= 0) {
    psiPlusPhi = atan2 ( r[XY] - r[YX], r[XX] + r[YY] );
    double s = -r[XY] - r[YX]; // sin (psi-phi) * (1 - cos theta)
    double c =  r[XX] - r[YY]; // cos (psi-phi) * (1 - cos theta)
    psiMinusPhi = atan2 ( s, c );
  } else if (cosTheta > -1) {
    psiMinusPhi = atan2 ( -r[XY] - r[YX], r[XX] - r[YY] );
    double s = r[XY] - r[YX]; // sin (psi+phi) * (1 + cos theta)
    double c = r[XX] + r[YY]; // cos (psi+phi) * (1 + cos theta)
    psiPlusPhi = atan2 ( s, c );
  } else { // cosTheta == -1
    psiMinusPhi = atan2 ( -r[XY] - r[YX], r[XX] - r[YY] );
    psiPlusPhi = 0;
  }
  
  psi = .5 * (psiPlusPhi + psiMinusPhi); 
  phi = .5 * (psiPlusPhi - psiMinusPhi); 

  // Now correct by pi if we have managed to get a value of psiPlusPhi
  // or psiMinusPhi that was off by 2 pi:

  // set up w[i], all of which would be positive if sin and cosine of
  // psi and phi were positive:
  double w[4];
  w[0] = r[XZ]; w[1] = r[ZX]; w[2] = r[YZ]; w[3] = -r[ZY];

  // find biggest relevant term, which is the best one to use in correcting.
  double maxw = std::fabs(w[0]); 
  int imax = 0;
  for (int i = 1; i < 4; ++i) {
    if (std::fabs(w[i]) > maxw) {
      maxw = std::fabs(w[i]);
      imax = i;
    }
  }
  // Determine if the correction needs to be applied:  The criteria are 
  // different depending on whether a sine or cosine was the determinor: 
  switch (imax) {
    case 0:
      if (w[0] > 0 && psi < 0)               correctByPi ( psi, phi );
      if (w[0] < 0 && psi > 0)               correctByPi ( psi, phi );
      break;
    case 1:
      if (w[1] > 0 && phi < 0)               correctByPi ( psi, phi );
      if (w[1] < 0 && phi > 0)               correctByPi ( psi, phi );
      break;
    case 2:
      if (w[2] > 0 && std::fabs(psi) > pi/2) correctByPi ( psi, phi );    
      if (w[2] < 0 && std::fabs(psi) < pi/2) correctByPi ( psi, phi );    
      break;
    case 3:
      if (w[3] > 0 && std::fabs(phi) > pi/2) correctByPi ( psi, phi );    
      if (w[3] < 0 && std::fabs(phi) < pi/2) correctByPi ( psi, phi );    
      break;
  }          
  
  to.SetComponents( phi, theta, psi );

} // convert to EulerAngles

void convert( Rotation3D const & from, Quaternion  & to)
{
  double m[9];
  from.GetComponents(m, m+9);

  const double d0 =   m[XX] + m[YY] + m[ZZ];
  const double d1 = + m[XX] - m[YY] - m[ZZ];
  const double d2 = - m[XX] + m[YY] - m[ZZ];
  const double d3 = - m[XX] - m[YY] + m[ZZ];

  // these are related to the various q^2 values;
  // choose the largest to avoid dividing two small numbers and losing accuracy.

  if ( d0 >= d1 && d0 >= d2 && d0 >= d3 ) {
    const double q0 = .5*std::sqrt(1+d0);
    const double f  = .25/q0;
    const double q1 = f*(m[ZY]-m[YZ]);
    const double q2 = f*(m[XZ]-m[ZX]);
    const double q3 = f*(m[YX]-m[XY]);
    to.SetComponents(q0,q1,q2,q3);
    to.Rectify();
    return;
 } else if ( d1 >= d2 && d1 >= d3 ) {
    const double q1 = .5*std::sqrt(1+d1);
    const double f  = .25/q1;
    const double q0 = f*(m[ZY]-m[YZ]);
    const double q2 = f*(m[XY]+m[YX]); 
    const double q3 = f*(m[XZ]+m[ZX]);
    to.SetComponents(q0,q1,q2,q3);
    to.Rectify();
    return;
 } else if ( d2 >= d3 ) {
    const double q2 = .5*std::sqrt(1+d2);
    const double f  = .25/q2;
    const double q0 = f*(m[XZ]-m[ZX]);
    const double q1 = f*(m[XY]+m[YX]);
    const double q3 = f*(m[YZ]+m[ZY]);
    to.SetComponents(q0,q1,q2,q3);
    to.Rectify();
    return;
 } else {
    const double q3 = .5*std::sqrt(1+d3);
    const double f  = .25/q3;
    const double q0 = f*(m[YX]-m[XY]);
    const double q1 = f*(m[XZ]+m[ZX]);
    const double q2 = f*(m[YZ]+m[ZY]);
    to.SetComponents(q0,q1,q2,q3);
    to.Rectify();
    return;
  }
}  // convert to Quaternion



// ----------------------------------------------------------------------
// conversions from AxisAngle

void convert( AxisAngle const & from, Rotation3D  & to)
{
  const double sinDelta = std::sin( from.Angle() );
  const double cosDelta = std::cos( from.Angle() );
  const double oneMinusCosDelta = 1.0 - cosDelta;

  const AxisAngle::AxisVector & u = from.Axis();
  const double uX = u.X();
  const double uY = u.Y();
  const double uZ = u.Z();

  double m[9];

  m[XX] = oneMinusCosDelta * uX * uX  +  cosDelta;
  m[XY] = oneMinusCosDelta * uX * uY  -  sinDelta * uZ;
  m[XZ] = oneMinusCosDelta * uX * uZ  +  sinDelta * uY;

  m[YX] = oneMinusCosDelta * uY * uX  +  sinDelta * uZ;
  m[YY] = oneMinusCosDelta * uY * uY  +  cosDelta;
  m[YZ] = oneMinusCosDelta * uY * uZ  -  sinDelta * uX;

  m[ZX] = oneMinusCosDelta * uZ * uX  -  sinDelta * uY;
  m[ZY] = oneMinusCosDelta * uZ * uY  +  sinDelta * uX;
  m[ZZ] = oneMinusCosDelta * uZ * uZ  +  cosDelta;

  to.SetComponents(m,m+9);
} // convert to Rotation3D

void convert( AxisAngle const & /* from */ , EulerAngles & /* to */ )
{
  // TODO
}

void convert( AxisAngle const & from, Quaternion  & to)
{
  double s = std::sin (from.Angle()/2);
  DisplacementVector3D< Cartesian3D<double> > axis = from.Axis();

  to.SetComponents( std::cos(from.Angle()/2),
                    s*axis.X(),
                    s*axis.Y(),
                    s*axis.Z()
                  );
} // convert to Quaternion



// ----------------------------------------------------------------------
// conversions from EulerAngles

void convert( EulerAngles const & from, Rotation3D  & to)
{
  typedef double Scalar; 
  const Scalar sPhi   = std::sin( from.Phi()   );
  const Scalar cPhi   = std::cos( from.Phi()   );
  const Scalar sTheta = std::sin( from.Theta() );
  const Scalar cTheta = std::cos( from.Theta() );
  const Scalar sPsi   = std::sin( from.Psi()   );
  const Scalar cPsi   = std::cos( from.Psi()   );
  to.SetComponents  
  (  cPsi*cPhi-sPsi*cTheta*sPhi,  cPsi*sPhi+sPsi*cTheta*cPhi, sPsi*sTheta
  , -sPsi*cPhi-cPsi*cTheta*sPhi, -sPsi*sPhi+cPsi*cTheta*cPhi, cPsi*sTheta
  ,        sTheta*sPhi,              -sTheta*cPhi,            cTheta
  );
}

void convert( EulerAngles const & from, AxisAngle   & to)
{
  Quaternion q;
  convert (from, q);
  convert (q, to);
}

void convert( EulerAngles const & from, Quaternion  & to)
{
  typedef double Scalar; 
  const Scalar plus   = (from.Phi()+from.Psi())/2;
  const Scalar minus  = (from.Phi()-from.Psi())/2;
  const Scalar sPlus  = std::sin( plus  );
  const Scalar cPlus  = std::cos( plus  );  
  const Scalar sMinus = std::sin( minus );
  const Scalar cMinus = std::cos( minus );  
  const Scalar sTheta = std::sin( from.Theta() );
  const Scalar cTheta = std::cos( from.Theta() );
  
  to.SetComponents ( cTheta*cPlus, sTheta*sMinus, sTheta*cMinus, cTheta*sPlus );
  // TODO -- carefully check that this is correct
}



// ----------------------------------------------------------------------
// conversions from Quaternion

void convert( Quaternion const & from, Rotation3D  & to)
{
  const double q0 = from.U();
  const double q1 = from.I();
  const double q2 = from.J();
  const double q3 = from.K();
  const double q00 = q0*q0;
  const double q01 = q0*q1;
  const double q02 = q0*q2;
  const double q03 = q0*q3;
  const double q11 = q1*q1;
  const double q12 = q1*q2;
  const double q13 = q1*q3;
  const double q22 = q2*q2;
  const double q23 = q2*q3;
  const double q33 = q3*q3;

  to.SetComponents (
    q00+q11-q22-q33 , 2*(q12-q03)     , 2*(q02+q13),
    2*(q12+q03)     , q00-q11+q22-q33 , 2*(q23-q01),
    2*(q13-q02)     , 2*(q23+q01)     , q00-q11-q22+q33 );

} // conversion to Rotation3D

void convert( Quaternion const & from, AxisAngle   & to)
{
  double u = from.U();
  if ( u >= 0 ) {
    if ( u > 1 ) u = 1;
    const double angle = 2.0 * std::acos ( from.U() );
    DisplacementVector3D< Cartesian3D<double> >
                                axis (from.I(), from.J(), from.K());
    to.SetComponents ( axis, angle );
  } else {
    if ( u < -1 ) u = -1;
    const double angle = 2.0 * std::acos ( -from.U() );
    DisplacementVector3D< Cartesian3D<double> >
                                axis (-from.I(), -from.J(), -from.K());
    to.SetComponents ( axis, angle );
  }
} // conversion to AxisAngle

void convert( Quaternion const & /* from */ , EulerAngles & /* to */ )
{
  // TODO
}



// ----------------------------------------------------------------------
// conversions from RotationX

void convert( RotationX const & from, Rotation3D  & to)
{
  const double c = from.CosAngle();
  const double s = from.SinAngle();
  to.SetComponents ( 1,  0,  0,
                     0,  c, -s,
                     0,  s,  c );
}

void convert( RotationX const & from, AxisAngle   & to)
{
  DisplacementVector3D< Cartesian3D<double> > axis (1, 0, 0);
  to.SetComponents ( axis, from.Angle() );
}

void convert( RotationX const & /* from */ , EulerAngles & /* to */ )
{
  // TODO
}

void convert( RotationX const & from, Quaternion  & to)
{
  to.SetComponents (std::cos(from.Angle()/2), std::sin(from.Angle()/2), 0, 0);
}



// ----------------------------------------------------------------------
// conversions from RotationY

void convert( RotationY const & from, Rotation3D  & to)
{
  const double c = from.CosAngle();
  const double s = from.SinAngle();
  to.SetComponents (  c, 0, s,
                      0, 1, 0,
                     -s, 0, c );
}

void convert( RotationY const & from, AxisAngle   & to)
{
  DisplacementVector3D< Cartesian3D<double> > axis (0, 1, 0);
  to.SetComponents ( axis, from.Angle() );
}

void convert( RotationY const & /* from */ , EulerAngles & /* to */ )
{
  // TODO
}

void convert( RotationY const & from, Quaternion  & to)
{
  to.SetComponents (std::cos(from.Angle()/2), 0, std::sin(from.Angle()/2), 0);
}



// ----------------------------------------------------------------------
// conversions from RotationZ

void convert( RotationZ const & from, Rotation3D  & to)
{
  const double c = from.CosAngle();
  const double s = from.SinAngle();
  to.SetComponents ( c, -s, 0,
                     s,  c, 0,
                     0,  0, 1 );
}

void convert( RotationZ const & from, AxisAngle   & to)
{
  DisplacementVector3D< Cartesian3D<double> > axis (0, 0, 1);
  to.SetComponents ( axis, from.Angle() );
}

void convert( RotationZ const & /* from */ , EulerAngles & /* to */ )
{
  // TODO
}

void convert( RotationZ const & from, Quaternion  & to)
{
  to.SetComponents (0, 0, std::cos(from.Angle()/2), std::sin(from.Angle()/2));
}

} //namespace gv_detail
} //namespace Math
} //namespace ROOT
