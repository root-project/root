// @(#)root/mathcore:$Id$
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
// Last update: $Id$
//

// TODO - For now, all conversions are grouped in this one compilation unit.
//        The intention is to seraparte them into a few .cpp files instead,
//        so that users needing one form need not incorporate code for them all.

#include "Math/GenVector/3DConversions.h"

#include "Math/Math.h"

#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationZYX.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"

#include <cmath>
#include <limits>

namespace ROOT {
namespace Math {
namespace gv_detail {

enum ERotation3DMatrixIndex
{ kXX = Rotation3D::kXX, kXY = Rotation3D::kXY, kXZ = Rotation3D::kXZ
, kYX = Rotation3D::kYX, kYY = Rotation3D::kYY, kYZ = Rotation3D::kYZ
, kZX = Rotation3D::kZX, kZY = Rotation3D::kZY, kZZ = Rotation3D::kZZ
};


// ----------------------------------------------------------------------
void convert( Rotation3D const & from, AxisAngle   & to)
{
   // conversions from Rotation3D
   double m[9];
   from.GetComponents(m, m+9);

   const double  uZ = m[kYX] - m[kXY];
   const double  uY = m[kXZ] - m[kZX];
   const double  uX = m[kZY] - m[kYZ];


   // in case of rotaiton of an angle PI, the rotation matrix is symmetric and
   // uX = uY = uZ  = 0. Use then conversion through the quaternion
   if ( std::fabs( uX ) < 8.*std::numeric_limits<double>::epsilon() &&
        std::fabs( uY ) < 8.*std::numeric_limits<double>::epsilon() &&
        std::fabs( uZ ) < 8.*std::numeric_limits<double>::epsilon() ) {
      Quaternion tmp;
      convert (from,tmp);
      convert (tmp,to);
      return;
   }

   AxisAngle::AxisVector u;

   u.SetCoordinates( uX, uY, uZ );

   static const double pi = M_PI;

   double angle;
   const double cosdelta = (m[kXX] + m[kYY] + m[kZZ] - 1.0) / 2.0;
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
   static const double pi = M_PI;
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
   // conversion from Rotation3D to Euler Angles
   // Mathematical justification appears in
   // http://www.cern.ch/mathlibs/documents/eulerAngleComputation.pdf

   double r[9];
   from.GetComponents(r,r+9);

   double phi, theta, psi;
   double psiPlusPhi, psiMinusPhi;
   static const double pi = M_PI;
   static const double pi_2 = M_PI_2;

   theta = (std::fabs(r[kZZ]) <= 1.0) ? std::acos(r[kZZ]) :
      (r[kZZ]  >  0.0) ?     0            : pi;

   double cosTheta = r[kZZ];
   if (cosTheta > 1)  cosTheta = 1;
   if (cosTheta < -1) cosTheta = -1;

   // Compute psi +/- phi:
   // Depending on whether cosTheta is positive or negative and whether it
   // is less than 1 in absolute value, different mathematically equivalent
   // expressions are numerically stable.
   if (cosTheta == 1) {
      psiPlusPhi = atan2 ( r[kXY] - r[kYX], r[kXX] + r[kYY] );
      psiMinusPhi = 0;
   } else if (cosTheta >= 0) {
      psiPlusPhi = atan2 ( r[kXY] - r[kYX], r[kXX] + r[kYY] );
      double s = -r[kXY] - r[kYX]; // sin (psi-phi) * (1 - cos theta)
      double c =  r[kXX] - r[kYY]; // cos (psi-phi) * (1 - cos theta)
      psiMinusPhi = atan2 ( s, c );
   } else if (cosTheta > -1) {
      psiMinusPhi = atan2 ( -r[kXY] - r[kYX], r[kXX] - r[kYY] );
      double s = r[kXY] - r[kYX]; // sin (psi+phi) * (1 + cos theta)
      double c = r[kXX] + r[kYY]; // cos (psi+phi) * (1 + cos theta)
      psiPlusPhi = atan2 ( s, c );
   } else { // cosTheta == -1
      psiMinusPhi = atan2 ( -r[kXY] - r[kYX], r[kXX] - r[kYY] );
      psiPlusPhi = 0;
   }

   psi = .5 * (psiPlusPhi + psiMinusPhi);
   phi = .5 * (psiPlusPhi - psiMinusPhi);

   // Now correct by pi if we have managed to get a value of psiPlusPhi
   // or psiMinusPhi that was off by 2 pi:

   // set up w[i], all of which would be positive if sin and cosine of
   // psi and phi were positive:
   double w[4];
   w[0] = r[kXZ]; w[1] = r[kZX]; w[2] = r[kYZ]; w[3] = -r[kZY];

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
         if (w[2] > 0 && std::fabs(psi) > pi_2) correctByPi ( psi, phi );
         if (w[2] < 0 && std::fabs(psi) < pi_2) correctByPi ( psi, phi );
            break;
      case 3:
         if (w[3] > 0 && std::fabs(phi) > pi_2) correctByPi ( psi, phi );
         if (w[3] < 0 && std::fabs(phi) < pi_2) correctByPi ( psi, phi );
            break;
   }

   to.SetComponents( phi, theta, psi );

} // convert to EulerAngles

////////////////////////////////////////////////////////////////////////////////
/// conversion from Rotation3D to Quaternion

void convert( Rotation3D const & from, Quaternion  & to)
{
   double m[9];
   from.GetComponents(m, m+9);

   const double d0 =   m[kXX] + m[kYY] + m[kZZ];
   const double d1 = + m[kXX] - m[kYY] - m[kZZ];
   const double d2 = - m[kXX] + m[kYY] - m[kZZ];
   const double d3 = - m[kXX] - m[kYY] + m[kZZ];

   // these are related to the various q^2 values;
   // choose the largest to avoid dividing two small numbers and losing accuracy.

   if ( d0 >= d1 && d0 >= d2 && d0 >= d3 ) {
      const double q0 = .5*std::sqrt(1+d0);
      const double f  = .25/q0;
      const double q1 = f*(m[kZY]-m[kYZ]);
      const double q2 = f*(m[kXZ]-m[kZX]);
      const double q3 = f*(m[kYX]-m[kXY]);
      to.SetComponents(q0,q1,q2,q3);
      to.Rectify();
      return;
   } else if ( d1 >= d2 && d1 >= d3 ) {
      const double q1 = .5*std::sqrt(1+d1);
      const double f  = .25/q1;
      const double q0 = f*(m[kZY]-m[kYZ]);
      const double q2 = f*(m[kXY]+m[kYX]);
      const double q3 = f*(m[kXZ]+m[kZX]);
      to.SetComponents(q0,q1,q2,q3);
      to.Rectify();
      return;
   } else if ( d2 >= d3 ) {
      const double q2 = .5*std::sqrt(1+d2);
      const double f  = .25/q2;
      const double q0 = f*(m[kXZ]-m[kZX]);
      const double q1 = f*(m[kXY]+m[kYX]);
      const double q3 = f*(m[kYZ]+m[kZY]);
      to.SetComponents(q0,q1,q2,q3);
      to.Rectify();
      return;
   } else {
      const double q3 = .5*std::sqrt(1+d3);
      const double f  = .25/q3;
      const double q0 = f*(m[kYX]-m[kXY]);
      const double q1 = f*(m[kXZ]+m[kZX]);
      const double q2 = f*(m[kYZ]+m[kZY]);
      to.SetComponents(q0,q1,q2,q3);
      to.Rectify();
      return;
   }
}  // convert to Quaternion

////////////////////////////////////////////////////////////////////////////////
/// conversion from Rotation3D to RotationZYX
/// same Math used as for EulerAngles apart from some different meaning of angles and
/// matrix elements. But the basic algoprithms principles are the same described in
/// http://www.cern.ch/mathlibs/documents/eulerAngleComputation.pdf

void convert( Rotation3D const & from, RotationZYX  & to)
{
   // theta is assumed to be in range [-PI/2,PI/2].
   // this is guranteed by the Rectify function

   static const double pi_2 = M_PI_2;

   double r[9];
   from.GetComponents(r,r+9);

   double phi,theta,psi = 0;

   // careful for numeical error make sin(theta) ourtside [-1,1]
   double sinTheta =  r[kXZ];
   if ( sinTheta < -1.0) sinTheta = -1.0;
   if ( sinTheta >  1.0) sinTheta =  1.0;
   theta = std::asin( sinTheta );

   // compute psi +/- phi
   // Depending on whether cosTheta is positive or negative and whether it
   // is less than 1 in absolute value, different mathematically equivalent
   // expressions are numerically stable.
   // algorithm from
   // adapted for the case 3-2-1

   double psiPlusPhi = 0;
   double psiMinusPhi = 0;

   // valid if sinTheta not eq to -1 otherwise is zero
   if (sinTheta > - 1.0)
      psiPlusPhi = atan2 ( r[kYX] + r[kZY], r[kYY] - r[kZX] );

   // valid if sinTheta not eq. to 1
   if (sinTheta < 1.0)
      psiMinusPhi = atan2 ( r[kZY] - r[kYX] , r[kYY] + r[kZX] );

   psi = .5 * (psiPlusPhi + psiMinusPhi);
   phi = .5 * (psiPlusPhi - psiMinusPhi);

   // correction is not necessary if sinTheta = +/- 1
   //if (sinTheta == 1.0 || sinTheta == -1.0) return;

   // apply the corrections according to max of the other terms
   // I think is assumed convention that theta is between -PI/2,PI/2.
   // OTHERWISE RESULT MIGHT BE DIFFERENT ???

   //since we determine phi+psi or phi-psi phi and psi can be both have a shift of +/- PI.
   // The shift must be applied on both (the sum (or difference) is knows to +/- 2PI )
   //This can be fixed looking at the other 4 matrix terms, which have terms in sin and cos of psi
   // and phi. sin(psi+/-PI) = -sin(psi) and cos(psi+/-PI) = -cos(psi).
   //Use then the biggest term for making the correction to minimize possible numerical errors

   // set up w[i], all of which would be positive if sin and cosine of
   // psi and phi were positive:
   double w[4];
   w[0] = -r[kYZ]; w[1] = -r[kXY]; w[2] = r[kZZ]; w[3] = r[kXX];

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
         if (w[2] > 0 && std::fabs(psi) > pi_2) correctByPi ( psi, phi );
         if (w[2] < 0 && std::fabs(psi) < pi_2) correctByPi ( psi, phi );
            break;
      case 3:
         if (w[3] > 0 && std::fabs(phi) > pi_2) correctByPi ( psi, phi );
         if (w[3] < 0 && std::fabs(phi) < pi_2) correctByPi ( psi, phi );
            break;
   }

   to.SetComponents(phi, theta, psi);

} // convert to RotationZYX

// ----------------------------------------------------------------------
// conversions from AxisAngle

void convert( AxisAngle const & from, Rotation3D  & to)
{
   // conversion from AxixAngle to Rotation3D

   const double sinDelta = std::sin( from.Angle() );
   const double cosDelta = std::cos( from.Angle() );
   const double oneMinusCosDelta = 1.0 - cosDelta;

   const AxisAngle::AxisVector & u = from.Axis();
   const double uX = u.X();
   const double uY = u.Y();
   const double uZ = u.Z();

   double m[9];

   m[kXX] = oneMinusCosDelta * uX * uX  +  cosDelta;
   m[kXY] = oneMinusCosDelta * uX * uY  -  sinDelta * uZ;
   m[kXZ] = oneMinusCosDelta * uX * uZ  +  sinDelta * uY;

   m[kYX] = oneMinusCosDelta * uY * uX  +  sinDelta * uZ;
   m[kYY] = oneMinusCosDelta * uY * uY  +  cosDelta;
   m[kYZ] = oneMinusCosDelta * uY * uZ  -  sinDelta * uX;

   m[kZX] = oneMinusCosDelta * uZ * uX  -  sinDelta * uY;
   m[kZY] = oneMinusCosDelta * uZ * uY  +  sinDelta * uX;
   m[kZZ] = oneMinusCosDelta * uZ * uZ  +  cosDelta;

   to.SetComponents(m,m+9);
} // convert to Rotation3D

void convert( AxisAngle const & from , EulerAngles & to  )
{
   // conversion from AxixAngle to EulerAngles
   // TODO better : temporary make conversion using  Rotation3D

   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

void convert( AxisAngle const & from, Quaternion  & to)
{
   // conversion from AxixAngle to Quaternion

   double s = std::sin (from.Angle()/2);
   DisplacementVector3D< Cartesian3D<double> > axis = from.Axis();

   to.SetComponents( std::cos(from.Angle()/2),
                     s*axis.X(),
                     s*axis.Y(),
                     s*axis.Z()
                     );
} // convert to Quaternion

void convert( AxisAngle const & from , RotationZYX & to  )
{
   // conversion from AxisAngle to RotationZYX
   // TODO better : temporary make conversion using  Rotation3D
   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}


// ----------------------------------------------------------------------
// conversions from EulerAngles

void convert( EulerAngles const & from, Rotation3D  & to)
{
   // conversion from EulerAngles to Rotation3D

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
   // conversion from EulerAngles to AxisAngle
   // make converting first to quaternion
   Quaternion q;
   convert (from, q);
   convert (q, to);
}

void convert( EulerAngles const & from, Quaternion  & to)
{
   // conversion from EulerAngles to Quaternion

   typedef double Scalar;
   const Scalar plus   = (from.Phi()+from.Psi())/2;
   const Scalar minus  = (from.Phi()-from.Psi())/2;
   const Scalar sPlus  = std::sin( plus  );
   const Scalar cPlus  = std::cos( plus  );
   const Scalar sMinus = std::sin( minus );
   const Scalar cMinus = std::cos( minus );
   const Scalar sTheta = std::sin( from.Theta()/2 );
   const Scalar cTheta = std::cos( from.Theta()/2 );

   to.SetComponents ( cTheta*cPlus, -sTheta*cMinus, -sTheta*sMinus, -cTheta*sPlus );
   // TODO -- carefully check that this is correct
}

void convert( EulerAngles const & from , RotationZYX & to  )
{
   // conversion from EulerAngles to RotationZYX
   // TODO better : temporary make conversion using  Rotation3D
   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}


// ----------------------------------------------------------------------
// conversions from Quaternion

void convert( Quaternion const & from, Rotation3D  & to)
{
   // conversion from Quaternion to Rotation3D

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
   // conversion from Quaternion to AxisAngle

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

void convert( Quaternion const &  from, EulerAngles & to  )
{
   // conversion from Quaternion to EulerAngles
   // TODO better
   // temporary make conversion using  Rotation3D

   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

void convert( Quaternion const & from , RotationZYX & to  )
{
   // conversion from Quaternion to RotationZYX
   // TODO better : temporary make conversion using  Rotation3D
   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

// ----------------------------------------------------------------------
// conversions from RotationZYX
void convert( RotationZYX const & from, Rotation3D  & to) {
   // conversion to Rotation3D (matrix)

   double phi,theta,psi = 0;
   from.GetComponents(phi,theta,psi);
   to.SetComponents( std::cos(theta)*std::cos(phi),
                      - std::cos(theta)*std::sin(phi),
                      std::sin(theta),

                      std::cos(psi)*std::sin(phi) + std::sin(psi)*std::sin(theta)*std::cos(phi),
                      std::cos(psi)*std::cos(phi) - std::sin(psi)*std::sin(theta)*std::sin(phi),
                      -std::sin(psi)*std::cos(theta),

                      std::sin(psi)*std::sin(phi) - std::cos(psi)*std::sin(theta)*std::cos(phi),
                      std::sin(psi)*std::cos(phi) + std::cos(psi)*std::sin(theta)*std::sin(phi),
                      std::cos(psi)*std::cos(theta)
      );

}
void convert( RotationZYX const & from, AxisAngle   & to) {
   // conversion to axis angle
   // TODO better : temporary make conversion using  Rotation3D
   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}
void convert( RotationZYX const & from, EulerAngles & to) {
   // conversion to Euler angle
   // TODO better : temporary make conversion using  Rotation3D
   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}
void convert( RotationZYX const & from, Quaternion  & to) {
   double phi,theta,psi = 0;
   from.GetComponents(phi,theta,psi);

   double sphi2   = std::sin(phi/2);
   double cphi2   = std::cos(phi/2);
   double stheta2 = std::sin(theta/2);
   double ctheta2 = std::cos(theta/2);
   double spsi2   = std::sin(psi/2);
   double cpsi2   = std::cos(psi/2);
   to.SetComponents(  cphi2 * cpsi2 * ctheta2 - sphi2 * spsi2 * stheta2,
                      sphi2 * cpsi2 * stheta2 + cphi2 * spsi2 * ctheta2,
                      cphi2 * cpsi2 * stheta2 - sphi2 * spsi2 * ctheta2,
                      sphi2 * cpsi2 * ctheta2 + cphi2 * spsi2 * stheta2
      );
}


// ----------------------------------------------------------------------
// conversions from RotationX

void convert( RotationX const & from, Rotation3D  & to)
{
   // conversion from RotationX to Rotation3D

   const double c = from.CosAngle();
   const double s = from.SinAngle();
   to.SetComponents ( 1,  0,  0,
                      0,  c, -s,
                      0,  s,  c );
}

void convert( RotationX const & from, AxisAngle   & to)
{
   // conversion from RotationX to AxisAngle

   DisplacementVector3D< Cartesian3D<double> > axis (1, 0, 0);
   to.SetComponents ( axis, from.Angle() );
}

void convert( RotationX const & from , EulerAngles &  to  )
{
   // conversion from RotationX to EulerAngles
   //TODO better: temporary make conversion using  Rotation3D

   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

void convert( RotationX const & from, Quaternion  & to)
{
   // conversion from RotationX to Quaternion

   to.SetComponents (std::cos(from.Angle()/2), std::sin(from.Angle()/2), 0, 0);
}

void convert( RotationX const & from , RotationZYX & to  )
{
   // conversion from RotationX to RotationZYX
   to.SetComponents(0,0,from.Angle());
}


// ----------------------------------------------------------------------
// conversions from RotationY

void convert( RotationY const & from, Rotation3D  & to)
{
   // conversion from RotationY to Rotation3D

   const double c = from.CosAngle();
   const double s = from.SinAngle();
   to.SetComponents (  c, 0, s,
                       0, 1, 0,
                      -s, 0, c );
}

void convert( RotationY const & from, AxisAngle   & to)
{
   // conversion from RotationY to AxisAngle

   DisplacementVector3D< Cartesian3D<double> > axis (0, 1, 0);
   to.SetComponents ( axis, from.Angle() );
}

void convert( RotationY const & from, EulerAngles & to  )
{
   // conversion from RotationY to EulerAngles
   // TODO better: temporary make conversion using  Rotation3D

   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

void convert( RotationY const & from , RotationZYX & to  )
{
   // conversion from RotationY to RotationZYX
   to.SetComponents(0,from.Angle(),0);
}


void convert( RotationY const & from, Quaternion  & to)
{
   // conversion from RotationY to Quaternion

   to.SetComponents (std::cos(from.Angle()/2), 0, std::sin(from.Angle()/2), 0);
}



// ----------------------------------------------------------------------
// conversions from RotationZ

void convert( RotationZ const & from, Rotation3D  & to)
{
   // conversion from RotationZ to Rotation3D

   const double c = from.CosAngle();
   const double s = from.SinAngle();
   to.SetComponents ( c, -s, 0,
                      s,  c, 0,
                      0,  0, 1 );
}

void convert( RotationZ const & from, AxisAngle   & to)
{
   // conversion from RotationZ to AxisAngle

   DisplacementVector3D< Cartesian3D<double> > axis (0, 0, 1);
   to.SetComponents ( axis, from.Angle() );
}

void convert( RotationZ const & from  , EulerAngles & to  )
{
   // conversion from RotationZ to EulerAngles
   // TODO better: temporary make conversion using  Rotation3D

   Rotation3D tmp;
   convert(from,tmp);
   convert(tmp,to);
}

void convert( RotationZ const & from , RotationZYX & to  )
{
   // conversion from RotationY to RotationZYX
   to.SetComponents(from.Angle(),0,0);
}

void convert( RotationZ const & from, Quaternion  & to)
{
   // conversion from RotationZ to Quaternion

   to.SetComponents (std::cos(from.Angle()/2), 0, 0, std::sin(from.Angle()/2));
}

} //namespace gv_detail
} //namespace Math
} //namespace ROOT
