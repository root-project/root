// @(#)root/geom:$Id$
// Author: Andrei Gheata   20/12/19

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoVector3
#define ROOT_TGeoVector3

#include <TMath.h>

struct TGeoVector3 {
   double fVec[3] = {0.};
 
   TGeoVector3(const double x, const double y, const double z)
   {
      fVec[0] = x;
      fVec[1] = y;
      fVec[2] = z;
   }

   TGeoVector3(const double a = 0.)
   {
      fVec[0] = a;
      fVec[1] = a;
      fVec[2] = a;
   }

   TGeoVector3(TGeoVector3 const &other)
   {
      fVec[0] = other[0];
      fVec[1] = other[1];
      fVec[2] = other[2];
   }
  
   double &operator[](const int index) { return fVec[index]; }
   double const &operator[](const int index) const { return fVec[index]; }

   // Inplace binary operators

#define TGEOVECTOR3_INPLACE_BINARY_OP(OPERATOR)                       \
   inline TGeoVector3 &operator OPERATOR(const TGeoVector3 &other)    \
   {                                                                  \
      fVec[0] OPERATOR other.fVec[0];                                 \
      fVec[1] OPERATOR other.fVec[1];                                 \
      fVec[2] OPERATOR other.fVec[2];                                 \
      return *this;                                                   \
   }                                                                  \
   inline TGeoVector3 &operator OPERATOR(const double &scalar)        \
   {                                                                  \
      fVec[0] OPERATOR scalar;                                        \
      fVec[1] OPERATOR scalar;                                        \
      fVec[2] OPERATOR scalar;                                        \
      return *this;                                                   \
   }
   TGEOVECTOR3_INPLACE_BINARY_OP(+=)
   TGEOVECTOR3_INPLACE_BINARY_OP(-=)
   TGEOVECTOR3_INPLACE_BINARY_OP(*=)
   TGEOVECTOR3_INPLACE_BINARY_OP(/=)
#undef TGEOVECTOR3_INPLACE_BINARY_OP

   double &x() { return fVec[0]; }
   double const &x() const { return fVec[0]; }

   double &y() { return fVec[1]; }
   double const &y() const { return fVec[1]; }

   double &z() { return fVec[2]; }
   double const &z() const { return fVec[2]; }
 
   void Set(double const &a, double const &b, double const &c)
   {
      fVec[0] = a;
      fVec[1] = b;
      fVec[2] = c;
   }

   void Set(const double a) { Set(a, a, a); }


   /// \Return the length squared perpendicular to z direction
   double Perp2() const { return fVec[0] * fVec[0] + fVec[1] * fVec[1]; }

   /// \Return the length perpendicular to z direction
   double Perp() const { return TMath::Sqrt(Perp2()); }

   /// The dot product of two vector objects
   static double Dot(TGeoVector3 const &left, TGeoVector3 const &right)
   {
      return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
   }

   /// The dot product of two vector
   double Dot(TGeoVector3 const &right) const { return Dot(*this, right); }

   /// \return Squared magnitude of the vector.
   double Mag2() const { return Dot(*this, *this); }

   /// \return Magnitude of the vector.
   double Mag() const { return TMath::Sqrt(Mag2()); }

   double Length() const { return Mag(); }

   double Length2() const { return Mag2(); }

   /// Normalizes the vector by dividing each entry by the length.
   void Normalize() { *this *= (1. / Length()); }

   //TGeoVector3 Normalized() const { return TGeoVector3(*this) * (1. / Length()); }

   // checks if vector is normalized
   bool IsNormalized() const
   {
      double norm = Mag2();
      constexpr double tolerance = 1.e-10;
      return 1. - tolerance < norm && norm < 1 + tolerance;
   }

   /// \return Azimuthal angle between -pi and pi.
   double Phi() const { return TMath::ATan2(fVec[1], fVec[0]); }

   /// \return Polar angle between 0 and pi.
   double Theta() const { return TMath::ACos(fVec[2] / Mag()); }

   /// The cross (vector) product of two Vector3D<T> objects
   static TGeoVector3 Cross(TGeoVector3 const &left, TGeoVector3 const &right)
   {
      return TGeoVector3(left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2],
                         left[0] * right[1] - left[1] * right[0]);
   }

   TGeoVector3 Abs() const
   {
      return TGeoVector3(TMath::Abs(fVec[0]), TMath::Abs(fVec[1]), TMath::Abs(fVec[2]));
   }

   double Min() const { return TMath::Min(TMath::Min(fVec[0], fVec[1]) , fVec[2]); }

   double Max() const { return TMath::Max(TMath::Max(fVec[0], fVec[1]) , fVec[2]); }

   TGeoVector3 Unit() const
   {
      constexpr double kMinimum = std::numeric_limits<double>::min();
      const double mag2 = Mag2();
      TGeoVector3 output(*this);
      output /= TMath::Sqrt(mag2 + kMinimum);
      return output;
   }
};

/*
std::ostream &operator<<(std::ostream &os, TGeoVector3 const &vec)
{
   os << "(" << vec[0] << ", " << vec[1] << ", " << vec[2] << ")";
   return os;
}
*/

#define TGEOVECTOR3_BINARY_OP(OPERATOR, INPLACE)                      \
inline TGeoVector3 operator OPERATOR(const TGeoVector3 &lhs,          \
                                     const TGeoVector3 &rhs)          \
{                                                                     \
   TGeoVector3 result(lhs);                                           \
   result INPLACE rhs;                                                \
   return result;                                                     \
}                                                                     \
inline TGeoVector3 operator OPERATOR(TGeoVector3 const &lhs,          \
                                     const double rhs)                \
{                                                                     \
   TGeoVector3 result(lhs);                                           \
   result INPLACE rhs;                                                \
   return result;                                                     \
}                                                                     \
inline TGeoVector3 operator OPERATOR(const double lhs,                \
                                     TGeoVector3 const &rhs)          \
{                                                                     \
   TGeoVector3 result(lhs);                                           \
   result INPLACE rhs;                                                \
   return result;                                                     \
}
TGEOVECTOR3_BINARY_OP(+, +=)
TGEOVECTOR3_BINARY_OP(-, -=)
TGEOVECTOR3_BINARY_OP(*, *=)
TGEOVECTOR3_BINARY_OP(/, /=)
#undef TGEOVECTOR3_BINARY_OP

#endif
