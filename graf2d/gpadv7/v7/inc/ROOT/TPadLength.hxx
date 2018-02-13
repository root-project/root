/// \file ROOT/TPadLength.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-07-06
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TPadLength
#define ROOT7_TPadLength

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TPadLength
  A coordinate in a `TPad`.
  */

class TPadLength {
public:
   template <class DERIVED>
   struct CoordSysBase {
      double fVal = 0.; ///< Coordinate value

      CoordSysBase() = default;
      CoordSysBase(double val): fVal(val) {}
      DERIVED &ToDerived() { return static_cast<DERIVED &>(*this); }

      DERIVED operator-() { return DERIVED(-fVal); }

      friend DERIVED operator+(DERIVED lhs, DERIVED rhs) { return DERIVED{lhs.fVal + rhs.fVal}; }
      friend DERIVED operator-(DERIVED lhs, DERIVED rhs) { return DERIVED{lhs.fVal - rhs.fVal}; }
      friend double operator/(DERIVED lhs, DERIVED rhs) { return lhs.fVal / rhs.fVal; }
      DERIVED &operator+=(const DERIVED &rhs)
      {
         fVal += rhs.fVal;
         return ToDerived();
      }
      DERIVED &operator-=(const DERIVED &rhs)
      {
         fVal -= rhs.fVal;
         return ToDerived();
      }
      DERIVED &operator*=(double scale)
      {
         fVal *= scale;
         return ToDerived();
      }
      friend bool operator<(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal < rhs.fVal; }
      friend bool operator>(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal > rhs.fVal; }
      friend bool operator<=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal <= rhs.fVal; }
      friend bool operator>=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal >= rhs.fVal; }
      // no ==, !=
   };

   /// \defgroup PadCoordSystems TPad coordinate systems
   /// These define typesafe coordinates used by TPad to identify which coordinate system a coordinate is referring to.
   /// The origin (0,0) is in the `TPad`'s bottom left corner for all of them.
   /// \{

   /** \class Normal
     A normalized coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `TPad`. Resizing the pad
     will resize the objects with it.
    */
   struct Normal: CoordSysBase<Normal> {
      using CoordSysBase<Normal>::CoordSysBase;
   };

   /** \class Pixel
     A pixel coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `TPad`. Resizing the pad will
     keep the pixel-position of the objects positioned in `Pixel` coordinates.
    */
   struct Pixel: CoordSysBase<Pixel> {
      using CoordSysBase<Pixel>::CoordSysBase;
   };

   /** \class User
     A user coordinate, as defined by the EUserCoordSystem parameter of the `TPad`.
    */
   struct User: CoordSysBase<User> {
      using CoordSysBase<User>::CoordSysBase;
   };
   /// \}

   /// The normalized coordinate summand.
   Normal fNormal;

   /// The pixel coordinate summand.
   Pixel fPixel;

   /// The user coordinate summand.
   User fUser;

   /// Default constructor, initializing all coordinate parts to `0.`.
   TPadLength() = default;

   /// Constructor from a `Normal` coordinate.
   TPadLength(Normal normal): fNormal(normal) {}

   /// Constructor from a `Pixel` coordinate.
   TPadLength(Pixel px): fPixel(px) {}

   /// Constructor from a `User` coordinate.
   TPadLength(User user): fUser(user) {}

   /// Sort-of aggregate initialization constructor taking normal, pixel and user parts.
   TPadLength(Normal normal, Pixel px, User user): fNormal(normal), fPixel(px), fUser(user) {}

   /// Add two `TPadLength`s.
   friend TPadLength operator+(TPadLength lhs, const TPadLength &rhs)
   {
      return TPadLength{lhs.fNormal + rhs.fNormal, lhs.fPixel + rhs.fPixel, lhs.fUser + rhs.fUser};
   }

   /// Subtract two `TPadLength`s.
   friend TPadLength operator-(TPadLength lhs, const TPadLength &rhs)
   {
      return TPadLength{lhs.fNormal - rhs.fNormal, lhs.fPixel - rhs.fPixel, lhs.fUser - rhs.fUser};
   }

   /// Unary -.
   TPadLength operator-() {
      return TPadLength(-fNormal, -fPixel, -fUser);
   }

   /// Add a `TPadLength`.
   TPadLength &operator+=(const TPadLength &rhs)
   {
      fNormal += rhs.fNormal;
      fPixel += rhs.fPixel;
      fUser += rhs.fUser;
      return *this;
   };

   /// Subtract a `TPadLength`.
   TPadLength &operator-=(const TPadLength &rhs)
   {
      fNormal -= rhs.fNormal;
      fPixel -= rhs.fPixel;
      fUser -= rhs.fUser;
      return *this;
   };

   TPadLength &operator*=(double scale)
   {
      fNormal *= scale;
      fPixel *= scale;
      fUser *= scale;
      return *this;
   }
};

/// User-defined literal for `TPadLength::Normal`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// TLine(0.1_normal, 0.0_normal, TLineExtent(0.2_normal, 0.5_normal));
/// ```
inline TPadLength::Normal operator"" _normal(long double val)
{
   return TPadLength::Normal{(double)val};
}
inline TPadLength::Normal operator"" _normal(unsigned long long int val)
{
   return TPadLength::Normal{(double)val};
}

/// User-defined literal for `TPadLength::Pixel`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// TLine(100_px, 0_px, TLineExtent(20_px, 50_px));
/// ```
inline TPadLength::Pixel operator"" _px(long double val)
{
   return TPadLength::Pixel{(double)val};
}
inline TPadLength::Pixel operator"" _px(unsigned long long int val)
{
   return TPadLength::Pixel{(double)val};
}

/// User-defined literal for `TPadLength::User`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// TLine(0.1_user, 0.0_user, TLineExtent(0.2_user, 0.5_user));
/// ```
inline TPadLength::User operator"" _user(long double val)
{
   return TPadLength::User{(double)val};
}
inline TPadLength::User operator"" _user(unsigned long long int val)
{
   return TPadLength::User{(double)val};
}

} // namespace Experimental
} // namespace ROOT

#endif
