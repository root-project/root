/// \file ROOT/RPadLength.hxx
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

#ifndef ROOT7_RPadLength
#define ROOT7_RPadLength

#include <string>

#include <ROOT/RDrawingAttr.hxx>

namespace ROOT {
namespace Experimental {


class RPadLength : public RAttributesVisitor {

protected:
   const RDrawableAttributes::Map_t &GetDefaults() const override
   {
      static auto dflts = RDrawableAttributes::Map_t().AddDouble("normal",0.).AddDouble("pixel",0.).AddDouble("user",0.);
      return dflts;
   }

   mutable double *fNormal{nullptr}; ///<! normal component from attributes container
   mutable double *fPixel{nullptr};  ///<! pixel component from attributes container
   mutable double *fUser{nullptr};   ///<! user component from attributes container

   double *SetValueGetPtr(const std::string &name, double v)
   {
      SetValue(name, v);
      return GetDoublePtr(name);
   }

   void GetFast() const
   {
      fNormal = GetDoublePtr("normal");
      fPixel = GetDoublePtr("pixel");
      fUser = GetDoublePtr("user");
   }

public:
   template <class DERIVED>
   struct CoordSysBase {
      double  fVal{0.};      ///<! Coordinate value

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
      friend DERIVED operator*(const DERIVED &lhs, double rhs) { return DERIVED(lhs.fVal * rhs); }
      friend DERIVED operator*(double lhs, const DERIVED &rhs) { return DERIVED(lhs * rhs.fVal); }
      friend DERIVED operator/(const DERIVED &lhs, double rhs) { return DERIVED(lhs.fVal * rhs); }
      friend bool operator<(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal < rhs.fVal; }
      friend bool operator>(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal > rhs.fVal; }
      friend bool operator<=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal <= rhs.fVal; }
      friend bool operator>=(const DERIVED &lhs, const DERIVED &rhs) { return lhs.fVal >= rhs.fVal; }
      // no ==, !=
   };

   /// \defgroup PadCoordSystems RPad coordinate systems
   /// These define typesafe coordinates used by RPad to identify which coordinate system a coordinate is referring to.
   /// The origin (0,0) is in the `RPad`'s bottom left corner for all of them.
   /// \{

   /** \class Normal
     A normalized coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `RPad`. Resizing the pad
     will resize the objects with it.
    */
   struct Normal: CoordSysBase<Normal> {
      using CoordSysBase<Normal>::CoordSysBase;
   };

   /** \class Pixel
     A pixel coordinate: 0 in the left, bottom corner, 1 in the top, right corner of the `RPad`. Resizing the pad will
     keep the pixel-position of the objects positioned in `Pixel` coordinates.
    */
   struct Pixel: CoordSysBase<Pixel> {
      using CoordSysBase<Pixel>::CoordSysBase;
   };

   /** \class User
     A user coordinate, as defined by the EUserCoordSystem parameter of the `RPad`.
    */
   struct User: CoordSysBase<User> {
      using CoordSysBase<User>::CoordSysBase;
   };
   /// \}

   using RAttributesVisitor::RAttributesVisitor;

   RPadLength(RDrawableAttributes &cont, const std::string &prefix = "") : RAttributesVisitor(cont, prefix)
   {
      GetFast();
   }

   RPadLength(RAttributesVisitor *parent, const std::string &prefix = "") : RAttributesVisitor(parent, prefix)
   {
      GetFast();
   }

   RPadLength(const RPadLength &src) : RAttributesVisitor()
   {
      CreateOwnAttr();
      SemanticCopy(src);
      GetFast();
   }

   RPadLength &operator=(const RPadLength &src)
   {
      Clear();
      Copy(src);
      return *this;
   }

   void Copy(const RPadLength &src)
   {
      RAttributesVisitor::Copy(src, false);
      GetFast();
   }

   /// Constructor from a `Normal` coordinate.
   RPadLength(Normal normal): RPadLength() { SetNormal(normal.fVal); }

   /// Constructor from a `Pixel` coordinate.
   RPadLength(Pixel px): RPadLength() { SetPixel(px.fVal); }

   /// Constructor from a `User` coordinate.
   RPadLength(User user) : RPadLength() { SetUser(user.fVal); }

   bool HasNormal() const { return fNormal != nullptr; }
   bool HasPixel() const { return fPixel != nullptr; }
   bool HasUser() const { return fUser != nullptr; }

   RPadLength &SetNormal(double v) { if (fNormal) *fNormal = v; else fNormal = SetValueGetPtr("normal", v); return *this; }
   RPadLength &SetPixel(double v) { if (fPixel) *fPixel = v; else fPixel = SetValueGetPtr("pixel", v); return *this; }
   RPadLength &SetUser(double v) { if (fUser) *fUser = v; else fUser = SetValueGetPtr("user",v); return *this; }

   double GetNormal() const { return fNormal ? *fNormal : 0.; }
   double GetPixel() const { return fPixel ? *fPixel : 0.; }
   double GetUser() const { return fUser ? *fUser : 0.; }

   void ClearNormal() { ClearValue("normal"); fNormal = nullptr; }
   void ClearPixel() { ClearValue("pixel");  fPixel = nullptr; }
   void ClearUser() { ClearValue("user"); fUser = nullptr; }

   void Clear() { ClearNormal(); ClearPixel(); ClearUser(); }

   /// Add two `RPadLength`s.
   friend RPadLength operator+(RPadLength lhs, const RPadLength &rhs)
   {
      RPadLength res;
      if (lhs.HasNormal() || rhs.HasNormal())
         res.SetNormal(lhs.GetNormal() + rhs.GetNormal());
      if (lhs.HasPixel() || rhs.HasPixel())
         res.SetPixel(lhs.GetPixel() + rhs.GetPixel());
      if (lhs.HasUser() || rhs.HasUser())
         res.SetUser(lhs.GetUser() + rhs.GetUser());
      return res;
   }

   /// Subtract two `RPadLength`s.
   friend RPadLength operator-(RPadLength lhs, const RPadLength &rhs)
   {
      RPadLength res;
      if (lhs.HasNormal() || rhs.HasNormal())
         res.SetNormal(lhs.GetNormal() - rhs.GetNormal());
      if (lhs.HasPixel() || rhs.HasPixel())
         res.SetPixel(lhs.GetPixel() - rhs.GetPixel());
      if (lhs.HasUser() || rhs.HasUser())
         res.SetUser(lhs.GetUser() - rhs.GetUser());
      return res;
   }

   /// Unary -.
   RPadLength operator-()
   {
      RPadLength res;
      if (HasNormal()) res.SetNormal(-GetNormal());
      if (HasPixel()) res.SetPixel(-GetPixel());
      if (HasUser()) res.SetUser(-GetUser());
      return res;
   }

   /// Add a `RPadLength`.
   RPadLength &operator+=(const RPadLength &rhs)
   {
      if (HasNormal() || rhs.HasNormal())
         SetNormal(GetNormal() + rhs.GetNormal());
      if (HasPixel() || rhs.HasPixel())
         SetPixel(GetPixel() + rhs.GetPixel());
      if (HasUser() || rhs.HasUser())
         SetUser(GetUser() + rhs.GetUser());
      return *this;
   };

   /// Subtract a `RPadLength`.
   RPadLength &operator-=(const RPadLength &rhs)
   {
      if (HasNormal() || rhs.HasNormal())
         SetNormal(GetNormal() - rhs.GetNormal());
      if (HasPixel() || rhs.HasPixel())
         SetPixel(GetPixel() - rhs.GetPixel());
      if (HasUser() || rhs.HasUser())
         SetUser(GetUser() - rhs.GetUser());
      return *this;
   };

   RPadLength &operator*=(double scale)
   {
      if (HasNormal()) SetNormal(scale*GetNormal());
      if (HasPixel()) SetPixel(scale*GetPixel());
      if (HasUser()) SetUser(scale*GetUser());
      return *this;
   }

};

/// User-defined literal for `RPadLength::Normal`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_normal, 0.0_normal, RLineExtent(0.2_normal, 0.5_normal));
/// ```
inline RPadLength::Normal operator"" _normal(long double val)
{
   return RPadLength::Normal{(double)val};
}
inline RPadLength::Normal operator"" _normal(unsigned long long int val)
{
   return RPadLength::Normal{(double)val};
}

/// User-defined literal for `RPadLength::Pixel`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(100_px, 0_px, RLineExtent(20_px, 50_px));
/// ```
inline RPadLength::Pixel operator"" _px(long double val)
{
   return RPadLength::Pixel{(double)val};
}
inline RPadLength::Pixel operator"" _px(unsigned long long int val)
{
   return RPadLength::Pixel{(double)val};
}

/// User-defined literal for `RPadLength::User`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_user, 0.0_user, RLineExtent(0.2_user, 0.5_user));
/// ```
inline RPadLength::User operator"" _user(long double val)
{
   return RPadLength::User{(double)val};
}
inline RPadLength::User operator"" _user(unsigned long long int val)
{
   return RPadLength::User{(double)val};
}



} // namespace Experimental
} // namespace ROOT

#endif
