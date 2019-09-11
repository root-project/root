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

/** \class ROOT::Experimental::RPadLength
  A coordinate in a `RPad`.
  */

class RPadLength {
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

   /// The normalized coordinate summand.
   Normal fNormal;

   /// The pixel coordinate summand.
   Pixel fPixel;

   /// The user coordinate summand.
   User fUser;

   /// Default constructor, initializing all coordinate parts to `0.`.
   RPadLength() = default;

   /// Constructor from a `Normal` coordinate.
   RPadLength(Normal normal): fNormal(normal) {}

   /// Constructor from a `Pixel` coordinate.
   RPadLength(Pixel px): fPixel(px) {}

   /// Constructor from a `User` coordinate.
   RPadLength(User user): fUser(user) {}

   /// Sort-of aggregate initialization constructor taking normal, pixel and user parts.
   RPadLength(Normal normal, Pixel px, User user): fNormal(normal), fPixel(px), fUser(user) {}

   /// Add two `RPadLength`s.
   friend RPadLength operator+(RPadLength lhs, const RPadLength &rhs)
   {
      return RPadLength{lhs.fNormal + rhs.fNormal, lhs.fPixel + rhs.fPixel, lhs.fUser + rhs.fUser};
   }

   /// Subtract two `RPadLength`s.
   friend RPadLength operator-(RPadLength lhs, const RPadLength &rhs)
   {
      return RPadLength{lhs.fNormal - rhs.fNormal, lhs.fPixel - rhs.fPixel, lhs.fUser - rhs.fUser};
   }

   /// Unary -.
   RPadLength operator-() {
      return RPadLength(-fNormal, -fPixel, -fUser);
   }

   /// Add a `RPadLength`.
   RPadLength &operator+=(const RPadLength &rhs)
   {
      fNormal += rhs.fNormal;
      fPixel += rhs.fPixel;
      fUser += rhs.fUser;
      return *this;
   };

   /// Subtract a `RPadLength`.
   RPadLength &operator-=(const RPadLength &rhs)
   {
      fNormal -= rhs.fNormal;
      fPixel -= rhs.fPixel;
      fUser -= rhs.fUser;
      return *this;
   };

   RPadLength &operator*=(double scale)
   {
      fNormal *= scale;
      fPixel *= scale;
      fUser *= scale;
      return *this;
   }

   void SetFromAttrString(const std::string &val, const std::string &name);
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

RPadLength FromAttributeString(const std::string &val, const std::string &name, RPadLength*);
std::string ToAttributeString(const RPadLength &len);



class RPadLengthNew : public RAttributesVisitor {

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

   RPadLengthNew(RDrawableAttributes &cont, const std::string &prefix = "") : RAttributesVisitor(cont, prefix)
   {
      GetFast();
   }

   RPadLengthNew(RAttributesVisitor *parent, const std::string &prefix = "") : RAttributesVisitor(parent, prefix)
   {
      GetFast();
   }

   RPadLengthNew(const RPadLengthNew &src) : RAttributesVisitor()
   {
      CreateOwnAttr();
      SemanticCopy(src);
      GetFast();
   }

   RPadLengthNew &operator=(const RPadLengthNew &src)
   {
      Clear();
      Copy(src);
      return *this;
   }

   void Copy(const RPadLengthNew &src)
   {
      RAttributesVisitor::Copy(src, false);
      GetFast();
   }

   /// Constructor from a `Normal` coordinate.
   RPadLengthNew(Normal normal): RPadLengthNew() { SetNormal(normal.fVal); }

   /// Constructor from a `Pixel` coordinate.
   RPadLengthNew(Pixel px): RPadLengthNew() { SetPixel(px.fVal); }

   /// Constructor from a `User` coordinate.
   RPadLengthNew(User user) : RPadLengthNew() { SetUser(user.fVal); }

   bool HasNormal() const { return fNormal != nullptr; }
   bool HasPixel() const { return fPixel != nullptr; }
   bool HasUser() const { return fUser != nullptr; }

   RPadLengthNew &SetNormal(double v) { if (fNormal) *fNormal = v; else fNormal = SetValueGetPtr("normal", v); return *this; }
   RPadLengthNew &SetPixel(double v) { if (fPixel) *fPixel = v; else fPixel = SetValueGetPtr("pixel", v); return *this; }
   RPadLengthNew &SetUser(double v) { if (fUser) *fUser = v; else fUser = SetValueGetPtr("user",v); return *this; }

   double GetNormal() const { return fNormal ? *fNormal : 0; }
   double GetPixel() const { return fPixel ? *fPixel : 0; }
   double GetUser() const { return fUser ? *fUser : 0; }

   void ClearNormal() { ClearValue("normal"); fNormal = nullptr; }
   void ClearPixel() { ClearValue("pixel");  fPixel = nullptr; }
   void ClearUser() { ClearValue("user"); fUser = nullptr; }

   void Clear() { RAttributesVisitor::Clear(); fNormal = fPixel = fUser = nullptr; }

   /// Add two `RPadLength`s.
   friend RPadLengthNew operator+(RPadLengthNew lhs, const RPadLengthNew &rhs)
   {
      RPadLengthNew res;
      auto nl = lhs.Eval("normal");
      auto nr = rhs.Eval("normal");
      if (nl || nr)
         res.SetNormal((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      nl = lhs.Eval("pixel");
      nr = rhs.Eval("pixel");
      if (nl || nr)
         res.SetPixel((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      nl = lhs.Eval("user");
      nr = rhs.Eval("user");
      if (nl || nr)
         res.SetUser((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      return res;
   }

   /// Subtract two `RPadLength`s.
   friend RPadLengthNew operator-(RPadLengthNew lhs, const RPadLengthNew &rhs)
   {
      RPadLengthNew res;
      auto nl = lhs.Eval("normal");
      auto nr = rhs.Eval("normal");
      if (nl || nr)
         res.SetNormal((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));

      nl = lhs.Eval("pixel");
      nr = rhs.Eval("pixel");
      if (nl || nr)
         res.SetPixel((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));

      nl = lhs.Eval("user");
      nr = rhs.Eval("user");
      if (nl || nr)
         res.SetUser((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));
      return res;
   }

   /// Unary -.
   RPadLengthNew operator-()
   {
      RPadLengthNew res;
      auto nl = Eval("normal");
      if (nl) res.SetNormal(-nl->GetDouble());

      nl = Eval("pixel");
      if (nl) res.SetPixel(-nl->GetDouble());

      nl = Eval("user");
      if (nl) res.SetUser(-nl->GetDouble());
      return res;
   }

   /// Add a `RPadLength`.
   RPadLengthNew &operator+=(const RPadLengthNew &rhs)
   {
      auto nl = Eval("normal");
      auto nr = rhs.Eval("normal");
      if (nl || nr)
         SetNormal((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      nl = Eval("pixel");
      nr = rhs.Eval("pixel");
      if (nl || nr)
         SetPixel((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      nl = Eval("user");
      nr = rhs.Eval("user");
      if (nl || nr)
         SetUser((nl ? nl->GetDouble() : 0.) + (nr ? nr->GetDouble() : 0.));

      return *this;
   };

   /// Subtract a `RPadLength`.
   RPadLengthNew &operator-=(const RPadLengthNew &rhs)
   {
      auto nl = Eval("normal");
      auto nr = rhs.Eval("normal");
      if (nl || nr)
         SetNormal((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));

      nl = Eval("pixel");
      nr = rhs.Eval("pixel");
      if (nl || nr)
         SetPixel((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));

      nl = Eval("user");
      nr = rhs.Eval("user");
      if (nl || nr)
         SetUser((nl ? nl->GetDouble() : 0) - (nr ? nr->GetDouble() : 0));
      return *this;
   };

   RPadLengthNew &operator*=(double scale)
   {
      auto nl = Eval("normal");
      if (nl) SetNormal(scale*nl->GetDouble());

      nl = Eval("pixel");
      if (nl) SetPixel(scale*nl->GetDouble());

      nl = Eval("user");
      if (nl) SetUser(scale*nl->GetDouble());
      return *this;
   }

};


/*

/// User-defined literal for `RPadLength::Normal`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_normal, 0.0_normal, RLineExtent(0.2_normal, 0.5_normal));
/// ```
inline RPadLengthNew::Normal operator"" _normal(long double val)
{
   return RPadLengthNew::Normal{(double)val};
}
inline RPadLengthNew::Normal operator"" _normal(unsigned long long int val)
{
   return RPadLengthNew::Normal{(double)val};
}

/// User-defined literal for `RPadLength::Pixel`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(100_px, 0_px, RLineExtent(20_px, 50_px));
/// ```
inline RPadLengthNew::Pixel operator"" _px(long double val)
{
   return RPadLengthNew::Pixel{(double)val};
}
inline RPadLengthNew::Pixel operator"" _px(unsigned long long int val)
{
   return RPadLengthNew::Pixel{(double)val};
}

/// User-defined literal for `RPadLength::User`
///
/// Use as
/// ```
/// using namespace ROOT::Experimental;
/// RLine(0.1_user, 0.0_user, RLineExtent(0.2_user, 0.5_user));
/// ```
inline RPadLengthNew::User operator"" _user(long double val)
{
   return RPadLengthNew::User{(double)val};
}
inline RPadLengthNew::User operator"" _user(unsigned long long int val)
{
   return RPadLengthNew::User{(double)val};
}


*/


} // namespace Experimental
} // namespace ROOT

#endif
