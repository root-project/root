#ifndef ROTATIONTRAITS_H
#define ROTATIONTRAITS_H

// $Id: RotationTraits.h,v 1.1 2005/08/11 14:18:00 fischler Exp $
//
// Rotation traits useful for testing purposes.
//
// For example, when reporting a problem, it is nice to be able
// to present a human-readable name for the rotation.
//
// Created by: Mark Fischler  at Thu Aug 12  2005
//

#include <string>
#include <typeinfo>
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "Math/GenVector/LorentzRotation.h"

namespace ROOT {
namespace Math {

template <class C>
struct RotationTraits {
  static const std::string name() {return "NOT-A-ROTATION!";}
};

template <>
struct RotationTraits < Rotation3D >{
  static const std::string name() {
    std::string s = "Rotation3D";
   return s;
  }
};

template <>
struct RotationTraits < AxisAngle >{
  static const std::string name() {
    std::string s = "AxisAngle";
   return s;
  }
};

template <>
struct RotationTraits < EulerAngles >{
  static const std::string name() {
    std::string s = "EulerAngles";
   return s;
  }
};

template <>
struct RotationTraits < Quaternion >{
  static const std::string name() {
    std::string s = "Quaternion";
   return s;
  }
};

template <>
struct RotationTraits < RotationX >{
  static const std::string name() {
    std::string s = "RotationX";
   return s;
  }
};

template <>
struct RotationTraits < RotationY >{
  static const std::string name() {
    std::string s = "RotationY";
   return s;
  }
};

template <>
struct RotationTraits < RotationZ >{
  static const std::string name() {
    std::string s = "RotationZ";
   return s;
  }
};

template <>
struct RotationTraits < LorentzRotation >{
  static const std::string name() {
    std::string s = "LorentzRotation";
   return s;
  }
};

#ifdef TODO
template <>
struct RotationTraits < Boost >{
  static const std::string name() {
    std::string s = "Boost";
   return s;
  }
};

template <>
struct RotationTraits < BoostX >{
  static const std::string name() {
    std::string s = "BoostX";
   return s;
  }
};

template <>
struct RotationTraits < BoostY >{
  static const std::string name() {
    std::string s = "BoostY";
   return s;
  }
};

template <>
struct RotationTraits < BoostZ >{
  static const std::string name() {
    std::string s = "BoostZ";
   return s;
  }
};
#endif // TODO



} // namespace Math
} // namespace ROOT

#endif // COORDINATETRAITS_H
