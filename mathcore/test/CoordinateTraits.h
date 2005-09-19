#ifndef COORDINATETRAITS_H
#define COORDINATETRAITS_H

// $Id: CoordinateTraits.h,v 1.5 2005/08/12 15:46:52 fischler Exp $
//
// Coordinate System traits useful for testing purposes.
//
// For example, when reporting a problem, it is nice to be able
// to present a human-readable name for the system.
//
// Created by: Mark Fischler  at Mon May 30 12:21:43 2005
//
// Last update: Wed Jun 1  2005

#include <string>
#include <typeinfo>
#include "Math/GenVector/Cartesian3Dfwd.h"
#include "Math/GenVector/Polar3Dfwd.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/PtEtaPhiE4D.h"
#include "Math/GenVector/PtEtaPhiM4D.h"

namespace ROOT {
namespace Math {

template <class C> 
struct CoordinateTraits {
  static const std::string name() {
    std::string s = "NOT-A-COORDINATE-SYSTEM: ";
    s += typeid(C).name();
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < Cartesian3D<Scalar> >{
  static const std::string name() {
    std::string s = "Cartesian Coordinates <";
    s += typeid(Scalar).name();
    s += "> (x, y, z)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < CylindricalEta3D<Scalar> >{
  static const std::string name() {
    std::string s = "Cylindrical/Eta Coordinates <";
    s += typeid(Scalar).name();
    s += "> (rho, eta, phi)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < Polar3D<Scalar> >{
  static const std::string name() {
    std::string s = "Polar Coordinates <";
    s += typeid(Scalar).name();
    s += "> (r, theta, phi)";
    return s;
  }
};

  // 4D COORDINATES

template <class Scalar>
struct CoordinateTraits < PxPyPzE4D<Scalar> >{
  static const std::string name() {
    std::string s = "PxPyPzE4D Coordinates <";
    s += typeid(Scalar).name();
    s += "> (Px, Py, Pz, E)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < PtEtaPhiE4D<Scalar> >{
  static const std::string name() {
    std::string s = "PtEtaPhiE4D4D Coordinates <";
    s += typeid(Scalar).name();
    s += "> (Pt, eta, phi, E)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < PtEtaPhiM4D<Scalar> >{
  static const std::string name() {
    std::string s = "PtEtaPhiM4D4D Coordinates <";
    s += typeid(Scalar).name();
    s += "> (Pt, eta, phi, mass)";
    return s;
  }
};


} // namespace Math 
} // namespace ROOT 

#endif // COORDINATETRAITS_H
