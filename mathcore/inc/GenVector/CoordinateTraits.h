// @(#)root/mathcore:$Name:  $:$Id: CoordinateTraits.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

#ifndef COORDINATETRAITS_H
#define COORDINATETRAITS_H

// $Id: CoordinateTraits.h,v 1.2 2005/06/03 21:41:09 fischler Exp $
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
#include "GenVector/Cartesian3D.h"
#include "GenVector/CylindricalEta3D.h"
#include "GenVector/Polar3D.h"

namespace ROOT {
namespace Math {

template <class C> 
struct CoordinateTraits {
  static const std::string name() {return "NOT-A-COORDINATE-SYSTEM!";}
};

template <class Scalar>
struct CoordinateTraits < Cartesian3D<Scalar> >{
  static const std::string name() {
    std::string s = "Cartesian Coordinates <";
    s += typeid(Scalar).name();
    s += "> (X, Y, Z)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < CylindricalEta3D<Scalar> >{
  static const std::string name() {
    std::string s = "Cylindrical/Eta Coordinates <";
    s += typeid(Scalar).name();
    s += "> (Rho, Eta, Phi)";
    return s;
  }
};

template <class Scalar>
struct CoordinateTraits < Polar3D<Scalar> >{
  static const std::string name() {
    std::string s = "Polar Coordinates <";
    s += typeid(Scalar).name();
    s += "> (R, Theta, Phi)";
    return s;
  }
};


} // namespace Math 
} // namespace ROOT 

#endif // COORDINATETRAITS_H
