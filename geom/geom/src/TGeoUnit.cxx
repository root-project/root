// @(#)root/geom:$Id$
// Author: Markus Frank   25/06/19

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoUnit
\ingroup Geometry_classes

Base class describing materials.

*/
#include "TError.h"
#include "TGeoSystemOfUnits.h"
#include "TGeant4SystemOfUnits.h"

namespace   {
  static bool s_type_changed = false;
  union _unit_type {
    TGeoUnit::UnitType    tgeo_unit_type;
    TGeant4Unit::UnitType tgeant4_unit_type;
    _unit_type(TGeoUnit::UnitType t) { tgeo_unit_type = t; }
  } s_unit_type(TGeoUnit::kTGeoUnits);
}

TGeoUnit::UnitType TGeoUnit::unitType()    {
  return s_unit_type.tgeo_unit_type;
}

TGeoUnit::UnitType TGeoUnit::setUnitType(UnitType new_type)    {
  UnitType tmp = s_unit_type.tgeo_unit_type;
  if ( !s_type_changed || new_type == s_unit_type.tgeo_unit_type )    {
    s_unit_type.tgeo_unit_type = new_type;
    s_type_changed = true;
    return tmp;
  }
  Fatal("TGeoUnit","The system of units may only be changed once at the beginning of the program!");
  return tmp;
}

TGeant4Unit::UnitType TGeant4Unit::unitType()    {
  return s_unit_type.tgeant4_unit_type;
}

TGeant4Unit::UnitType TGeant4Unit::setUnitType(UnitType new_type)    {
  UnitType tmp = s_unit_type.tgeant4_unit_type;
  if ( !s_type_changed || new_type == s_unit_type.tgeant4_unit_type )    {
    s_unit_type.tgeant4_unit_type = new_type;
    s_type_changed = true;
    return tmp;
  }
  Fatal("TGeoUnit","The system of units may only be changed once at the beginning of the program!");
  return tmp;
}
