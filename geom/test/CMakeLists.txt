# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

configure_file(no_extrusion.gdml no_extrusion.gdml COPYONLY)
ROOT_ADD_GTEST(geomTests
  test_material_units.cxx
  test_boolean_extrusion.cxx
  LIBRARIES Geom)
