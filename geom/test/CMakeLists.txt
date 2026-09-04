# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(gdml)
  configure_file(no_extrusion.gdml no_extrusion.gdml COPYONLY)
  configure_file(material_name_collision.gdml material_name_collision.gdml COPYONLY)
  ROOT_ADD_GTEST(material_units
    test_material_units.cxx
    LIBRARIES Geom)
  ROOT_ADD_GTEST(gdml_material_name_collision
    test_gdml_material_name_collision.cxx
    LIBRARIES Geom)
  ROOT_ADD_GTEST(boolean_extrusion
    test_boolean_extrusion.cxx
    LIBRARIES Geom GeomChecker)
  ROOT_ADD_GTEST(gdml_reflection
    test_gdml_reflection.cxx
    LIBRARIES Geom)
endif()
ROOT_ADD_GTEST(tessellated
  test_tessellated.cxx
  LIBRARIES Geom)

ROOT_ADD_GTEST(overlap_navigation
  test_overlap_navigation.cxx
  LIBRARIES Geom)

ROOT_ADD_GTEST(thread_navigation
  test_thread_navigation.cxx
  LIBRARIES Geom)

if(imt)
  ROOT_ADD_GTEST(manager_lifetime
    test_manager_lifetime.cxx
    LIBRARIES Geom GeomChecker Imt)
endif()
