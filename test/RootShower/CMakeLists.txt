# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(RootShower)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR})

set(HEADERS
  GButtonFrame.h
  GTitleFrame.h
  MyDetector.h
  MyEvent.h
  MyParticle.h
  ParticlesDef.h
  RootShower.h
  RSAbout.h
  RSHelpText.h
  RSMsgBox.h
  RSVersion.h
  SettingsDlg.h
)

set(SOURCES
  GButtonFrame.cxx
  GTitleFrame.cxx
  MyDetector.cxx
  MyEvent.cxx
  MyParticle.cxx
  RootShower.cxx
  RSAbout.cxx
  RSHelpText.cxx
  RSMsgBox.cxx
  SettingsDlg.cxx
)

if(MSVC)
  set(RCFILE RootShower.rc)
endif()

ROOT_GENERATE_DICTIONARY(RootShowerDict MyParticle.h MyDetector.h MyEvent.h LINKDEF RSLinkDef.h)

add_executable(RootShower ${SOURCES} RootShowerDict.cxx ${RCFILE})
target_link_libraries(RootShower ROOT::Core ROOT::Rint ROOT::Gui ROOT::RIO ROOT::Hist ROOT::Gpad
  ROOT::Graf ROOT::EG ROOT::Html ROOT::Geom ROOT::Tree
)
set_target_properties(RootShower PROPERTIES ENABLE_EXPORTS 1)
if(MSVC)
  add_custom_command(TARGET RootShower POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/RootShower.exe
                                     ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_target_properties(RootShower PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()
