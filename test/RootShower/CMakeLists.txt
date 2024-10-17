# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(RootShower)

find_package(ROOT REQUIRED)

set(CMAKE_CXX_FLAGS ${ROOT_CXX_FLAGS})

include_directories(${ROOT_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR})

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

set(HEADERS MyParticle.h MyDetector.h MyEvent.h)

if(MSVC)
  set(RCFILE RootShower.rc)
endif()

ROOT_GENERATE_DICTIONARY(RootShowerDict ${HEADERS} LINKDEF RSLinkDef.h)
file(COPY ${HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ".rootshowerrc" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(RootShower ${SOURCES} RootShowerDict.cxx ${RCFILE})
target_link_libraries(RootShower ROOT::Core ROOT::Rint ROOT::Gui ROOT::RIO ROOT::Hist ROOT::Gpad
  ROOT::Graf ROOT::EG ROOT::Html ROOT::Geom ROOT::Tree
)
set_target_properties(RootShower PROPERTIES ENABLE_EXPORTS 1)
if(MSVC AND NOT CMAKE_GENERATOR MATCHES Ninja)
  add_custom_command(TARGET RootShower POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/RootShower.exe
                                     ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_target_properties(RootShower PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

file(COPY anim DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY icons DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

install(TARGETS RootShower RUNTIME DESTINATION ${PROJECT_NAME} COMPONENT applications)
install(DIRECTORY anim  DESTINATION ${PROJECT_NAME}/anim COMPONENT anim)
install(DIRECTORY icons  DESTINATION ${PROJECT_NAME}/icons COMPONENT icons)
install(FILES ${HEADERS} DESTINATION ${PROJECT_NAME} COMPONENT headers)
install(FILES ".rootshowerrc" DESTINATION ${PROJECT_NAME} COMPONENT settings)
