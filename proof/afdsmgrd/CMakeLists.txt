############################################################################
# CMakeLists.txt file for building PROOF's afdsmgrd
# @author Dario Berzano
############################################################################

#
# Variables for afdsmgrd: all of them are *absolute* paths
#

set(AFDSMGRD_UNPACK_DIR  ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrd)
set(AFDSMGRD_INSTALL_DIR ${AFDSMGRD_UNPACK_DIR}/install)
set(AFDSMGRD_BUILD_DIR   ${AFDSMGRD_UNPACK_DIR}/build)
set(AFDSMGRD_BIN         ${AFDSMGRD_INSTALL_DIR}/bin/afdsmgrd)

# Taking MonALISA from MONALISA_DIR, if defined, or from AliEn. If neither
# MonALISA nor AliEn are enabled, don't build afdsmgrd with MonALISA support
if(monalisa)
  set(AFDSMGRD_MONALISA_DIR "${MONALISA_DIR}")
elseif(alien)
  set(AFDSMGRD_MONALISA_DIR "${ALIEN_DIR}")
endif()

# MonALISA is disabled when no MonALISA directory was given
if(AFDSMGRD_MONALISA_DIR)
  set(AFDSMGRD_MONALISA_DISABLED 0)
else()
  set(AFDSMGRD_MONALISA_DISABLED 1)
endif()

#
# Pass all possible ROOT include paths to underlying CMake
#

# Get all possible include paths into a variable (one path per line)
execute_process(
  COMMAND find "${CMAKE_SOURCE_DIR}" -maxdepth 3 -name inc -type d
  OUTPUT_VARIABLE AFDSMGRD_INCDIR_Raw
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Separate paths with colons
string(REPLACE "\n" ":" AFDSMGRD_INCDIR_Path ${AFDSMGRD_INCDIR_Raw})

#
# Build targets
#

# afdsmgrd steering target, always built (i.e., "phony", i.e., always ood). Let
# this target depend on ALL known ROOT libraries (TODO: Maybe there is a better
# way to get the list of ROOT libs?)
add_custom_target(afdsmgrd ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdInstall)

# make install on afdsmgrd
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdInstall
  COMMENT "Building afdsmgrd"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdPrepare

  COMMAND ${CMAKE_COMMAND} -E remove -f afdsmgrdInstall

  COMMAND make install -C ${AFDSMGRD_BUILD_DIR}

  # Workaround: on some configurations libdir is not created (because there is
  # no need to) causing subsequent "install" command to fail
  COMMAND ${CMAKE_COMMAND} -E make_directory ${AFDSMGRD_INSTALL_DIR}/lib

  COMMAND ${CMAKE_COMMAND} -E touch afdsmgrdInstall
)

# CMake configuration of afdsmgrd
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdPrepare
  COMMENT "Preparing afdsmgrd"
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdUnpack

  # The file afdsmgrdPrepare is created to signal a success: reset it
  COMMAND ${CMAKE_COMMAND} -E remove -f afdsmgrdPrepare

  # Create build directory
  COMMAND ${CMAKE_COMMAND} -E make_directory ${AFDSMGRD_BUILD_DIR}

  # Run cmake on the afdsmgrd project: mind the EXTRA_INCDIR variable
  COMMAND ${CMAKE_COMMAND} -E chdir ${AFDSMGRD_BUILD_DIR} ${CMAKE_COMMAND} ..
    -DCMAKE_INSTALL_PREFIX="${AFDSMGRD_INSTALL_DIR}"
    -DROOTSYS="${CMAKE_BINARY_DIR}"
    -DApMon_PREFIX="${AFDSMGRD_MONALISA_DIR}"
    -DApMon_DISABLED="${AFDSMGRD_MONALISA_DISABLED}"
    -DCMAKE_BUILD_TYPE=Release
    -DEXTRA_INCDIR="${AFDSMGRD_INCDIR_Path}"

  # The following command won't be executed if previous command has failed
  COMMAND ${CMAKE_COMMAND} -E touch afdsmgrdPrepare
)

# Unpacks afdsmgrd tar.gz found in source directory into build directory (which
# is current working directory by default)
add_custom_command(
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/afdsmgrdUnpack
  DEPENDS Core RIO Net Hist Graf Graf3d Gpad Tree Rint Postscript Matrix
    Physics MathCore Thread Proof
  COMMENT "Unpacking afdsmgrd tarball"
  COMMAND ${CMAKE_COMMAND} -E tar xzf
    ${CMAKE_CURRENT_SOURCE_DIR}/afdsmgrd-v*.tar.gz
  COMMAND ${CMAKE_COMMAND} -E touch afdsmgrdUnpack
)

#
# Install targets -- note that the trailing slashes are fundamental
#

# etc --> <prefix>/etc/proof
install(
  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/etc/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/proof
  USE_SOURCE_PERMISSIONS
)

# bin --> <prefix>/bin
install(
  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/bin/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  USE_SOURCE_PERMISSIONS
)

# lib --> <prefix>/etc/proof/lib
install(
  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/lib/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/proof/lib
  USE_SOURCE_PERMISSIONS
)

# libexec --> <prefix>/etc/proof
install(
  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/libexec/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/proof
  USE_SOURCE_PERMISSIONS
)

# var --> <prefix>/etc/proof/var
# Nothing goes here for the moment!
#install(
#  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/var/
#  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/proof/var
#  USE_SOURCE_PERMISSIONS
#)

# share --> <prefix>/etc/proof/utils/afdsmgrd
install(
  DIRECTORY ${AFDSMGRD_INSTALL_DIR}/share/
  DESTINATION ${CMAKE_INSTALL_PREFIX}/etc/proof/utils/afdsmgrd
  USE_SOURCE_PERMISSIONS
)
