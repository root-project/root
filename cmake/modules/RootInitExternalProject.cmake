MACRO(ROOT_INIT_EXTERNAL_PROJECT project_name)
	cmake_minimum_required(VERSION 3.28)
	project(${project_name})
	set(CMAKE_CXX_STANDARD 17)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)

	set(CMAKE_MODULE_PATH ${ROOT_GLOBAL_SOURCE_DIR}/cmake/modules/)

	#if(ROOT_GLOBAL_BINARY_DIR)
	#	set(CMAKE_INSTALL_INCLUDEDIR ${ROOT_GLOBAL_BINARY_DIR}/include)
	#endif()

	#if(CMAKE_BINARY_DIR)
	#	set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_BINARY_DIR}/include)
	#endif()

	#include(SearchInstalledSoftware)
	include(SetROOTVersion)

	#include(${ROOT_GLOBAL_BINARY_DIR}/ROOTConfig.cmake)
	include(${ROOT_GLOBAL_SOURCE_DIR}/cmake/modules/RootInstallDirs.cmake)
	include(${ROOT_GLOBAL_SOURCE_DIR}/cmake/modules/CheckCompiler.cmake)
	include(${ROOT_GLOBAL_SOURCE_DIR}/cmake/modules/RootMacros.cmake)
	include(${ROOT_GLOBAL_SOURCE_DIR}/cmake/modules/RootBuildOptions.cmake)

	#---Set paths where to put the libraries, executables and headers------------------------------
	file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # prevent mkdir races
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

	# RPM
	#set(VERSION "1.0.1")
	set(VERSION "${ROOT_FULL_VERSION}")
	# <----snip your usual build instructions snip--->
	set(CPACK_PACKAGE_VERSION ${VERSION})
	set(CPACK_GENERATOR "RPM")
	string(REPLACE "::" "-" RPMNAME "${project_name}")
	message(STATUS ${RPMNAME})
	#set(CPACK_PACKAGE_NAME "${project_name}")
	set(CPACK_PACKAGE_NAME "${RPMNAME}")
	set(CPACK_PACKAGE_RELEASE 1)
	#set(CPACK_PACKAGE_CONTACT "John Explainer")
	set(CPACK_PACKAGE_VENDOR "CERN")
	set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
	set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

SET(CPACK_RPM_POST_INSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/post.sh")
SET(CPACK_RPM_POST_UNINSTALL_SCRIPT_FILE "${CMAKE_BINARY_DIR}/postun.sh")
	include(CPack)
	
ENDMACRO()
