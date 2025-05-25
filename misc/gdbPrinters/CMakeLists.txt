# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(NOT MSVC AND _BUILD_TYPE_UPPER MATCHES "DEBUG|RELWITHDEBINFO")
  file(GLOB PRETTY_PRINTERS "*.so-gdb.py")
  if(NOT roofit)
    list(FILTER PRETTY_PRINTERS EXCLUDE REGEX libRooFitCore.so-gdb.py)
  endif()
  set(PRETTY_PRINTER_DESTS)
  foreach(PRETTY_PRINTER ${PRETTY_PRINTERS})
    get_filename_component(PRETTY_PRINTER_DEST ${PRETTY_PRINTER} NAME)
    if(soversion)
      string(REPLACE ".so-gdb.py" ".so.${ROOT_VERSION}-gdb.py"
        PRETTY_PRINTER_DEST ${PRETTY_PRINTER_DEST})
    endif(soversion)
    set(PRETTY_PRINTER_DEST ${CMAKE_BINARY_DIR}/lib/${PRETTY_PRINTER_DEST})
    add_custom_command(OUTPUT ${PRETTY_PRINTER_DEST}
      COMMAND ${CMAKE_COMMAND} -E copy ${PRETTY_PRINTER} ${PRETTY_PRINTER_DEST}
      DEPENDS ${PRETTY_PRINTER})
    list(APPEND PRETTY_PRINTER_DESTS ${PRETTY_PRINTER_DEST})
  endforeach()
  add_custom_target(copy_pretty_printers ALL DEPENDS ${PRETTY_PRINTER_DESTS})

  install(FILES ${PRETTY_PRINTER_DESTS}
          DESTINATION ${CMAKE_INSTALL_LIBDIR}
          CONFIGURATIONS Debug RelWithDebInfo)
endif()
