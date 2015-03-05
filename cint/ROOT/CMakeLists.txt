ROOT_USE_PACKAGE(core/base)
ROOT_USE_PACKAGE(core/meta)
ROOT_USE_PACKAGE(core/clib)
ROOT_USE_PACKAGE(core/cont)
ROOT_USE_PACKAGE(core/thread)
ROOT_USE_PACKAGE(cint/cint)

include_directories(${CMAKE_BINARY_DIR}/include)   # All headers have been moved

set(CINTSTLDLLNAMES
    vector 
    vectorbool 
    list 
    deque 
    map 
    map2 
    set 
    multimap 
    multimap2 
    multiset 
    stack 
    queue 
    exception 
    stdexcept 
    climits 
    complex
    stdcxxfunc
   )
set(CINTDLLDIR ${CMAKE_SOURCE_DIR}/cint/cint/lib/dll_stl/)
set(CINTSTLDLLHEADERS
    ${CINTDLLDIR}/vec.h
    ${CINTDLLDIR}/vecbool.h 
    ${CINTDLLDIR}/lst.h 
    ${CINTDLLDIR}/dqu.h 
    ${CINTDLLDIR}/mp.h 
    ${CINTDLLDIR}/mp.h 
    ${CINTDLLDIR}/st.h 
    ${CINTDLLDIR}/multmp.h 
    ${CINTDLLDIR}/multmp.h 
    ${CINTDLLDIR}/multst.h 
    ${CINTDLLDIR}/stk.h 
    ${CINTDLLDIR}/que.h 
    ${CINTDLLDIR}/cinteh.h 
    ${CINTDLLDIR}/se.h 
    ${CINTDLLDIR}/clim.h 
    ${CINTDLLDIR}/cmplx.h
    ${CMAKE_SOURCE_DIR}/cint/cint/lib/stdstrct/stdcxxfunc.h
  )
set(CINTINCDLLNAMES 
    stdfunc 
#    stdcxxfunc
   )
set(CINTINCDLLHEADERS
    ${CMAKE_SOURCE_DIR}/cint/cint/lib/stdstrct/stdfunc.h 
#    ${CMAKE_SOURCE_DIR}/cint/cint/lib/stdstrct/stdcxxfunc.h
   )
if(NOT WIN32)
  set(CINTSTLDLLNAMES ${CINTSTLDLLNAMES}  valarray)
  set(CINTSTLDLLHEADERS ${CINTSTLDLLHEADERS} ${CINTDLLDIR}/vary.h)
  set(CINTINCDLLNAMES ${CINTINCDLLNAMES} posix ipc)
  set(CINTINCDLLHEADERS ${CINTINCDLLHEADERS} 
    ${CMAKE_SOURCE_DIR}/cint/cint/lib/posix/exten.h 
    ${CMAKE_SOURCE_DIR}/cint/cint/lib/ipc/ipcif.h)
endif()

set(CINTBUILDLOADER
    vector 
    list 
    deque 
    map 
    map2 
    set 
    multimap 
    multimap2 
    multiset 
    valarray
    complex
)

set(CINTDLLCXXFLAGS "${CINT_CXX_DEFINITIONS} -DG__HAVE_CONFIG -DG__NOMAKEINFO")
set(CINTDLLCFLAGS "${CINT_C_DEFINITIONS} -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__SYSTYPES")

if(MACOSX_TMPNAM_DEPRECATED OR CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-Wno-deprecated-declarations)
endif()

set(IncludeDirs -I${CMAKE_SOURCE_DIR}/cint/cint/lib/dll_stl  -I${CMAKE_SOURCE_DIR}/cint/cint/lib -I${HEADER_OUTPUT_PATH})

foreach(_name ${CINTSTLDLLNAMES}) 
  set(OutFileName ${CMAKE_CURRENT_BINARY_DIR}/G__cpp_${_name}.cxx)
  set(LoaderName ${CMAKE_CURRENT_BINARY_DIR}/stlLoader_${_name}.cc)
  list(FIND CINTSTLDLLNAMES ${_name} _position)
  list(GET CINTSTLDLLHEADERS ${_position} InFileName)

  if(${_name} MATCHES map2)
    set(AdditionalFlags ${AdditionalFlags} "-DG__MAP2")
  endif()
  if(cxx14)
    set(AdditionalFlags ${AdditionalFlags} "-DG__CXX14")
  endif()

  add_Custom_Command(OUTPUT ${OutFileName} 
                     COMMAND cint_tmp -w1 -z${_name} -n${OutFileName} ${AdditionalFlags} -D__MAKECINT__ -DG__MAKECINT ${IncludeDirs} -c-1 -A -Z0 ${InFileName} ${AdditionalHeaderFiles}
                     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                     DEPENDS ${CINTDEP}
                    ) 
                    
  list(FIND CINTBUILDLOADER ${_name} _position)
  if(NOT _position EQUAL -1)
    add_Custom_Command(OUTPUT ${LoaderName}
                       COMMAND cmake -E copy_if_different ${CMAKE_SOURCE_DIR}/core/metautils/src/stlLoader.cc  ${LoaderName}
                      )

    set_Source_Files_Properties(${LoaderName} PROPERTIES
                                COMPILE_FLAGS "${CINTDLLCXXFLAGS} -I${CMAKE_SOURCE_DIR}/cint/cint/inc"
                                COMPILE_DEFINITIONS WHAT="${_name}"
                               )
    set(Dictionary ${CMAKE_CURRENT_BINARY_DIR}/rootcint_${_name})

    if(${_name} MATCHES map)
      set(_name1 map)
    elseif(${_name} MATCHES set)
      set(_name1 set)
    else()
      set(_name1 ${_name})
    endif()
    set(linkdeffile "${CMAKE_SOURCE_DIR}/core/metautils/src/${_name}Linkdef.h")
    
    #ROOT_GENERATE_DICTIONARY( ${Dictionary} ${_name1} LINKDEF ${linkdeffile})
    add_custom_command(OUTPUT ${Dictionary}.cxx ${Dictionary}.h
                       COMMAND rootcint_tmp -cint -f ${Dictionary}.cxx -c 
                               -I${CMAKE_CURRENT_SOURCE_DIR}/inc 
                               -I${CMAKE_BINARY_DIR}/include
                               -I${CMAKE_SOURCE_DIR}/cint/cint/include 
                               -I${CMAKE_SOURCE_DIR}/cint/cint/stl 
                               -I${CMAKE_SOURCE_DIR}/cint/cint/lib
                               -I${CMAKE_SOURCE_DIR}/core/base/inc
                               -I${CMAKE_SOURCE_DIR}/core/meta/inc 
                               -I${CMAKE_SOURCE_DIR}/core/clib/inc 
                               ${_name1} ${linkdeffile} 
                       DEPENDS ${linkdeffile} ${ROOTCINTDEP})

    set_source_files_properties(${Dictionary}.cxx 
                                 COMPILE_FLAGS "${CINTDLLCXXFLAGS} -DG__CINTBODY -I${HEADER_OUTPUT_PATH}"
                                )                                
                                
    add_Library(${_name}Dict SHARED ${Dictionary}.cxx )
    set_Target_properties(${_name}Dict PROPERTIES PREFIX ${libprefix} SUFFIX ${libsuffix} FOLDER Dictionaries MACOSX_RPATH 1)
    install(TARGETS ${_name}Dict RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                                 LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                                 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

    if(explicitlink)
      target_link_libraries(${_name}Dict Cint Core)
    endif()
    if(_name STREQUAL vector) 
      ROOT_GENERATE_ROOTMAP( ${_name}Dict LIBRARY vector.dll DEPENDENCIES vectorbool.dll LINKDEF ${linkdeffile})
    else()
      ROOT_GENERATE_ROOTMAP( ${_name}Dict LIBRARY ${_name}.dll LINKDEF ${linkdeffile})
    endif()
  else()
    set(LoaderName)
  endif()

  add_Library(${_name} SHARED ${LoaderName} ${OutFileName} )
  set_target_properties(${_name} PROPERTIES PREFIX "" SUFFIX ".dll" FOLDER Dictionaries MACOSX_RPATH 1)
  if(explicitlink)
    target_link_libraries(${_name} Cint)
  endif()
  install(TARGETS ${_name} DESTINATION ${CMAKE_INSTALL_CINTINCDIR}/cint/stl)
endforeach()


foreach(_name ${CINTINCDLLNAMES}) 
  set(OutFileName ${CMAKE_CURRENT_BINARY_DIR}/G__c_${_name}.c)
 
  list(FIND CINTINCDLLNAMES ${_name} _position)
  list(GET CINTINCDLLHEADERS ${_position} InFileName)

  if(_name MATCHES stdfunc)
    add_custom_command(OUTPUT ${OutFileName} 
                       COMMAND cint_tmp -K -w1 -z${_name} -n${OutFileName} -D__MAKECINT__ -DG__MAKECINT ${IncludeDirs} -c-2 -Z0 ${InFileName} ${AdditionalHeaderFiles}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       DEPENDS ${CINTDEP}
                      ) 

    set_source_files_properties(${OutFileName} 
                                COMPILE_FLAGS "${CINTDLLCFLAGS} -I${CMAKE_SOURCE_DIR}/cint/cint/inc"
                               )
    add_Library(${_name} SHARED ${OutFileName})
    set_target_properties(${_name} PROPERTIES PREFIX "" SUFFIX ".dll" FOLDER Dictionaries MACOSX_RPATH 1)
    if(explicitlink)
      target_link_libraries(${_name} Cint ${CMAKE_M_LIBS})
    endif()
    install(TARGETS ${_name} DESTINATION ${CMAKE_INSTALL_CINTINCDIR}/cint/include)

  elseif(_name MATCHES ipc)
    add_custom_command(OUTPUT ${OutFileName} 
                       COMMAND cint_tmp -K -w1 -z${_name} -n${OutFileName} -D__MAKECINT__ -DG__MAKECINT ${IncludeDirs} -c-2 -Z0 ${InFileName} ${AdditionalHeaderFiles}
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       DEPENDS ${CMAKE_BINARY_DIR}/cint/cint/include/sys/types.h ${CINTDEP}
                      ) 

    set_source_files_properties(${OutFileName} 
                                COMPILE_FLAGS "${CINTDLLCFLAGS} -I${CMAKE_SOURCE_DIR}/cint/cint/inc"
                               )

    add_Library(${_name} SHARED ${OutFileName})
    set_target_properties(${_name} PROPERTIES PREFIX "" SUFFIX ".dll" FOLDER Dictionaries MACOSX_RPATH 1)
    if(explicitlink)
      target_link_libraries(${_name} Cint)
    endif()
    install(TARGETS ${_name} DESTINATION ${CMAKE_INSTALL_CINTINCDIR}/cint/include)

  elseif(_name MATCHES posix)
    
    add_executable(mktypes ${CMAKE_SOURCE_DIR}/cint/cint/lib/posix/mktypes.c)
    add_custom_command(OUTPUT ${HEADER_OUTPUT_PATH}/systypes.h 
                       COMMAND mktypes
                       DEPENDS mktypes
                      ) 
    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/cint/cint/include/sys/types.h 
                       COMMAND cmake -E copy_if_different 
                               ${HEADER_OUTPUT_PATH}/systypes.h
                               ${CMAKE_BINARY_DIR}/cint/cint/include/sys/types.h
                       COMMAND cmake -E copy_if_different 
                               ${HEADER_OUTPUT_PATH}/systypes.h
                               ${CMAKE_BINARY_DIR}/cint/cint/include/systypes.h
                       DEPENDS ${HEADER_OUTPUT_PATH}/systypes.h
                      ) 

    if(MACOSX_MINOR GREATER 4)
      set(_ExtraFlag "-D__DARWIN_UNIX03")
    endif()

    add_custom_command(OUTPUT ${OutFileName} 
                       COMMAND cint_tmp -K -w1 -z${_name} -n${OutFileName} -D__MAKECINT__ -DG__MAKECINT ${_ExtraFlag} -c-2 -Z0 ${InFileName} ${AdditionalHeaderFiles} ${CMAKE_BINARY_DIR}/cint/cint/include/sys/types.h ${CMAKE_SOURCE_DIR}/cint/cint/lib/posix/posix.h
                       WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                       DEPENDS ${CMAKE_BINARY_DIR}/cint/cint/include/sys/types.h ${CINTDEP}
                      ) 
    set_source_files_properties(${OutFileName} 
                                COMPILE_FLAGS "${CINT_C_DEFINITIONS}  -DG__HAVE_CONFIG -DG__NOMAKEINFO -I${CMAKE_SOURCE_DIR}/cint/cint/inc -I${CMAKE_SOURCE_DIR}/cint/cint/src -I${CMAKE_SOURCE_DIR}/cint/cint/src/dict -DG__SYSTYPES_H" 
                               )
    set_source_files_properties(${CMAKE_SOURCE_DIR}/cint/cint/lib/posix/exten.c
                                COMPILE_FLAGS " -DG__HAVE_CONFIG -DG__NOMAKEINFO -DG__CINTBODY"
                               )

    add_library(${_name} SHARED ${OutFileName} ${CMAKE_SOURCE_DIR}/cint/cint/lib/posix/exten.c)
    set_target_properties(${_name} PROPERTIES PREFIX "" SUFFIX ".dll" FOLDER Dictionaries MACOSX_RPATH 1)
    if(explicitlink)
      target_link_libraries(${_name} Cint)
    endif()
    install(TARGETS ${_name} DESTINATION ${CMAKE_INSTALL_CINTINCDIR}/cint/include)
  endif()
endforeach()
