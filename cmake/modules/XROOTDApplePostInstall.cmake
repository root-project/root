# Post install routine needed to change the LC_RPATH variable of the XRootD
# libraries on macOS after installation. The function changes the value from
# ${build_libdir} to ${install_libdir} with the `install_name_tool` executable.
# On successive reiterations of the `cmake --install` command,
# `install_name_tool` would error out since LC_RPATH would be already set to
# ${install_libdir}. Since that is what we want in the end, we discard these
# errors.

function(xrootd_libs_change_rpath build_libdir install_libdir)

  set(XROOTD_ALL_LIBRARIES libXrdAppUtils.dylib libXrdBlacklistDecision-4.so
                           libXrdBwm-4.so libXrdCksCalczcrc32-4.so
                           libXrdCl.dylib libXrdClProxyPlugin-4.so
                           libXrdClient.dylib libXrdCmsRedirectLocal-4.so
                           libXrdCrypto.dylib libXrdCryptoLite.dylib
                           libXrdFfs.dylib libXrdFileCache-4.so
                           libXrdN2No2p-4.so libXrdOssSIgpfsT-4.so
                           libXrdPosix.dylib libXrdPosixPreload.dylib
                           libXrdPss-4.so libXrdSec-4.so libXrdSecProt-4.so
                           libXrdSeckrb5-4.so libXrdSecpwd-4.so
                           libXrdSecsss-4.so libXrdSecunix-4.so
                           libXrdServer.dylib libXrdSsi-4.so
                           libXrdSsiLib.dylib libXrdSsiLog-4.so
                           libXrdSsiShMap.dylib libXrdThrottle-4.so
                           libXrdUtils.dylib libXrdXml.dylib libXrdXrootd-4.so
  )

  find_program(INSTALL_NAME_TOOL install_name_tool)

  if(INSTALL_NAME_TOOL)
    message(STATUS "Found tool ${INSTALL_NAME_TOOL}")
    message(STATUS "Adjusting LC_RPATH variable of XRootD libraries in ${install_libdir}")

    foreach(XRD_LIB_PATH IN LISTS XROOTD_ALL_LIBRARIES)
      execute_process(COMMAND ${INSTALL_NAME_TOOL} -rpath 
                              ${build_libdir} ${install_libdir} 
                              ${install_libdir}/${XRD_LIB_PATH}
                      ERROR_QUIET
      )
    endforeach()
  endif()

endfunction()
