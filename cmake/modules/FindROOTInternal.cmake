macro(check_value_ok _value)
  # check _value (TRUE, ENABLE, 1 -> TRUE)
  #              (FALSE, DISABLE, 0 -> FALSE)
  set (true_allowed TRUE ENABLE 1)
  set (false_allowed FALSE DISABLE 0)
  string(TOUPPER ${_value} _value_upper)
  set (true_value FALSE)
  set (false_value FALSE)
  foreach(_true ${true_allowed})
    if (${_value_upper} MATCHES ${_true})
      set(true_value TRUE)
    endif (${_value_upper} MATCHES ${_true})
  endforeach(_true ${true_allowed})
  foreach(_false ${false_allowed})
    if (${_value_upper} MATCHES ${_false})
      set(false_value TRUE)
    endif (${_value_upper} MATCHES ${_false})
  endforeach(_false ${false_allowed})
  if (true_value OR false_value)
    if (true_value AND false_value)
       MESSAGE(FATAL_ERROR "check_value_ok: Value for option is true and false. This can never happen. If this error message is plotted something is wrong with the logic of the macro.") 
    endif(true_value AND false_value)
  else (true_value OR false_value)
    MESSAGE("check_value_ok: This value for options is not known.")
    MESSAGE("check_value_ok: Allowed value for TRUE are TRUE, Enable or 1")
    MESSAGE(FATAL_ERROR "check_value_ok: Allowed value for FALSE are FALSE, Disable or 0")
  endif (true_value OR false_value)
endmacro(check_value_ok _value)

macro(show_root_install_options)
  foreach(_actual_option ${ROOT_INSTALL_OPTIONS})
    list(FIND ROOT_INSTALL_OPTIONS ${_actual_option} position)
    list(GET ROOT_INSTALL_OPTIONS ${position} __option)
    list(GET ROOT_INSTALL_OPTIONS_VALUE ${position} __value)
    MESSAGE("${__option}": ${__value})
  endforeach(_actual_option ${ROOT_INSTALL_OPTIONS})
endmacro(show_root_install_options)

macro(is_root_install_option_enabled _option)
    list(FIND ROOT_INSTALL_OPTIONS ${_option} position)
    list(GET ROOT_INSTALL_OPTIONS_VALUE ${position} __value)
    if(__value)
      set (BLA option_${_option}_is_enabled)
      set(${BLA} TRUE)
    else(__value)
      set (BLA option_${_option}_is_enabled)
      set(${BLA} FALSE)
    endif(__value)
endmacro(is_root_install_option_enabled _option)

macro(get_enabled_root_install_options)
  set(installed_options_list)
  foreach(_actual_option ${ROOT_INSTALL_OPTIONS})
    list(FIND ROOT_INSTALL_OPTIONS ${_actual_option} position)
    list(GET ROOT_INSTALL_OPTIONS ${position} __option)
    list(GET ROOT_INSTALL_OPTIONS_VALUE ${position} __value)
    if(${__value} MATCHES TRUE)
      set(installed_options_list "${installed_options_list} ${__option}")
    endif(${__value} MATCHES TRUE)
  endforeach(_actual_option ${ROOT_INSTALL_OPTIONS})

endmacro(get_enabled_root_install_options)


macro(CHANGE_ROOT_INSTALL_OPTIONS _option _value) 
  # get position of option from array ROOT_INSTAL_OPTIONS
  list(FIND ROOT_INSTALL_OPTIONS ${_option} position)
  if (position EQUAL -1)
    MESSAGE(FATAL_ERROR "CHANGE_ROOT_INSTALL_OPTIONS: The option ${_option} is not known. \n Possible options are ${ROOT_INSTALL_OPTIONS}")
  endif (position EQUAL -1)

  #check if the value to be set is okay
  check_value_ok(${_value})

  #Now set the value _value for option _option 
  if (true_value)
    list(INSERT ROOT_INSTALL_OPTIONS_VALUE ${position} TRUE)
    math(EXPR rel_pos ${position}+1)
    list(REMOVE_AT ROOT_INSTALL_OPTIONS_VALUE ${rel_pos})
  endif (true_value)
  if (false_value)
    list(INSERT ROOT_INSTALL_OPTIONS_VALUE ${position} FALSE)
    math(EXPR rel_pos ${position}+1)
    list(REMOVE_AT ROOT_INSTALL_OPTIONS_VALUE ${rel_pos})
  endif (false_value)
    
  #check if after the operation the length of ROOT_INSTALL_OPTIONS and ROOT_INSTALL_OPTIONS_VALUE
  #is still the same
  list(LENGTH ROOT_INSTALL_OPTIONS_VALUE length_value)  
  list(LENGTH ROOT_INSTALL_OPTIONS length_options)  
  if(NOT ${length_value} EQUAL ${length_options})
    MESSAGE(FATAL_ERROR "CHANGE_ROOT_INSTALL_OPTIONS: After changing the option the length of both arrays is different (${length_value} NEQ ${length_options}). Something is wrong with this operation")
  endif(NOT ${length_value} EQUAL ${length_options})

endmacro(CHANGE_ROOT_INSTALL_OPTIONS _option _value) 


# define all packages to be build. By default all of them are enabled at start.
# We switch them off later on during the process of configuration.

set (ROOT_INSTALL_OPTIONS
   afs                 
   alien               
   asimage             
   astiff              
   builtin_afterimage  
   builtin_ftgl        
   builtin_freetype    
   builtin_pcre        
   builtin_zlib        
   castor              
   chirp               
   cint7               
   cintex              
   clarens             
   dcache              
   exceptions          
   explicitlink        
   fftw3               
   gdml                
   gfal                
   g4root              
   glite               
   globus              
   gsl_shared          
   krb5                
   ldap                
   genvector           
   mathmore            
   memstat             
   monalisa            
   mysql               
   odbc                
   opengl              
   oracle              
   pch                 
   peac                
   pgsql               
   pythia6             
   pythia8             
   python              
   qt                  
   qtgsi               
   reflex              
   roofit              
   minuit2             
   ruby                
   rfio                
   rpath               
   sapdb               
   shadowpw            
   shared              
   soversion           
   srp                 
   ssl                 
   table              
   unuran              
   winrtdebug          
   xft                 
   xml                 
   xrootd              
)

set (ROOT_INSTALL_OPTIONS_VALUE
  TRUE                    
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
  TRUE   
)

CHANGE_ROOT_INSTALL_OPTIONS("afs" "enable")
CHANGE_ROOT_INSTALL_OPTIONS("afs" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("cint7" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("gdml" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("globus" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("explicitlink" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("pch" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("qt" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("qtgsi" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("roofit" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("minuit2" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("rpath" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("ruby" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("shadowpw" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("soversion" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("table" "disable")
list(APPEND ROOT_INSTALL_OPTIONS "thread")
list(APPEND ROOT_INSTALL_OPTIONS_VALUE "TRUE")
CHANGE_ROOT_INSTALL_OPTIONS("unuran" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("winrtdebug" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("xrootd" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("winrtdebug" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("builtin_freetype" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("builtin_ftgl" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("builtin_pcre" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("builtin_zlib" "disable")
CHANGE_ROOT_INSTALL_OPTIONS("mathmore" "disable")

#show_root_install_options()

#set all directories where to install parts of root
#up to now everything is installed according to the setting of
#CMAKE_INSTALL_DIR
#TODO: Make installation layout more flexible

if(ROOT_INSTALL_DIR)
  set(CMAKE_INSTALL_PREFIX ${ROOT_INSTALL_DIR})
  add_definitions(-DR__HAVE_CONFIG)
else(ROOT_INSTALL_DIR)
  set(CMAKE_INSTALL_PREFIX ${ROOTSYS})
endif(ROOT_INSTALL_DIR)

set(ROOT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(BIN_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib")
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include")
set(ETC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/etc")
set(DATA_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(DOC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}")
set(MACRO_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/macros")
set(SRC_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/src")
set(ICON_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/icons")
set(FONT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/fonts")
set(CINT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cint")

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

# the RPATH to be used when installing
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#MESSAGE("RPATH: ${CMAKE_INSTALL_RPATH}")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

