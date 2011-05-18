IF (MSVC80 OR MSVC90)
   ADD_DEFINITIONS(-D_CRT_SECURE_NO_DEPRECATE -D_CRT_NONSTDC_NO_DEPRECATE -D_SCL_SECURE_NO_DEPRECATE)
ENDIF (MSVC80 OR MSVC90)

# enable structured exception handling and disable the following warnings:
# warning C4800: 'const unsigned int' : forcing value to bool 'true' or 'false' (performance warning)
SET(REFLEX_CXX_FLAGS "/EHsc /wd4800")

# enable /MP (Build with Multiple Processes) for VC9. It was buggy and undocumented for previous versions.
IF (MSVC90)
   SET(REFLEX_CXX_FLAGS "${REFLEX_CXX_FLAGS} /MP")
ENDIF (MSVC90)
