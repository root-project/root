# if we don't have a compiler config set, try and find one:
IF (NOT REFLEX_COMPILER_CONFIG)
   INCLUDE(config/SelectCompilerConfig)
ENDIF (NOT REFLEX_COMPILER_CONFIG)

# if we have a compiler config, include it now:
IF (REFLEX_COMPILER_CONFIG)
   INCLUDE(${REFLEX_COMPILER_CONFIG})
ENDIF (REFLEX_COMPILER_CONFIG)

# if we don't have a platform config set, try and find one:
IF (NOT REFLEX_PLATFORM_CONFIG)
   INCLUDE(config/SelectPlatformConfig)
ENDIF (NOT REFLEX_PLATFORM_CONFIG)

# if we have a platfrom config, include it now:
IF (REFLEX_PLATFORM_CONFIG)
   INCLUDE(${REFLEX_PLATFORM_CONFIG})
ENDIF (REFLEX_PLATFORM_CONFIG)
