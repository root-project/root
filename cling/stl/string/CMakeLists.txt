# Add define to fix test.
add_definitions(-DClingWorkAroundMissingDynamicScope)

ROOT_ADD_ROOTTEST(cling-stl-string-stringPassingText MACRO run.C OUTREF passing.ref)
