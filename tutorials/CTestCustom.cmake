#---Custom CTest settings---------------------------------------------------
if(WIN32)
  list(APPEND CTEST_CUSTOM_TESTS_IGNORE tutorial-roofit-rf509_wsinteractive)   #  workspaces as namespace does not work on windows 
endif()
