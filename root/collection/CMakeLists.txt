#---Copy from source to binary some of the files
configure_file(HistArray.root . COPYONLY)

ROOTTEST_ADD_AUTOMACROS()
