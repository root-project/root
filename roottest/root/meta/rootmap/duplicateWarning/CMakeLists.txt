configure_file(rfile1.rootmap . COPYONLY)
configure_file(rfile2.rootmap . COPYONLY)

ROOTTEST_ADD_TEST(myclassduplicated
                  COMMAND ${ROOT_root_CMD} -b -q -l
                  PASSREGEX  "Warning in <TInterpreter::ReadRootmapFile>: While processing ./rfile[12]\.rootmap, class myclassduplicated was found to be associated to libdup[12]\.so although it is already associated to libdup[12]\.so")
