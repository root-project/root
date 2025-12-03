ROOTTEST_GENERATE_DICTIONARY(CandCluster
                             AlgClusterSRList.h CandStripHandle.h
                             SOURCES AlgClusterSRList.cxx CandStripHandle.cxx
                             LINKDEF LinkDef.h
                             COMPILE_OPTIONS -DUNIX -DLINUX -D__UNIX__ -D__LINUX__ -DDATAREP_LITTLE_IEEE -DDATAREP_LITTLE_ENDIAN -DDEFECT_NO_IOSTREAM_NAMESPACES -DDEFECT_NO_JZEXT -DDEFECT_NO_INTHEX -DDEFECT_NO_INTHOLLERITH -DDEFECT_NO_READONLY -DDEFECT_NO_DIRECT_FIXED -DDEFECT_NO_STRUCTURE  -DUSE_ROOT
                             NO_ROOTMAP NO_CXXMODULE)
