#ifdef __CLING__

#pragma link C++ struct CyclicMember;
#pragma link C++ class ClassWithUnsplitMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntupleSplit(true) class CustomStreamerForceSplit - ;
#pragma link C++ options = rntupleSplit(false) class CustomStreamerForceUnsplit + ;
#pragma link C++ class IgnoreUnsplitComment + ;

#endif
