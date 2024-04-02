#ifdef __CLING__

#pragma link C++ class ClassWithUnsplitMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntuplesplit class CustomStreamerForceSplit - ;
#pragma link C++ options = rntupleunsplit class CustomStreamerForceUnsplit + ;
#pragma link C++ class IgnoreUnsplitComment + ;

#endif
