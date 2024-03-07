#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class ClassWithUnsplitMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntuplesplit class CustomStreamerForceSplit - ;
#pragma link C++ options = rntupleunsplit class CustomStreamerForceUnsplit + ;

#endif
