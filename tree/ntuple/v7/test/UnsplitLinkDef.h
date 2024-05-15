#ifdef __CLING__

#pragma link C++ struct CyclicMember;
#pragma link C++ class ClassWithUnsplitMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntupleSplit(true) class CustomStreamerForceSplit - ;
#pragma link C++ options = rntupleSplit(false) class CustomStreamerForceUnsplit + ;
#pragma link C++ class IgnoreUnsplitComment + ;

#pragma link C++ class PolyBase + ;
#pragma link C++ class PolyA + ;
#pragma link C++ class PolyB + ;
#pragma link C++ options = rntupleSplit(false) class PolyContainer + ;

#endif
