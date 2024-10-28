#ifdef __CLING__

#pragma link C++ struct CyclicMember;
#pragma link C++ class ClassWithStreamedMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntupleSplit(true) class CustomStreamerForceNative - ;
#pragma link C++ options = rntupleSplit(false) class CustomStreamerForceStreamed + ;
#pragma link C++ class IgnoreUnsplitComment + ;

#pragma link C++ class PolyBase + ;
#pragma link C++ class PolyA + ;
#pragma link C++ class PolyB + ;
#pragma link C++ options = rntupleSplit(false) class PolyContainer + ;

#endif
