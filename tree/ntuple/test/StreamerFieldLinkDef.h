#ifdef __CLING__

#pragma link C++ struct TrivialStreamedClass + ;
#pragma link C++ struct ContainsTrivialStreamedClass + ;
#pragma link C++ struct ConstructorStreamedClass + ;
#pragma link C++ struct ContainsConstructorStreamedClass + ;
#pragma link C++ struct DestructorStreamedClass + ;
#pragma link C++ struct ContainsDestructorStreamedClass + ;

#pragma link C++ struct CyclicMember;
#pragma link C++ class ClassWithStreamedMember + ;
#pragma link C++ class CustomStreamer - ;
#pragma link C++ options = rntupleStreamerMode(false) class CustomStreamerForceNative - ;
#pragma link C++ options = rntupleStreamerMode(true) class CustomStreamerForceStreamed + ;
#pragma link C++ class IgnoreUnsplitComment + ;

#pragma link C++ class PolyBase + ;
#pragma link C++ class PolyA + ;
#pragma link C++ class PolyB + ;
#pragma link C++ options = rntupleStreamerMode(true) class PolyContainer + ;

#pragma link C++ options = rntupleStreamerMode(true), version(3) class OldStreamerName < int> + ;
#pragma link C++ options = rntupleStreamerMode(true), version(3) class NewStreamerName < int> + ;
#pragma read sourceClass = "OldStreamerName<int>" targetClass = "NewStreamerName<int>" version = "[3]"

#pragma link C++ options = rntupleStreamerMode(true) class TemperatureCelsius + ;
#pragma link C++ options = rntupleStreamerMode(true) class TemperatureKelvin + ;

#endif
