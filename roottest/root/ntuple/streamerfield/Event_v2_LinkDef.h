#ifdef __ROOTCLING__

#pragma link C++ class StreamerBase+;
#pragma link C++ class StreamerDerived+;
#pragma link C++ options = rntupleStreamerMode(true) class StreamerContainer+;
#pragma link C++ class Event+;

#endif
