#ifdef __ROOTCLING__

#pragma link C++ class StreamerBase+;
#pragma link C++ class StreamerDerived+;
#pragma link C++ options = rntupleStreamerMode(true) class StreamerContainer+;
#pragma link C++ class Event+;

#pragma read sourceClass = "StreamerBase" source = "int fBase;" version = "[2]" \
    targetClass = "StreamerBase" target = "fBase" \
    code = "{ fBase = onfile.fBase + 1000; }"

#endif
