#ifdef __CINT__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ class EventHeader+;
#pragma link C++ class Event+;
#pragma link C++ class HistogramManager+;
#pragma link C++ class Track+;

#pragma link C++ class vector<EventHeader>;
#pragma link C++ class vector<Track*>;

#pragma ifdef G__INTEL_COMPILER
#pragma link C++ typedef vector<EventHeader>::iterator;
#pragma link C++ typedef vector<Track*>::iterator;
#pragma else
#pragma link C++ class vector<EventHeader>::iterator-;
#pragma link C++ class vector<Track*>::iterator-;
#pragma link C++ function operator!=(vector<EventHeader>::iterator,vector<EventHeader>::iterator);
#pragma link C++ function operator!=(vector<Track*>::iterator,vector<Track*>::iterator);
#pragma endif

#endif
