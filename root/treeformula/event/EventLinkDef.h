#ifdef __CINT__

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ class EventHeader+;
#pragma link C++ class Event+;
#pragma link C++ class HistogramManager+;
#pragma link C++ class Track+;

#pragma link C++ class vector<EventHeader>;
#pragma link C++ class vector<Track*>;

#ifdef __INTEL_COMPILER
#pragma link C++ typedef vector<EventHeader>::iterator;
#pragma link C++ typedef vector<Track*>::iterator;
#elseif _WIN32
// Intentionally empty because of vc8 (we do not compile the dictionary in
// this dictionary with the 'right' switches.
#else
#pragma link C++ class vector<EventHeader>::iterator-;
#pragma link C++ class vector<Track*>::iterator-;
#pragma link C++ function operator!=(vector<EventHeader>::iterator,vector<EventHeader>::iterator);
#pragma link C++ function operator!=(vector<Track*>::iterator,vector<Track*>::iterator);
#endif

#endif
