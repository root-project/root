#ifdef __CINT__

#pragma link C++ class EventHeader+;
#pragma link C++ class Event+;
#pragma link C++ class HistogramManager+;
#pragma link C++ class Track+;

#pragma link C++ nestedclasses;
#pragma link C++ nestedtypedef;

#pragma link C++ class vector<EventHeader>;
#pragma link C++ class vector<EventHeader>::iterator-;
#pragma link C++ class vector<Track*>;
#pragma link C++ class vector<Track*>::iterator-;
#pragma link C++ function operator!=(vector<EventHeader>::iterator,vector<EventHeader>::iterator);
#pragma link C++ function operator!=(vector<Track*>::iterator,vector<Track*>::iterator);

#endif
