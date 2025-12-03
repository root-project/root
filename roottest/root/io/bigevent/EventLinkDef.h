#ifdef __CINT__

//#ppragma link off all globals;
//#ppragma link off all classes;
//#ppragma link off all functions;

#pragma link C++ class EventHeader+;
#pragma link C++ class Event+;
#pragma link C++ class HistogramManager;
#pragma link C++ class Track+;
#pragma link C++ class BigTrack+;
#pragma link C++ class UShortVector+;

#pragma link C++ class template1<int>+;
#pragma link C++ class template2< template1<int> >+;

//#ppragma link C++ class vector<TAxis*>;
//#ppragma link C++ class pair<float,float>;
#endif
