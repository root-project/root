#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// #pragma link C++ class crap+;

#pragma link C++ class RootPCtemp<int>+;

#pragma link C++ class RootPCtempObj<TObject>+;
#pragma link C++ class RootPCtempObj<TObject*>+;
#pragma link C++ class RootPCtempObj<const TObject>+;
//#pragma link C++ class RootPCtempObj<const TObject *>+;
#pragma link C++ class RootPCtempObj<TObject * const>+;

#pragma link C++ class RootData+;
#pragma link C++ class RootCaloHit+;
#pragma link C++ class RootPCellID+;

#pragma link C++ class RootPCfix+;
#pragma link C++ class RootPCvirt+;

//#pragma link C++ class vector<int>;

//#pragma link C++ class NotAClass!;

#pragma link C++ namespace ROOT;
//#pragma link C++ function ROOT::ShowMembers(RootPCellID*, TMemberInspector&, char*);

#pragma link C++ nestedclass;
#pragma link C++ namespace Local;
#pragma link C++ class Local::RootPCtop+;
#pragma link C++ class Local::RootPCbottom+;

#pragma link C++ class RootPCobject+;
#pragma link C++ class RootPCobject2+;

#pragma link C++ class RootPCmisClDef+;

#pragma link C++ class RootPrivPCobject+;
#pragma link C++ class RootPrivPCobject2+;
#pragma link C++ class RootPrivPC+;

#endif
