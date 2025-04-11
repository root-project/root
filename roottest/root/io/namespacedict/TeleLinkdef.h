#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

//Added
//# pragma link C++ nestedclasses;

//# pragma link C++ namespace Sash;
//# pragma link C++ namespace Reco;

//# pragma link C++ class Sash::Telescope;
//#pragma link C++ class A::Class2<A::Class1>!-;
//#pragma link C++ class B::Class3+;

#pragma link C++ class Sash::EnvelopeEntry<Sash::Telescope,Reco::HillasParameters>!;
#pragma link C++ class Sash::EnvelopeEntrySimple<Sash::Telescope,Reco::HillasParameters>+;

#endif
