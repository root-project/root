#include "TFile.h"
#include "JansEvent.h"
#include "TTree.h"
#include "Riostream.h"
void runJan() {
//gROOT->LoadMacro( "JansEvent.C+" );
{
B_Parameters *b = new B_Parameters;
TObject *o = b;
CandidateParameters *c = b;
//std::cout << (void*)b << " : " << (void*)c << " and " << (void*)o << endl;
std::cout << b->gamma.minTrackDTheta << std::endl;
std::cout << b->gamma.uid << std::endl;
std::cout << b->gamma.GetName() << std::endl;
}
TFile* f = new TFile("janbug.root");
TTree* t; f->GetObject("evtTree2",t);
JansEvent* j = new JansEvent();
t->SetBranchAddress("event", &j);
t->GetEvent(0);
std::cout << j->bList.GetEntries() << std::endl;
B_Parameters *b = dynamic_cast<B_Parameters*>(j->bList[0]);
//std::cout << (void*)j->bList[0] << " vs " << (void*)b << endl;
std::cout << b->gamma.minTrackDTheta << std::endl;
std::cout << b->gamma.uid << std::endl;
std::cout << b->gamma.GetName() << std::endl;
}
