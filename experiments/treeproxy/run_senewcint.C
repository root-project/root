{
// bug in the optimization code.
// when deleting the temporaries (due to call to SetTree) it fails trying to delete a TString with the
// weird error:
//    Error: ~TString() header declared but not defined
gROOT->ProcessLine(".O 0");

// We need to open the file first because of our
// class Event; in senew.h
TFile *file = new TFile("Event.new.split9.root");


//gROOT->ProcessLine(".L TProxy.h+g");
//if (gROOT->GetClass("TProxy")==0) return;
//gROOT->ProcessLine(".L TProxyTemplate.h");

gROOT->ProcessLine(".L TProxy.h+g");
if (gROOT->GetClass("TProxy")==0) return;
gROOT->ProcessLine(".L TProxyTemplate.h");

bool hasEvent = gROOT->GetClass("Event")!=0;
if (hasEvent) {
   gROOT->ProcessLine("#define WITH_EVENT");
}
TTree *tree = (TTree*)file->Get("T");
//tree->Process("senewcint.C");   
return;

gROOT->ProcessLine(".L senewcint.C");
if (gROOT->GetClass("senewcint")==0) return;
senewcint secint(tree);
tree->Process(&secint);

}
/*

TFile *file = new TFile("Event.new.split9.root");


gROOT->ProcessLine(".L TProxy.h");

TTree *tree = (TTree*)file->Get("T");

tree->Process("senewcint.C");   

 */
/*
30
30
20.9114
1
3 
1.17697
t
type1
fMatrix[2][1]: 0.867107 //wrong with CINT go an address 
fH->GetMean() 0.615174
1
30 
-1.03195
-1.28227
fTracks.fNsp[0]: 1
fTracks.fPointValue[0][0]: 1
fLastTrack: 30
fTracks[2].fTriggerBits.fNbits 64==64
*/
