// Author: Heiko.Scheit@mpi-hd.mpg.de
//
//  simple Event class example
//
// execute as: .x tree0.C++
//
//  You have to copy it first to a directory where you have write access!
//  Note that .x tree0.C cannot work with this example
//
//
///////////////////////////////
//  Effect of ClassDef() and ClassImp() macros
//===============================================
//
// After running this macro create an instance of Det and Event
//
//   Det d;
//   Event e;
//
// now you can see the effect of the  ClassDef() and ClassImp() macros.
// (for the Det class these commands are commented!)
// For instance 'e' now knows who it is:
//
//   cout<<e.Class_Name()<<endl;
//
// whereas d does not.
//
// The methods that are added by the ClassDef()/Imp() marcro can be listed with
// .class
//   .class Event
//   .class Det
///////////////////

#include <TRandom.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TStyle.h>

#include <Riostream.h>

//class Det  : public TObject  {
class Det {  // each detector gives an energy and time signal
public:
  Double_t e; //energy
  Double_t t; //time

//  ClassDef(Det,1)
};

//ClassImp(Det)

//class Event { //TObject is not required by this example
class Event : public TObject {
public:

  Det a; // say there are two detectors (a and b) in the experiment
  Det b;
  ClassDef(Event,1)
};

ClassImp(Event)

void tree0() {
  // create a TTree
  TTree *tree = new TTree("tree","treelibrated tree");
  Event *e = new Event;

  // create a branch with energy
  tree->Branch("event",&e);

  // fill some events with random numbers
  Int_t nevent=10000;
  for (Int_t iev=0;iev<nevent;iev++) {
    if (iev%1000==0) cout<<"Processing event "<<iev<<"..."<<endl;

    Float_t ea,eb;
    gRandom->Rannor(ea,eb); // the two energies follow a gaus distribution
    e->a.e=ea;
    e->b.e=eb;
    e->a.t=gRandom->Rndm();  // random
    e->b.t=e->a.t + gRandom->Gaus(0.,.1);  // identical to a.t but a gaussian
                                           // 'resolution' was added with sigma .1

    tree->Fill();  // fill the tree with the current event
  }

  // start the viewer
  // here you can investigate the structure of your Event class
  tree->StartViewer();

  //gROOT->SetStyle("Plain");   // uncomment to set a different style
  gStyle->SetPalette(1);        // use precomputed color palette 1

  // now draw some tree variables
  TCanvas *c1 = new TCanvas();
  c1->Divide(2,2);
  c1->cd(1);
  tree->Draw("a.e");  //energy of det a
  tree->Draw("a.e","3*(-.2<b.e && b.e<.2)","same");  // same but with condition on energy b; scaled by 3
  c1->cd(2);
  tree->Draw("b.e:a.e","","colz");        // one energy against the other
  c1->cd(3);
  tree->Draw("b.t","","e");    // time of b with errorbars
  tree->Draw("a.t","","same"); // overlay time of detector a
  c1->cd(4);
  tree->Draw("b.t:a.t");       // plot time b again time a

  cout<<endl;
  cout<<"You can now examine the structure of your tree in the TreeViewer"<<endl;
  cout<<endl;
}

