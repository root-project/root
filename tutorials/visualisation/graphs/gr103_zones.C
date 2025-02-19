/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview Example of script showing how to divide a canvas
/// into adjacent subpads + axis labels on the top and right side
/// of the pads.
///
/// See the [Divide documentation](https://root.cern/doc/master/classTPad.html#a2714ddd7ba72d5def84edc1fbaea8658)
///
/// Note that the last 2 arguments in
///     `c1->Divide(2,2,0,0)`
/// define 0 space between the pads. With this, the axis labels where the pads
/// touch may be cut, as in this tutorial. To avoid this, either add some spacing
/// between pads (instead of 0) or change the limits of the plot in the pad (histos
/// in this tutorial). E.g. `h3` could be defined as
///     `TH2F *h3 = new TH2F("h3","test3",10,0,1,22,-1.1,1.1);`
/// but note that this can change the displayed axis labels (requiring SetNdivisions
/// to readjust).
///
/// SetLabelOffset changes the (perpendicular) distance to the axis. The label
/// position along the axis cannot be changed
///
/// \macro_image
/// \macro_code
/// \author Rene Brun

void gr103_zones() {
   TCanvas *c1 = new TCanvas("c1","multipads",900,700);
   gStyle->SetOptStat(0);
   c1->Divide(2,2,0,0);
   TH2F *h1 = new TH2F("h1","test1",10,0,1,20,0,20);
   TH2F *h2 = new TH2F("h2","test2",10,0,1,20,0,100);
   TH2F *h3 = new TH2F("h3","test3",10,0,1,20,-1,1);
   TH2F *h4 = new TH2F("h4","test4",10,0,1,20,0,1000);

   c1->cd(1);
   gPad->SetTickx(2);
   h1->Draw();

   c1->cd(2);
   gPad->SetTickx(2);
   gPad->SetTicky(2);
   h2->GetYaxis()->SetLabelOffset(0.01);
   h2->Draw();

   c1->cd(3);
   h3->Draw();

   c1->cd(4);
   gPad->SetTicky(2);
   h4->Draw();
}
