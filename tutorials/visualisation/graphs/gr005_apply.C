/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// A macro to demonstrate the functionality of TGraph::Apply() method.
/// TGraph::Apply applies a function `f` to all the data TGraph points, `f` may be a 1-D function TF1 or 2-d function TF2.
/// The Y values of the graph are replaced by the new values computed using the function.
///
///
/// The Apply() method can be used as well for TGraphErrors and TGraphAsymmErrors.
///
/// \macro_image
/// \macro_code
///
/// \author Miro Helbich

void gr005_apply() {

   const Int_t npoints=3;
   Double_t xaxis[npoints] = {1.,2.,3.};
   Double_t yaxis[npoints] = {10.,20.,30.};

   TGraph *gr1 = new TGraph(npoints,xaxis,yaxis);
   TF2 *ff = new TF2("ff","-1./y"); //Defining the function `f`

   TCanvas *c1 = new TCanvas("c1","c1", 0,0,700,500);
   c1->Divide(2,1);

   c1->cd(1);
   gr1->DrawClone("A*"); //Using DrawClone to create a copy of the graph in the canvas.
   c1->cd(2);
   gr1->Apply(ff); //Applies the function `f` to all the data TGraph points
   gr1->Draw("A*"); 
   /* Without DrawClone, the modifications to gr1 via Apply(ff) are reflected in the original graph
   displayed in c1 (the two drawn graphs are not independent). */
}
