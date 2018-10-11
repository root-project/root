/// \file
/// \ingroup tutorial_hist
/// \notebook
/// Display the various 2-d drawing options
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void draw2dopt()
{
   gStyle->SetOptStat(0);
   gStyle->SetCanvasColor(33);
   gStyle->SetFrameFillColor(18);
   TF2 *f2 = new TF2("f2","xygaus + xygaus(5) + xylandau(10)",-4,4,-4,4);
   Double_t params[] = {130,-1.4,1.8,1.5,1, 150,2,0.5,-2,0.5, 3600,-2,0.7,-3,0.3};
   f2->SetParameters(params);
   auto h2 = new TH2F("h2","xygaus + xygaus(5) + xylandau(10)",20,-4,4,20,-4,4);
   h2->SetFillColor(46);
   h2->FillRandom("f2",40000);
   auto pl = new TPaveLabel();

   //basic 2-d options
   Float_t xMin=0.67, yMin=0.875, xMax=0.85, yMax=0.95;
   Int_t cancolor = 17;
   auto c2h = new TCanvas("c2h","2-d options",10,10,800,600);
   c2h->Divide(2,2);
   c2h->SetFillColor(cancolor);
   c2h->cd(1);
   h2->Draw();       pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"SCAT","brNDC");
   c2h->cd(2);
   h2->Draw("box");  pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"BOX","brNDC");
   c2h->cd(3);
   h2->Draw("arr");  pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"ARR","brNDC");
   c2h->cd(4);
   h2->Draw("colz"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"COLZ","brNDC");
   c2h->Update();

   //text option
   auto ctext = new TCanvas("ctext","text option",50,50,800,600);
   gPad->SetGrid();
   ctext->SetFillColor(cancolor);
   ctext->SetGrid();
   h2->Draw("text"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"TEXT","brNDC");
   ctext->Update();

   //contour options
   auto cont = new TCanvas("contours","contours",100,100,800,600);
   cont->Divide(2,2);
   gPad->SetGrid();
   cont->SetFillColor(cancolor);
   cont->cd(1);
   h2->Draw("contz"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"CONTZ","brNDC");
   cont->cd(2);
   gPad->SetGrid();
   h2->Draw("cont1"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"CONT1","brNDC");
   cont->cd(3);
   gPad->SetGrid();
   h2->Draw("cont2"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"CONT2","brNDC");
   cont->cd(4);
   gPad->SetGrid();
   h2->Draw("cont3"); pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"CONT3","brNDC");
   cont->Update();

   //lego options
   auto lego = new TCanvas("lego","lego options",150,150,800,600);
   lego->Divide(2,2);
   lego->SetFillColor(cancolor);
   lego->cd(1);
   h2->Draw("lego");     pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"LEGO","brNDC");
   lego->cd(2);
   h2->Draw("lego1");    pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"LEGO1","brNDC");
   lego->cd(3);
   gPad->SetTheta(61); gPad->SetPhi(-82);
   h2->Draw("surf1pol"); pl->DrawPaveLabel(xMin,yMin,xMax+0.05,yMax,"SURF1POL","brNDC");
   lego->cd(4);
   gPad->SetTheta(21); gPad->SetPhi(-90);
   h2->Draw("surf1cyl"); pl->DrawPaveLabel(xMin,yMin,xMax+0.05,yMax,"SURF1CYL","brNDC");
   lego->Update();

   //surface options
   auto surf = new TCanvas("surfopt","surface options",200,200,800,600);
   surf->Divide(2,2);
   surf->SetFillColor(cancolor);
   surf->cd(1);
   h2->Draw("surf1");   pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"SURF1","brNDC");
   surf->cd(2);
   h2->Draw("surf2z");  pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"SURF2Z","brNDC");
   surf->cd(3);
   h2->Draw("surf3");   pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"SURF3","brNDC");
   surf->cd(4);
   h2->Draw("surf4");   pl->DrawPaveLabel(xMin,yMin,xMax,yMax,"SURF4","brNDC");
   surf->Update();
}
