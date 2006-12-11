{
//A macro to demonstrate the functionality of TGraphX::Apply() method
//Author: Miro Helbich oct.2001


Int_t npoints=3;
Double_t xaxis[npoints] = {1.,2.,3.};
Double_t yaxis[npoints] = {10.,20.,30.};
Double_t errorx[npoints] = {0.5,0.5,0.5};
Double_t errory[npoints] = {5.,5.,5.};

Double_t exl[npoints] = {0.5,0.5,0.5};
Double_t exh[npoints] = {0.5,0.5,0.5};
Double_t eyl[npoints] = {5.,5.,5.};
Double_t eyh[npoints] = {5.,5.,5.};

TGraph *gr1 = new TGraph(npoints,xaxis,yaxis);
TGraphErrors *gr2 = new TGraphErrors(npoints,xaxis,yaxis,errorx,errory);
TGraphAsymmErrors *gr3 = new TGraphAsymmErrors(npoints,xaxis,yaxis,exl,exh,eyl,eyh);
TF2 *ff = new TF2("ff","-1./y");

TCanvas *c1 = new TCanvas("c1","c1");
c1->Divide(2,3);
// TGraph
c1->cd(1);
gr1->DrawClone("A*");
c1->cd(2);
gr1->Apply(ff);
gr1->Draw("A*");

// TGraphErrors
c1->cd(3);
gr2->DrawClone("A*");
c1->cd(4);
gr2->Apply(ff);
gr2->Draw("A*");

// TGraphAsymmErrors
c1->cd(5);
gr3->DrawClone("A*");
c1->cd(6);
gr3->Apply(ff);
gr3->Draw("A*");

}
