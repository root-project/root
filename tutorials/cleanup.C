{
  gROOT->Reset();
  TCanvas *c = 0;
  c = (TCanvas*)gROOT->FindObject("c1"); if (c) c->Delete(); c = 0;
  c = (TCanvas*)gROOT->FindObject("c2"); if (c) c->Delete(); c = 0;
  c = (TCanvas*)gROOT->FindObject("ntuple"); if (c) c->Delete(); c = 0;
  c = (TCanvas*)gROOT->FindObject("tornadoCanvas"); if (c) c->Delete(); c = 0;
}
