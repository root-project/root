int execScanFail() {
  TFile *f = new TFile("run00155-short.root");
  //gROOT->ProcessLine(".x read-gates-98MeV-forMelamine.C");
  //gROOT->ProcessLine(".x chain-Melamine-98MeV.C");

  gROOT->ProcessLine(".x CUTpad1pad2.C");
  gROOT->ProcessLine(".x CUTpad1tof.C");

  TTree *t; f->GetObject("DATA",t);
  if (!t) return 1;
  Long64_t res = t->Scan("ThSCAT:X1pos:CUTpad1pad2:CUTpad1tof","CUTpad1pad2 * CUTpad1tof * ThSCAT>0","col=6.3:8.6");

  //DATA->Draw("ThSCAT:X1pos>>hThSCATvsX1","CUTpad1pad2 && CUTpad1tof && (ThSCAT>0)","col");
  //DATA->Draw("ThSCAT:X1pos>>hThSCATvsX1","CUTpad1pad2 && CUTpad1tof && ThSCAT>0","col");
  //TCanvas *second = new TCanvas("second","second",400,400);
  //DATA->Draw("ThSCAT:X1pos>>hThSCATvsX1","ThSCAT>0 && CUTpad1pad2 && CUTpad1tof","col");
  //DATA->Draw("ThSCAT:X1pos>>hThSCATvsX1","ThSCAT>0 && CUTpad1pad2","col");
  
  if (res != 4) {
     Error("execScanFail","The number of selected entries is wrong %lld\n",res);
     return 1;
  }
  return 0;
}
