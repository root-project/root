// this test program compares the I/O performance obtained with
// STL vector of objects or pointers to objects versus the native
// Root collection class TClonesArray.
// Trees in compression and non compression mode are created for each
// of the following cases:
//  -vector<THit>
//  -vector<THit*>
//  -TClonesArray(TObjHit) in no split mode
//  -TClonesArray(TObjHit) in split mode
// where:
//  THit is a class not derived from TObject
//  TObjHit derives from TObject and THit
//
// The test prints a summary table comparing performances for all above cases
// (CPU, file size, compression factors).
// Reference numbers on a Pentium III 650 Mhz machine are given as reference.
//
// A canvas is created showing the same results in a graphical form.
// The bench can be run in batch mode (bench -b).
// A Postscript file bench.ps is also produced.
//      Author:  Rene Brun

#include "TROOT.h"
#include "TClonesArray.h"
#include "TStopwatch.h"
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TCanvas.h"
#include "TPaveText.h"
#include "TArrow.h"
#include "TH1.h"
#include "TBox.h"
#include "TStyle.h"
#include "TText.h"
#include "TRint.h"

#include "TBench.h"

void showhist(const char *title, const char *ytitle, float a, float b, float c, float d, float ar, float br, float cr, float dr)
{
   //function to display one result unit
   TH1F *h = new TH1F("h",title,4,0,4);
   h->SetDirectory(0);
   h->GetXaxis()->SetLabelFont(22);
   h->GetXaxis()->SetLabelOffset(99);
   h->GetXaxis()->SetNdivisions(4);
   h->GetYaxis()->SetNdivisions(5);
   h->GetYaxis()->SetLabelSize(.1);
   h->SetStats(0);
   gPad->SetGrid();
   gPad->SetBottomMargin(0.05);
   gPad->SetTopMargin(0.16);
   gPad->SetLeftMargin(0.14);
   gPad->SetRightMargin(0.05);
   if (strlen(ytitle)) {
      h->GetYaxis()->SetTitle(ytitle);
      h->GetYaxis()->SetTitleOffset(0.7);
      h->GetYaxis()->SetTitleSize(0.09);
   } else {
      h->GetYaxis()->SetLabelOffset(99);
   }

   float ymax = a;
   if (b  > ymax) ymax = b;
   if (c  > ymax) ymax = c;
   if (d  > ymax) ymax = d;
   if (ar > ymax) ymax = ar;
   if (br > ymax) ymax = br;
   if (cr > ymax) ymax = cr;
   if (dr > ymax) ymax = dr;
   if (b > ymax) ymax = b;
   ymax *=1.05;
   h->SetMinimum(0);
   h->SetMaximum(ymax);
   h->Draw();
   float dx  = 0.4;
   float dxr = 0.1;
   TBox *boxa  = new TBox(0.5-dx ,0,0.5+dx ,a); boxa->SetFillColor(46);  boxa->Draw();
   TBox *boxb  = new TBox(1.5-dx ,0,1.5+dx ,b); boxb->SetFillColor(5);  boxb->Draw();
   TBox *boxc  = new TBox(2.5-dx ,0,2.5+dx ,c); boxc->SetFillColor(6);  boxc->Draw();
   TBox *boxd  = new TBox(3.5-dx ,0,3.5+dx ,d); boxd->SetFillColor(2);  boxd->Draw();
   TBox *boxar = new TBox(0.5-dxr,0,0.5+dxr,ar); boxar->SetFillColor(1); boxar->Draw();
   TBox *boxbr = new TBox(1.5-dxr,0,1.5+dxr,br); boxbr->SetFillColor(1); boxbr->Draw();
   TBox *boxcr = new TBox(2.5-dxr,0,2.5+dxr,cr); boxcr->SetFillColor(1); boxcr->Draw();
   TBox *boxdr = new TBox(3.5-dxr,0,3.5+dxr,dr); boxdr->SetFillColor(1); boxdr->Draw();
   boxa->SetFillStyle(3013);
   boxb->SetFillStyle(3010);
   boxc->SetFillStyle(3014);
   boxd->SetFillStyle(3012);
}

int main(int argc, char** argv)
{
  TRint *theApp = new TRint("Rint", &argc, argv, 0, 0);

  int nhits       = 1000;
  int nevents     = 400;
  Float_t cx;

  TTree::SetBranchStyle(1); // use the new Bronch style

  //testing STL vector of THit
  Double_t cptot = 0;
  TStopwatch timer;
  timer.Start();
  TSTLhit *STLhit = new TSTLhit(nhits);
  STLhit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt1 = timer.RealTime();
  Double_t cp1 = timer.CpuTime();
  cptot += cp1;
  printf("1  STLhit :  RT=%6.2f s  Cpu=%6.2f s\n",rt1,cp1);
  timer.Start(kTRUE);
  Int_t nbytes1 = STLhit->MakeTree(1,nevents,0,0,cx);
  timer.Stop();
  Double_t rt2w = timer.RealTime();
  Double_t cp2w = timer.CpuTime();
  cptot += cp2w;
  printf("2  STLhitw:  RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt2w-rt1,cp2w-cp1,nbytes1,cx);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt2r = timer.RealTime();
  Double_t cp2r = timer.CpuTime();
  cptot += cp2r;
  printf("3  STLhitr:  RT=%6.2f s  Cpu=%6.2f s\n",rt2r,cp2r);
  timer.Start(kTRUE);
  Float_t cx3;
  Int_t nbytes3 = STLhit->MakeTree(1,nevents,1,0,cx3);
  timer.Stop();
  Double_t rt3w = timer.RealTime();
  Double_t cp3w = timer.CpuTime();
  cptot += cp3w;
  printf("4  STLhitw:  RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt3w-rt1,cp3w-cp1,nbytes3,cx3);
  timer.Start(kTRUE);
  STLhit->ReadTree();
  timer.Stop();
  Double_t rt3r = timer.RealTime();
  Double_t cp3r = timer.CpuTime();
  cptot += cp3r;
  printf("5  STLhitr:  RT=%6.2f s  Cpu=%6.2f s\n",rt3r,cp3r);

  //testing STL vector of pointers to THit
  timer.Start();
  TSTLhitStar *STLhitStar = new TSTLhitStar(nhits);
  STLhitStar->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt4 = timer.RealTime();
  Double_t cp4 = timer.CpuTime();
  cptot += cp4;
  printf("6  STLhit* : RT=%6.2f s  Cpu=%6.2f s\n",rt4,cp4);
  timer.Start(kTRUE);
  Int_t nbytes5 = STLhitStar->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt5w = timer.RealTime();
  Double_t cp5w = timer.CpuTime();
  cptot += cp5w;
  printf("7  STLhit*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt5w-rt4,cp5w-cp4,nbytes5,cx);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt5r = timer.RealTime();
  Double_t cp5r = timer.CpuTime();
  cptot += cp5r;
  printf("8  STLhit*r: RT=%6.2f s  Cpu=%6.2f s\n",rt5r,cp5r);
  timer.Start(kTRUE);
  Float_t cx6;
  Int_t nbytes6 = STLhitStar->MakeTree(1,nevents,1,99,cx6);
  timer.Stop();
  Double_t rt6w = timer.RealTime();
  Double_t cp6w = timer.CpuTime();
  cptot += cp6w;
  printf("9  STLhit*w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt6w-rt4,cp6w-cp4,nbytes6,cx6);
  timer.Start(kTRUE);
  STLhitStar->ReadTree();
  timer.Stop();
  Double_t rt6r = timer.RealTime();
  Double_t cp6r = timer.CpuTime();
  cptot += cp6r;
  printf("10 STLhit*r: RT=%6.2f s  Cpu=%6.2f s\n",rt6r,cp6r);

  //testing TClonesArray of TObjHit deriving from THit
  timer.Start();
  TCloneshit *Cloneshit = new TCloneshit(nhits);
  Cloneshit->MakeTree(0,nevents,0,0,cx);
  timer.Stop();
  Double_t rt7 = timer.RealTime();
  Double_t cp7 = timer.CpuTime();
  cptot += cp7;
  printf("11 Clones1 : RT=%6.2f s  Cpu=%6.2f s\n",rt7,cp7);
  timer.Start(kTRUE);
  Int_t nbytes8 = Cloneshit->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt8w = timer.RealTime();
  Double_t cp8w = timer.CpuTime();
  cptot += cp8w;
  printf("12 Clones1w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt8w-rt7,cp8w-cp7,nbytes8,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt8r = timer.RealTime();
  Double_t cp8r = timer.CpuTime();
  cptot += cp8r;
  printf("13 Clones1r: RT=%6.2f s  Cpu=%6.2f s\n",rt8r,cp8r);
  timer.Start(kTRUE);
  Float_t cx9;
  Int_t nbytes9 = Cloneshit->MakeTree(1,nevents,1,99,cx9);
  timer.Stop();
  Double_t rt9w = timer.RealTime();
  Double_t cp9w = timer.CpuTime();
  cptot += cp9w;
  printf("14 Clones1w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt9w-rt7,cp9w-cp7,nbytes9,cx9);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt9r = timer.RealTime();
  Double_t cp9r = timer.CpuTime();
  cptot += cp9r;
  printf("15 Clones1r: RT=%6.2f s  Cpu=%6.2f s\n",rt9r,cp9r);
  timer.Start(kTRUE);
  Int_t nbytes10 = Cloneshit->MakeTree(1,nevents,0,99,cx);
  timer.Stop();
  Double_t rt10w = timer.RealTime();
  Double_t cp10w = timer.CpuTime();
  cptot += cp10w;
  printf("16 Clones2w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt10w-rt7,cp10w-cp7,nbytes10,cx);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt10r = timer.RealTime();
  Double_t cp10r = timer.CpuTime();
  cptot += cp10r;
  printf("17 Clones2r: RT=%6.2f s  Cpu=%6.2f s\n",rt10r,cp10r);
  timer.Start(kTRUE);
  Float_t cx11;
  Int_t nbytes11 = Cloneshit->MakeTree(1,nevents,1,99,cx11);
  timer.Stop();
  Double_t rt11w = timer.RealTime();
  Double_t cp11w = timer.CpuTime();
  cptot += cp11w;
  printf("18 Clones2w: RT=%6.2f s  Cpu=%6.2f s, size= %8d bytes, cx=%5.2f\n",rt11w-rt7,cp11w-cp7,nbytes11,cx11);
  timer.Start(kTRUE);
  Cloneshit->ReadTree();
  timer.Stop();
  Double_t rt11r = timer.RealTime();
  Double_t cp11r = timer.CpuTime();
  cptot += cp11r;
  printf("19 Clones2r: RT=%6.2f s  Cpu=%6.2f s\n",rt11r,cp11r);
  Double_t cpref = 76.33;
  Double_t rootmarks = cpref*200/cptot;

  //delete temp file used for the benchmark
  gSystem->Exec("rm -f /tmp/bench.root");

  //print all results
  char line1[100], line2[100];
  printf("\n");
  printf("******************************************************************************\n");
  sprintf(line1,"Comparing STL vector with TClonesArray: Root %-8s",gROOT->GetVersion());
  printf("*       %s                *\n",line1);
  Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
  if (UNIX) {
     FILE *fp = gSystem->OpenPipe("uname -a", "r");
     char line[60];
     fgets(line,60,fp); line[59] = 0;
     sprintf(line2,"%s",line);
     printf("*  %s\n",line);
     gSystem->ClosePipe(fp);
  } else {
     const char *os = gSystem->Getenv("OS");
     sprintf(line2,"Windows");
     if (!os) printf("*  Windows 95\n");
     else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
  }
  printf("*     Reference machine pcnotebrun.cern.ch  RedHat Linux 6.1                 *\n");
  printf("*         (Pentium III 650 Mhz 256 Mbytes RAM, IDE disk)                     *\n");
  printf("*           (send your results to rootdev@root.cern.ch)                      *\n");
  printf("******************************************************************************\n");
  printf("* Time to fill the structures (seconds)   Reference      cx      Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        1.91     %5.2f        4.57     *\n",cp1,cx3);
  printf("* vector<THit*>                 %6.2f        1.86     %5.2f        4.57     *\n",cp4,cx6);
  printf("* TClonesArray(TObjHit)         %6.2f        1.62     %5.2f        6.76     *\n",cp7,cx9);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.62     %5.2f        6.75     *\n",cp7,cx11);
  printf("******************************************************************************\n");
  printf("* Size of file in bytes         comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %8d   42053031   %8d    9213459    *\n",nbytes1,nbytes3);
  printf("* vector<THit*>                 %8d   42079941   %8d    9215935    *\n",nbytes5,nbytes6);
  printf("* TClonesArray(TObjHit)         %8d   39807325   %8d    5892837    *\n",nbytes8,nbytes9);
  printf("* TClonesArray(TObjHit) split   %8d   39807325   %8d    5901163    *\n",nbytes10,nbytes11);
  printf("******************************************************************************\n");
  printf("* Time to write in seconds      comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        1.74    %6.2f        9.58     *\n",cp2w-cp1, cp3w-cp1);
  printf("* vector<THit*>                 %6.2f        1.80    %6.2f        9.62     *\n",cp5w-cp1, cp6w-cp1);
  printf("* TClonesArray(TObjHit)         %6.2f        1.60    %6.2f        7.32     *\n",cp8w-cp1, cp9w-cp1);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.51    %6.2f        6.18     *\n",cp10w-cp1,cp11w-cp1);
  printf("******************************************************************************\n");
  printf("* Time to read in seconds       comp 0    Reference    comp 1    Reference   *\n");
  printf("******************************************************************************\n");
  printf("* vector<THit>                  %6.2f        2.29    %6.2f        3.67     *\n",cp2r,cp3r);
  printf("* vector<THit*>                 %6.2f        2.10    %6.2f        3.27     *\n",cp5r,cp6r);
  printf("* TClonesArray(TObjHit)         %6.2f        1.53    %6.2f        2.14     *\n",cp8r,cp9r);
  printf("* TClonesArray(TObjHit) split   %6.2f        1.35    %6.2f        1.94     *\n",cp10r,cp11r);
  printf("******************************************************************************\n");
  printf("* Total CPU time              %8.2f    %8.2f                           *\n",cptot,cpref);
  printf("* Estimated ROOTMARKS         %8.2f      200.00                           *\n",rootmarks);
  printf("******************************************************************************\n");

  // show results with graphics
   gStyle->SetTitleH(0.12);
   gStyle->SetTitleW(0.8);
   gStyle->SetFrameFillColor(33);
   TCanvas *cbench = new TCanvas("cbench","Results of Root benchmark",10,10,600,800);
   TPaveText *head = new TPaveText(.05,.81,.95,.99);
   head->SetFillColor(42);
   head->SetTextAlign(22);
   head->AddText(line1);
   head->AddText(line2);
   head->AddText("Reference machine pcnotebrun.cern.ch  RedHat Linux 6.1");
   head->AddText("(Pentium III 650 Mhz 256 Mbytes RAM, IDE disk)");
   head->AddText("(send your results to rootdev@root.cern.ch)");
   head->Draw();
   TPad *pmain = new TPad("pmain","pmain",0,0,1,.8);
   pmain->SetFillColor(20);
   pmain->Draw();
   pmain->Divide(2,4);
   pmain->cd(1);
   showhist("legend","",5,4,3,2,9,6,4,3);
   TText t;
   t.SetTextFont(23);
   t.SetTextSize(15);
   t.DrawText(1.0,8.0,"vector<THit>");
   t.DrawText(1.5,6.8,"vector<THit*>");
   t.DrawText(2.5,5.6,"TClonesArray");
   t.DrawText(2.8,4.5,"-nosplit");
   t.DrawText(3.1,3.4,"-split");
   TArrow *arrow = new TArrow(-0.0519789,8.04398,0.327782,6.94924,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   TPaveText *pt = new TPaveText(0.02,0.65,0.16,0.80,"brNDC");
   pt->AddText("Ref.");
   pt->AddText("machine");
   pt->Draw();
   arrow = new TArrow(1.19087,7.87556,0.828376,5.43344,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(2.05397,6.61239,1.82956,4.33869,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(2.8998,4.50712,2.72718,3.41237,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   arrow = new TArrow(3.24504,3.24395,3.2623,2.40184,0.03,"|>");
   arrow->SetFillColor(1);
   arrow->SetFillStyle(1001);
   arrow->Draw();
   Float_t z = 1.e-6;
   pmain->cd(2);
   showhist("Time to fill collections","seconds",cp1,cp4,cp7,cp7,1.88,1.81,1.60,1.60);
   pmain->cd(3);
   showhist("File size no compression","Megabytes",z*nbytes1,z*nbytes5,z*nbytes8,z*nbytes10, 42.05,42.08,39.69,39.78);
   pmain->cd(4);
   showhist("File size compression 1","Megabytes",z*nbytes3,z*nbytes6,z*nbytes9,z*nbytes11,9.21,9.21,6.09,5.93);
   pmain->cd(5);
   showhist("Time to write no compression","seconds",cp2w-cp1,cp5w-cp1,cp8w-cp1,cp10w-cp1,1.76,1.77,1.30,1.30);
   pmain->cd(6);
   showhist("Time to write compression 1","seconds",cp3w-cp1,cp6w-cp1,cp9w-cp1,cp11w-cp1,9.68,9.70,7.62,6.01);
   pmain->cd(7);
   showhist("Time to read no compression","seconds",cp2r,cp5r,cp8r,cp10r,2.21,2.04,1.58,1.36);
   pmain->cd(8);
   showhist("Time to read compression 1","seconds",cp3r,cp6r,cp9r,cp11r,3.46,3.17,2.12,1.95);
   cbench->Print();

   theApp->Run();

}
