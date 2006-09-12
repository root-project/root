// @(#)root/test:$name:  $:$id: stressGraphics.cxx,v 1.0 exp $
// Author: O.Couet

//
//    ROOT  Graphics test suite and benchmarks.
//
// The suite of programs below tests many elements of the graphics classes

#include <Riostream.h>
#include <time.h>
#include <TString.h>
#include <TROOT.h>
#include <TError.h>
#include <TRandom.h>
#include <TBenchmark.h>
#include <TSystem.h>
#include <TDatime.h>
#include <TFile.h>
#include <TF1.h>
#include <TF3.h>
#include <TH1.h>
#include <TNtuple.h>
#include <TProfile.h>

#include <TStyle.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TFrame.h>
#include <TPostScript.h>
#include <TLine.h>
#include <TMarker.h>
#include <TPolyLine.h>
#include <TLatex.h>
#include <TEllipse.h>
#include <TCurlyArc.h>
#include <TArc.h>
#include <TPaveText.h>
#include <TPaveStats.h>
#include <TPaveLabel.h>
#include <TGaxis.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TMultiGraph.h>
#include <TGraph2D.h>
#include <TImage.h>


void     stressGraphics (Int_t verbose=0);
void     StatusPrint    (Int_t id, const TString &title, Int_t res, Int_t ref, Int_t err);
Int_t    AnalysePS      (const TString &filename);
void     StartTest      (Int_t w, Int_t h);
void     EndTest        (const TString &title, Int_t nPS);

// Tests functions.
void     tline          ();
void     tmarker        ();
void     tpolyline      ();
void     patterns       ();
void     ttext1         ();
void     ttext2         ();
void     tlatex1        ();
void     tlatex2        ();
void     tlatex3        ();
void     tlatex4        ();
void     tlatex5        ();
void     tgaxis1        ();
void     tgaxis2        ();
void     tgaxis3        ();
void     tgaxis4        ();
void     tellipse       ();
void     feynman        ();
void     tgraph1        ();
void     tgraph2        ();
void     tmultigraph    ();
void     options2d1     ();
void     options2d2     ();
void     options2d3     ();
void     options2d4     ();
void     options2d5     ();
void     earth          ();
void     tgraph2d1      ();
void     tgraph2d2      ();
void     tgraph2d3      ();
void     ntuple1        ();
void     quarks         ();
void     timage         ();

// Auxiliary functions
void     patterns_box   (Int_t pat, Double_t x1, Double_t y1, Double_t x2, Double_t  y2);
void     tmarker_draw   (Double_t x, Double_t y, Int_t mt, Double_t d);
void     cleanup        ();

// Global variables.
Int_t    gVerbose;
Int_t    gTestNum;
TCanvas *gC;
Int_t    gRefNb[32];
Int_t    gErrNb[32];
Bool_t   gOptionR;
TH2F    *gH2;
TFile   *gFile;


//______________________________________________________________________________
int main(int argc,const char *argv[])
{
   gBenchmark = new TBenchmark();

   TString opt;
   Int_t verbose = 0;
   if (argc > 1) verbose = atoi(argv[1]);
   opt = argv[1];

   if (opt.Contains("-h")) {
      printf("Usage: stressGraphics [-h] [-r]\n");
      printf("Options:\n");
      printf("  -r : Generate de reference output.\n");
      printf("       Redirect the output in the file \"stressGraphics.ref\"\n");
      printf("       to redefine the reference file.\n");
      printf("\n");
      printf("  -h : Print usage\n");
      return 0;
   }

   if (opt.Contains("-r")) {
      gOptionR = kTRUE;
   } else {
      gOptionR = kFALSE;
   }

   stressGraphics(verbose);

   cleanup();
   return 0;
}


//______________________________________________________________________________
void stressGraphics(Int_t verbose)
{
   // Run all graphics stress tests.

   // Check if $ROOTSYS/tutorials/hsimple.root exists
   gErrorIgnoreLevel = 9999;
   gFile = new TFile("$ROOTSYS/tutorials/hsimple.root");
   if (gFile->IsZombie()) {
      printf("File $ROOTSYS/tutorials/hsimple.root does not exist. Run tutorial hsimple.C first\n");
      return;
   }
   gErrorIgnoreLevel = 0;
   
   // Read the reference file "stressGraphics.ref"
   FILE *sg = fopen("stressGraphics.ref","r");
   char line[80];
   Int_t i = -1;
   while (fgets(line,80,sg)) {
      if (i>=0) {
         sscanf(&line[4], "%d",&gRefNb[i]);
         sscanf(&line[12],"%d",&gErrNb[i]);
      }
      i++;
   }
   fclose(sg);

   if (gOptionR) {
      cout << "Test#   Ref#   Err#   Time" <<endl;
   } else {
      cout << "******************************************************************" <<endl;
      cout << "*  Starting  Graphics - S T R E S S suite                        *" <<endl;
      cout << "******************************************************************" <<endl;
   }

   gROOT->SetBatch();

   gVerbose = verbose;
   gTestNum = 0;
   gC       = 0;

   gBenchmark->Start("stress");

   if (!gOptionR) {
      cout << "*  Starting Basic Graphics - S T R E S S                         *" <<endl;
      cout << "******************************************************************" <<endl;
   }
   tline       ();
   tmarker     ();
   tpolyline   ();
   patterns    ();
   ttext1      ();
   ttext2      ();
   tlatex1     ();
   tlatex2     ();
   tlatex3     ();
   tlatex4     ();
   tlatex5     ();
   if (!gOptionR) {
      cout << "******************************************************************" <<endl;
      cout << "*  Starting High Level 2D Primitives - S T R E S S               *" <<endl;
      cout << "******************************************************************" <<endl;
   }
   tgaxis1     ();
   tgaxis2     ();
   tgaxis3     ();
   tgaxis4     ();
   tellipse    ();
   feynman     ();
   tgraph1     ();
   tgraph2     ();
   tmultigraph ();
   if (!gOptionR) {
      cout << "******************************************************************" <<endl;
      cout << "*  Starting High Level 3D Primitives - S T R E S S               *" <<endl;
      cout << "******************************************************************" <<endl;
   }
   options2d1  ();
   options2d2  ();
   options2d3  ();
   options2d4  ();
   options2d5  ();
   earth       ();
   tgraph2d1   ();
   tgraph2d2   ();
   tgraph2d3   ();
   if (!gOptionR) {
      cout << "******************************************************************" <<endl;
      cout << "*  Starting complex drawing and TPad - S T R E S S               *" <<endl;
      cout << "******************************************************************" <<endl;
   }
   ntuple1     ();
   quarks      ();
   timage      ();
   if (!gOptionR) {
      cout << "******************************************************************" <<endl;

      gBenchmark->Stop("stress");

      //Print table with results
      Bool_t UNIX = strcmp(gSystem->GetName(), "Unix") == 0;
      if (UNIX) {
         FILE *fp = gSystem->OpenPipe("uname -a", "r");
         Char_t line[60];
         fgets(line,60,fp); line[59] = 0;
         printf("*  %s\n",line);
         gSystem->ClosePipe(fp);
      } else {
         const Char_t *os = gSystem->Getenv("OS");
         if (!os) printf("*  Windows 95\n");
         else     printf("*  %s %s \n",os,gSystem->Getenv("PROCESSOR_IDENTIFIER"));
      }

      printf("******************************************************************\n");
      printf("* ");
      gBenchmark->Print("stress");

      Double_t ct = gBenchmark->GetCpuTime("stress");
      const Double_t rootmarks = 600*(4.85/ct);

      printf("******************************************************************\n");
      printf("*  ROOTMARKS =%6.1f   *  Root%-8s  %d/%d\n",rootmarks,gROOT->GetVersion(),
             gROOT->GetVersionDate(),gROOT->GetVersionTime());
      printf("******************************************************************\n");
   }
}


//______________________________________________________________________________
void StatusPrint(Int_t id, const TString &title, Int_t res, Int_t ref, Int_t err)

{
   // Print test program number and its title

   Char_t number[4];
   sprintf(number,"%2d",id);
   if (!gOptionR) {
      TString header = TString("Test ")+number+" : "+title;
      const Int_t nch = header.Length();
      if (TMath::Abs(res-ref)<=err) {
         for (Int_t i = nch; i < 63; i++) header += '.';
         cout << header <<  " OK" << endl;
         gSystem->Unlink(Form("sg%2.2d.ps",id));
      } else {
         for (Int_t i = nch; i < 59; i++) header += '.';
         cout << header << " FAILED" << endl;
         cout << "          Result    = "  << res << endl;
         cout << "          Reference = "  << ref << endl;
         cout << "          Error     = "  << TMath::Abs(res-ref) 
                                           << " (was " << err << ")"<< endl;
      }
   } else {
      printf("%3d %8d %5d\n",id,res,err);
   }
}


//______________________________________________________________________________
Int_t AnalysePS(const TString &filename)
{
   // Analyse the PS file "filename" and return the number of character in the
   // meaningful part of the file. The variable part (date etc..) are not
   // counted.
   
   Bool_t counting = kFALSE;
   Int_t count = 0;

   char *line = new char[251];
   TString l;
   FILE *fp;
   Int_t status;
   if ((fp=fopen(filename.Data(), "r"))==NULL) {
      printf("ERROR1 : File can not open !..\n");
      return 0;
   }
   while((status=fscanf(fp, "%s", line)) != EOF) {
      l = line;
      if (l.Contains("%!PS-Adobe"))  counting = kFALSE;
      if (l.Contains("%%EndProlog")) counting = kTRUE;
      if (counting) count = count+l.Length();
   }
   if (gVerbose==1) printf(">>>>>>>>> Number of characters found in %s: %d\n",filename.Data(),count);
   return count;
}


//______________________________________________________________________________
void StartTest(Int_t w, Int_t h)
{
   // Start Test:
   // Open the TCanvas gC and set the acceptable error (number of characters)

   gTestNum++;
   gC = new TCanvas("gC","gC",0,0,w,h);
}

//______________________________________________________________________________
void EndTest(const TString &title)
{
   // End Test:
   // Draw the canvas generate as PostScript, count the number of characters in
   // the PS file and compare the result with the reference value.

   TString fileName = Form("sg%2.2d.ps",gTestNum);
   
   TPostScript *ps = new TPostScript(fileName,111);
   gC->Draw();
   ps->Close();

   StatusPrint(gTestNum, title, AnalysePS(fileName),
                                gRefNb[gTestNum-1],
                                gErrNb[gTestNum-1]);

   if (gC) {delete gC; gC=0;}
   return;
}


//______________________________________________________________________________
void tline()
{
   // Test TLine.

   StartTest(800,800);

   TLine *l1 = new TLine(0.1,0.1,0.9,0.1);
   l1->SetLineColor(1); l1->SetLineWidth(1) ; l1->SetLineStyle(1) ; l1->Draw();
   TLine *l2 = new TLine(0.1,0.2,0.9,0.2);
   l2->SetLineColor(2); l2->SetLineWidth(2) ; l2->SetLineStyle(2) ; l2->Draw();
   TLine *l3 = new TLine(0.1,0.3,0.9,0.3);
   l3->SetLineColor(3); l3->SetLineWidth(3) ; l3->SetLineStyle(3) ; l3->Draw();
   TLine *l4 = new TLine(0.1,0.4,0.9,0.4);
   l4->SetLineColor(4); l4->SetLineWidth(4) ; l4->SetLineStyle(4) ; l4->Draw();
   TLine *l5 = new TLine(0.1,0.5,0.9,0.5);
   l5->SetLineColor(5); l5->SetLineWidth(5) ; l5->SetLineStyle(5) ; l5->Draw();
   TLine *l6 = new TLine(0.1,0.6,0.9,0.6);
   l6->SetLineColor(6); l6->SetLineWidth(6) ; l6->SetLineStyle(6) ; l6->Draw();
   TLine *l7 = new TLine(0.1,0.7,0.9,0.7);
   l7->SetLineColor(7); l7->SetLineWidth(7) ; l7->SetLineStyle(7) ; l7->Draw();
   TLine *l8 = new TLine(0.1,0.8,0.9,0.8);
   l8->SetLineColor(8); l8->SetLineWidth(8) ; l8->SetLineStyle(8) ; l8->Draw();
   TLine *l9 = new TLine(0.1,0.9,0.9,0.9);
   l9->SetLineColor(9); l9->SetLineWidth(9) ; l9->SetLineStyle(9) ; l9->Draw();

   EndTest("TLine");
};


//______________________________________________________________________________
void tmarker()
{
   // Test TMarker

   StartTest(100,800);

   gC->Range(0,0,1,1);
   gC->SetFillColor(0);
   gC->SetBorderSize(2);
   int i;
   Double_t x = 0.5;
   Double_t y = 0.1;
   Double_t dy = 0.05;
   for (i = 1; i<=7; i++) {
      tmarker_draw(x, y, i, dy);
      y = y+dy;
   }
   for (i = 20; i<=30; i++) {
      tmarker_draw(x, y, i, dy);
      y = y+dy;
   }

   EndTest("TMarker");
};


//______________________________________________________________________________
void tmarker_draw(Double_t x, Double_t y, Int_t mt, Double_t d)
{
   // Auxiliary function used by "tmarker"

   char val[3];
   sprintf(val,"%d",mt);
   double dy=d/3;
   TMarker *m  = new TMarker(x+0.1, y, mt);
   TText   *t  = new TText(x-0.1, y, val);
   TLine   *l1 = new TLine(0,y,1,y);
   TLine   *l2 = new TLine(0,y+dy,1,y+dy);
   TLine   *l3 = new TLine(0,y-dy,1,y-dy);
   l2->SetLineStyle(2);
   l3->SetLineStyle(2);
   m->SetMarkerSize(3.8);
   m->SetMarkerColor(kRed);
   t->SetTextAlign(32);
   t->SetTextSize(0.3);
   t->Draw();
   l1->Draw();
   l2->Draw();
   l3->Draw();
   m->Draw();
}


//______________________________________________________________________________
void tpolyline()
{
   // Test TPolyLine

   StartTest(700,500);

   gC->Range(0,30,11,650);
   Double_t x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
   Double_t y[10] = {200, 300, 300, 200, 200, 100, 10, 580, 10, 600};
   TPolyLine *p = new TPolyLine(10,x,y);
   p->SetLineWidth(3);
   p->SetLineColor(2);
   p->Draw("F");
   p->Draw("");

   EndTest("TPolyLine");
};


//______________________________________________________________________________
void patterns()
{
   // Test Patterns

   StartTest(700,900);

   gC->Range(0,0,1,1);
   gC->SetBorderSize(2);
   gC->SetFrameFillColor(0);
   Double_t bh = 0.059;
   Double_t db = 0.01; 
   Double_t y  = 0.995;
   Int_t i,j=3001;
   for (i=1; i<=5; i++) {
      patterns_box(j++, 0.01, y-bh, 0.19, y);
      patterns_box(j++, 0.21, y-bh, 0.39, y);
      patterns_box(j++, 0.41, y-bh, 0.59, y);
      patterns_box(j++, 0.61, y-bh, 0.79, y);
      patterns_box(j++, 0.81, y-bh, 0.99, y);
      y = y-bh-db;
   }
   y = y-3*db;
   gStyle->SetHatchesSpacing(2.0);
   gStyle->SetHatchesLineWidth(3);
   Int_t j1 = 3144;
   Int_t j2 = 3305;
   Int_t j3 = 3350;
   Int_t j4 = 3490;
   Int_t j5 = 3609;
   for (i=1; i<=9; i++) {
      if (i==6) {j2 += 10; j3 += 1; j4 += 1; j5 += 10;}
      if (i==5) {j4 -= 10; j5 -= 1;}
      patterns_box(j1, 0.01, y-bh, 0.19, y);
      patterns_box(j2, 0.21, y-bh, 0.39, y);
      patterns_box(j3, 0.41, y-bh, 0.59, y);
      patterns_box(j4, 0.61, y-bh, 0.79, y);
      patterns_box(j5, 0.81, y-bh, 0.99, y);
      j1 += 100;
      j2 += 10;
      j3 += 1;
      j4 -= 9;
      j5 += 9;
      y = y-bh-db;
   }

   EndTest("Fill patterns");
};


//______________________________________________________________________________
void patterns_box(Int_t pat, Double_t x1, Double_t y1, Double_t x2, Double_t  y2)
{
   // Auxiliary function used by "patterns"

   TBox b;
   b.SetFillColor(1);
   b.SetFillStyle(pat);
   b.DrawBox(x1,y1,x2,y2);
   b.SetFillStyle(0);
   b.DrawBox(x1,y1,x2,y2);
   b.SetFillColor(0);
   b.SetFillStyle(1000);
   Double_t dx = (x2-x1)/3;
   Double_t dy = (y2-y1)/3;
   Double_t h  = (y2-y1)/3;
   b.DrawBox(x1+dx, y1+dy, x2-dx, y2-dy);
   b.SetFillStyle(0);
   b.DrawBox(x1+dx, y1+dy, x2-dx, y2-dy);
   TLatex l;
   l.SetTextAlign(22);
   l.SetTextSize(h);
   char s[1];
   sprintf(s,"%d",pat);
   l.DrawLatex((x1+x2)/2, (y1+y2)/2, s);
}


//______________________________________________________________________________
void ttext1()
{
   // 1st TText test.

   StartTest(900,500);

   gC->Range(0,0,1,1);
   gC->SetBorderSize(2);
   gC->SetFrameFillColor(0);
   TLine *lv = new TLine(0.5,0.0,0.5,1.0);
   lv->Draw();
   for (float s=0.1; s<1.0 ; s+=0.1) {
      TLine *lh = new TLine(0.,s,1.,s);
      lh->Draw();
   }
   TText *tex1b = new TText(0.02,0.4,"jgabcdefhiklmnopqrstuvwxyz_{}");
   tex1b->SetTextFont(161);
   tex1b->SetTextColor(2);
   tex1b->SetTextAngle(0);
   tex1b->SetTextAlign(11);
   tex1b->SetTextSize(0.1);
   tex1b->Draw();
   TText *tex1 = new TText(0.5,0.1,"j0al {&`ag}_:^)Jj");
   tex1->SetTextFont(41);
   tex1->SetTextColor(2);
   tex1->SetTextAngle( 0);
   tex1->SetTextAlign(21);
   tex1->SetTextSize(0.15);
   tex1->Draw();
   TText *tex2 = new TText(0.5,0.5,"j0Al {&`ag}_:^)Jj");
   tex2->SetTextColor(3);
   tex2->SetTextFont(21);
   tex2->SetTextAlign(21);
   tex2->SetTextSize(0.1);
   tex2->Draw();
   TText *tex3 = new TText(0.5,0.3,"j0Al {&`ag}_:^)Jj");
   tex3->SetTextColor(4);
   tex3->SetTextFont(31);
   tex3->SetTextAlign(31);
   tex3->SetTextSize(0.1);
   tex3->Draw();
   TText *tex4 = new TText(0.5,0.8,"j0Al {&`ag}_:^)Jj");
   tex4->SetTextColor(5);
   tex4->SetTextFont(71);
   tex4->SetTextAlign(22);
   tex4->SetTextSize(0.07);
   tex4->Draw();
   TText *tex5 = new TText(0.5,0.7,"13 j0Al {&`ag}_:^)Jj");
   tex5->SetTextColor(6);
   tex5->SetTextFont(51);
   tex5->SetTextAlign(13);
   tex5->SetTextSize(0.1);
   tex5->Draw();

   EndTest("TText 1 (Text attributes)");
}


//______________________________________________________________________________
void ttext2()
{
   // 2nd TText test. A very long text string.

   StartTest(600,600);

   TText t(0.001,0.5,"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789");
   t.SetTextFont(42); t.SetTextSize(0.02);
   t.Draw();

   EndTest("TText 2 (A very long text string)");
}


//______________________________________________________________________________
void tlatex1()
{
   // 1st TLatex test.

   StartTest(600,700);

   TLatex l;
   l.SetTextAlign(12);
   l.SetTextSize(0.04);
   l.DrawLatex(0.1,0.9,"1) C(x) = d #sqrt{#frac{2}{#lambdaD}}  #int^{x}_{0}cos(#frac{#pi}{2}t^{2})dt");
   l.DrawLatex(0.1,0.7,"2) C(x) = d #sqrt{#frac{2}{#lambdaD}}  #int^{x}cos(#frac{#pi}{2}t^{2})dt");
   l.DrawLatex(0.1,0.5,"3) R = |A|^{2} = #frac{1}{2}#left(#[]{#frac{1}{2}+C(V)}^{2}+#[]{#frac{1}{2}+S(V)}^{2}#right)");
   l.DrawLatex(0.1,0.3,"4) F(t) = #sum_{i=-#infty}^{#infty}A(i)cos#[]{#frac{i}{t+i}}");
   l.DrawLatex(0.1,0.1,"5) {}_{3}^{7}Li");

   EndTest("TLatex 1");
}


//______________________________________________________________________________
void tlatex2()
{
   // 2nd TLatex test.

   StartTest(700,500);

   TLatex l;
   l.SetTextAlign(23);
   l.SetTextSize(0.1);
   l.DrawLatex(0.5,0.95,"e^{+}e^{-}#rightarrowZ^{0}#rightarrowI#bar{I}, q#bar{q}");
   l.DrawLatex(0.5,0.75,"|#vec{a}#bullet#vec{b}|=#Sigmaa^{i}_{jk}+b^{bj}_{i}");
   l.DrawLatex(0.5,0.5,"i(#partial_{#mu}#bar{#psi}#gamma^{#mu}+m#bar{#psi})=0#Leftrightarrow(#Box+m^{2})#psi=0");
   l.DrawLatex(0.5,0.3,"L_{em}=eJ^{#mu}_{em}A_{#mu} , J^{#mu}_{em}=#bar{I}#gamma_{#mu}I , M^{j}_{i}=#SigmaA_{#alpha}#tau^{#alphaj}_{i}");

   EndTest("TLatex 2");
}


//______________________________________________________________________________
void tlatex3()
{
   // 3rd TLatex test.

   StartTest(700,500);

   TPaveText pt(.05,.1,.95,.8);
   pt.AddText("#frac{2s}{#pi#alpha^{2}}  #frac{d#sigma}{dcos#theta} (e^{+}e^{-} #rightarrow f#bar{f} ) = \
   #left| #frac{1}{1 - #Delta#alpha} #right|^{2} (1+cos^{2}#theta)");
   pt.AddText("+ 4 Re #left{ #frac{2}{1 - #Delta#alpha} #chi(s) #[]{#hat{g}_{#nu}^{e}#hat{g}_{#nu}^{f} \
   (1 + cos^{2}#theta) + 2 #hat{g}_{a}^{e}#hat{g}_{a}^{f} cos#theta) } #right}");
   pt.AddText("+ 16#left|#chi(s)#right|^{2}\
   #left[(#hat{g}_{a}^{e}^{2} + #hat{g}_{v}^{e}^{2})\
   (#hat{g}_{a}^{f}^{2} + #hat{g}_{v}^{f}^{2})(1+cos^{2}#theta)\
   + 8 #hat{g}_{a}^{e} #hat{g}_{a}^{f} #hat{g}_{v}^{e} #hat{g}_{v}^{f}cos#theta#right] ");
   pt.SetLabel("Born equation");
   pt.Draw();

   EndTest("TLatex 3 (TLatex in TPaveText)");
}


//______________________________________________________________________________
void tlatex4()
{
   // 4th TLatex test.

   StartTest(600,700);

   TLatex l;
   l.SetTextSize(0.03);
   l.SetTextAlign(22);
   l.DrawLatex(0.165, 0.95, "Lower case");
   l.DrawLatex(0.495, 0.95, "Upper case");
   l.DrawLatex(0.825, 0.95, "Variations");
   l.SetTextAlign(12);
   float y, x1, x2;
   y = 0.90; x1 = 0.07; x2 = x1+0.2;
                 l.DrawLatex(x1, y, "alpha : ")   ; l.DrawLatex(x2, y, "#alpha");
   y -= 0.0375 ; l.DrawLatex(x1, y, "beta : ")    ; l.DrawLatex(x2, y, "#beta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "gamma : ")   ; l.DrawLatex(x2, y, "#gamma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "delta : ")   ; l.DrawLatex(x2, y, "#delta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "epsilon : ") ; l.DrawLatex(x2, y, "#epsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "zeta : ")    ; l.DrawLatex(x2, y, "#zeta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "eta : ")     ; l.DrawLatex(x2, y, "#eta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "theta : ")   ; l.DrawLatex(x2, y, "#theta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "iota : ")    ; l.DrawLatex(x2, y, "#iota");
   y -= 0.0375 ; l.DrawLatex(x1, y, "kappa : ")   ; l.DrawLatex(x2, y, "#kappa");
   y -= 0.0375 ; l.DrawLatex(x1, y, "lambda : ")  ; l.DrawLatex(x2, y, "#lambda");
   y -= 0.0375 ; l.DrawLatex(x1, y, "mu : ")      ; l.DrawLatex(x2, y, "#mu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "nu : ")      ; l.DrawLatex(x2, y, "#nu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "xi : ")      ; l.DrawLatex(x2, y, "#xi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "omicron : ") ; l.DrawLatex(x2, y, "#omicron");
   y -= 0.0375 ; l.DrawLatex(x1, y, "pi : ")      ; l.DrawLatex(x2, y, "#pi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "rho : ")     ; l.DrawLatex(x2, y, "#rho");
   y -= 0.0375 ; l.DrawLatex(x1, y, "sigma : ")   ; l.DrawLatex(x2, y, "#sigma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "tau : ")     ; l.DrawLatex(x2, y, "#tau");
   y -= 0.0375 ; l.DrawLatex(x1, y, "upsilon : ") ; l.DrawLatex(x2, y, "#upsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "phi : ")     ; l.DrawLatex(x2, y, "#phi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "chi : ")     ; l.DrawLatex(x2, y, "#chi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "psi : ")     ; l.DrawLatex(x2, y, "#psi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "omega : ")   ; l.DrawLatex(x2, y, "#omega");
   y = 0.90; x1 = 0.40; x2 = x1+0.2;
                 l.DrawLatex(x1, y, "Alpha : ")   ; l.DrawLatex(x2, y, "#Alpha");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Beta : ")    ; l.DrawLatex(x2, y, "#Beta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Gamma : ")   ; l.DrawLatex(x2, y, "#Gamma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Delta : ")   ; l.DrawLatex(x2, y, "#Delta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Epsilon : ") ; l.DrawLatex(x2, y, "#Epsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Zeta : ")    ; l.DrawLatex(x2, y, "#Zeta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Eta : ")     ; l.DrawLatex(x2, y, "#Eta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Theta : ")   ; l.DrawLatex(x2, y, "#Theta");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Iota : ")    ; l.DrawLatex(x2, y, "#Iota");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Kappa : ")   ; l.DrawLatex(x2, y, "#Kappa");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Lambda : ")  ; l.DrawLatex(x2, y, "#Lambda");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Mu : ")      ; l.DrawLatex(x2, y, "#Mu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Nu : ")      ; l.DrawLatex(x2, y, "#Nu");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Xi : ")      ; l.DrawLatex(x2, y, "#Xi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Omicron : ") ; l.DrawLatex(x2, y, "#Omicron");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Pi : ")      ; l.DrawLatex(x2, y, "#Pi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Rho : ")     ; l.DrawLatex(x2, y, "#Rho");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Sigma : ")   ; l.DrawLatex(x2, y, "#Sigma");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Tau : ")     ; l.DrawLatex(x2, y, "#Tau");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Upsilon : ") ; l.DrawLatex(x2, y, "#Upsilon");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Phi : ")     ; l.DrawLatex(x2, y, "#Phi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Chi : ")     ; l.DrawLatex(x2, y, "#Chi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Psi : ")     ; l.DrawLatex(x2, y, "#Psi");
   y -= 0.0375 ; l.DrawLatex(x1, y, "Omega : ")   ; l.DrawLatex(x2, y, "#Omega");
   x1 = 0.73; x2 = x1+0.2;
   y = 0.7500 ; l.DrawLatex(x1, y, "varepsilon : ") ; l.DrawLatex(x2, y, "#varepsilon");
   y = 0.6375 ; l.DrawLatex(x1, y, "vartheta : ")   ; l.DrawLatex(x2, y, "#vartheta");
   y = 0.2625 ; l.DrawLatex(x1, y, "varsigma : ")   ; l.DrawLatex(x2, y, "#varsigma");
   y = 0.1875 ; l.DrawLatex(x1, y, "varUpsilon : ") ; l.DrawLatex(x2, y, "#varUpsilon");
   y = 0.1500 ; l.DrawLatex(x1, y, "varphi : ")     ; l.DrawLatex(x2, y, "#varphi");
   y = 0.0375 ; l.DrawLatex(x1, y, "varomega : ")   ; l.DrawLatex(x2, y, "#varomega");

   EndTest("TLatex 4 (Greek letters)");
}


//______________________________________________________________________________
void tlatex5()
{
   // 5th TLatex test.

   StartTest(600,600);

   TLatex l;
   l.SetTextSize(0.03);
   l.SetTextAlign(12);
   float y, step, x1, x2;
   y = 0.96; step = 0.0465; x1 = 0.02; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#club")           ; l.DrawText(x2, y, "#club");
   y -= step ; l.DrawLatex(x1, y, "#voidn")          ; l.DrawText(x2, y, "#voidn");
   y -= step ; l.DrawLatex(x1, y, "#leq")            ; l.DrawText(x2, y, "#leq");
   y -= step ; l.DrawLatex(x1, y, "#approx")         ; l.DrawText(x2, y, "#approx");
   y -= step ; l.DrawLatex(x1, y, "#in")             ; l.DrawText(x2, y, "#in");
   y -= step ; l.DrawLatex(x1, y, "#supset")         ; l.DrawText(x2, y, "#supset");
   y -= step ; l.DrawLatex(x1, y, "#cap")            ; l.DrawText(x2, y, "#cap");
   y -= step ; l.DrawLatex(x1, y, "#ocopyright")     ; l.DrawText(x2, y, "#ocopyright");
   y -= step ; l.DrawLatex(x1, y, "#trademark")      ; l.DrawText(x2, y, "#trademark");
   y -= step ; l.DrawLatex(x1, y, "#times")          ; l.DrawText(x2, y, "#times");
   y -= step ; l.DrawLatex(x1, y, "#bullet")         ; l.DrawText(x2, y, "#bullet");
   y -= step ; l.DrawLatex(x1, y, "#voidb")          ; l.DrawText(x2, y, "#voidb");
   y -= step ; l.DrawLatex(x1, y, "#doublequote")    ; l.DrawText(x2, y, "#doublequote");
   y -= step ; l.DrawLatex(x1, y, "#lbar")           ; l.DrawText(x2, y, "#lbar");
   y -= step ; l.DrawLatex(x1, y, "#arcbottom")      ; l.DrawText(x2, y, "#arcbottom");
   y -= step ; l.DrawLatex(x1, y, "#downarrow")      ; l.DrawText(x2, y, "#downarrow");
   y -= step ; l.DrawLatex(x1, y, "#leftrightarrow") ; l.DrawText(x2, y, "#leftrightarrow");
   y -= step ; l.DrawLatex(x1, y, "#Downarrow")      ; l.DrawText(x2, y, "#Downarrow");
   y -= step ; l.DrawLatex(x1, y, "#Leftrightarrow") ; l.DrawText(x2, y, "#Leftrightarrow");
   y -= step ; l.DrawLatex(x1, y, "#void8")          ; l.DrawText(x2, y, "#void8");
   y -= step ; l.DrawLatex(x1, y, "#hbar")           ; l.DrawText(x2, y, "#hbar");
   y = 0.96; step = 0.0465; x1 = 0.27; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#diamond")        ; l.DrawText(x2, y, "#diamond");
   y -= step ; l.DrawLatex(x1, y, "#aleph")          ; l.DrawText(x2, y, "#aleph");
   y -= step ; l.DrawLatex(x1, y, "#geq")            ; l.DrawText(x2, y, "#geq");
   y -= step ; l.DrawLatex(x1, y, "#neq")            ; l.DrawText(x2, y, "#neq");
   y -= step ; l.DrawLatex(x1, y, "#notin")          ; l.DrawText(x2, y, "#notin");
   y -= step ; l.DrawLatex(x1, y, "#subseteq")       ; l.DrawText(x2, y, "#subseteq");
   y -= step ; l.DrawLatex(x1, y, "#cup")            ; l.DrawText(x2, y, "#cup");
   y -= step ; l.DrawLatex(x1, y, "#copyright")      ; l.DrawText(x2, y, "#copyright");
   y -= step ; l.DrawLatex(x1, y, "#void3")          ; l.DrawText(x2, y, "#void3");
   y -= step ; l.DrawLatex(x1, y, "#divide")         ; l.DrawText(x2, y, "#divide");
   y -= step ; l.DrawLatex(x1, y, "#circ")           ; l.DrawText(x2, y, "#circ");
   y -= step ; l.DrawLatex(x1, y, "#infty")          ; l.DrawText(x2, y, "#infty");
   y -= step ; l.DrawLatex(x1, y, "#angle")          ; l.DrawText(x2, y, "#angle");
   y -= step ; l.DrawLatex(x1, y, "#cbar")           ; l.DrawText(x2, y, "#cbar");
   y -= step ; l.DrawLatex(x1, y, "#arctop")         ; l.DrawText(x2, y, "#arctop");
   y -= step ; l.DrawLatex(x1, y, "#leftarrow")      ; l.DrawText(x2, y, "#leftarrow");
   y -= step ; l.DrawLatex(x1, y, "#otimes")         ; l.DrawText(x2, y, "#otimes");
   y -= step ; l.DrawLatex(x1, y, "#Leftarrow")      ; l.DrawText(x2, y, "#Leftarrow");
   y -= step ; l.DrawLatex(x1, y, "#prod")           ; l.DrawText(x2, y, "#prod");
   y -= step ; l.DrawLatex(x1, y, "#Box")            ; l.DrawText(x2, y, "#Box");
   y -= step ; l.DrawLatex(x1, y, "#parallel")       ; l.DrawText(x2, y, "#parallel");
   y = 0.96; step = 0.0465; x1 = 0.52; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#heart")          ; l.DrawText(x2, y, "#heart");
   y -= step ; l.DrawLatex(x1, y, "#Jgothic")        ; l.DrawText(x2, y, "#Jgothic");
   y -= step ; l.DrawLatex(x1, y, "#LT")             ; l.DrawText(x2, y, "#LT");
   y -= step ; l.DrawLatex(x1, y, "#equiv")          ; l.DrawText(x2, y, "#equiv");
   y -= step ; l.DrawLatex(x1, y, "#subset")         ; l.DrawText(x2, y, "#subset");
   y -= step ; l.DrawLatex(x1, y, "#supseteq")       ; l.DrawText(x2, y, "#supseteq");
   y -= step ; l.DrawLatex(x1, y, "#wedge")          ; l.DrawText(x2, y, "#wedge");
   y -= step ; l.DrawLatex(x1, y, "#oright")         ; l.DrawText(x2, y, "#oright");
   y -= step ; l.DrawLatex(x1, y, "#AA")             ; l.DrawText(x2, y, "#AA");
   y -= step ; l.DrawLatex(x1, y, "#pm")             ; l.DrawText(x2, y, "#pm");
   y -= step ; l.DrawLatex(x1, y, "#3dots")          ; l.DrawText(x2, y, "#3dots");
   y -= step ; l.DrawLatex(x1, y, "#nabla")          ; l.DrawText(x2, y, "#nabla");
   y -= step ; l.DrawLatex(x1, y, "#downleftarrow")  ; l.DrawText(x2, y, "#downleftarrow");
   y -= step ; l.DrawLatex(x1, y, "#topbar")         ; l.DrawText(x2, y, "#topbar");
   y -= step ; l.DrawLatex(x1, y, "#arcbar")         ; l.DrawText(x2, y, "#arcbar");
   y -= step ; l.DrawLatex(x1, y, "#uparrow")        ; l.DrawText(x2, y, "#uparrow");
   y -= step ; l.DrawLatex(x1, y, "#oplus")          ; l.DrawText(x2, y, "#oplus");
   y -= step ; l.DrawLatex(x1, y, "#Uparrow")        ; l.DrawText(x2, y, "#Uparrow");
   y -= step ; l.DrawLatex(x1, y-0.01, "#sum")       ; l.DrawText(x2, y, "#sum");
   y -= step ; l.DrawLatex(x1, y, "#perp")           ; l.DrawText(x2, y, "#perp");
   y = 0.96; step = 0.0465; x1 = 0.77; x2 = x1+0.04;
               l.DrawLatex(x1, y, "#spade")          ; l.DrawText(x2, y, "#spade");
   y -= step ; l.DrawLatex(x1, y, "#Rgothic")        ; l.DrawText(x2, y, "#Rgothic");
   y -= step ; l.DrawLatex(x1, y, "#GT")             ; l.DrawText(x2, y, "#GT");
   y -= step ; l.DrawLatex(x1, y, "#propto")         ; l.DrawText(x2, y, "#propto");
   y -= step ; l.DrawLatex(x1, y, "#notsubset")      ; l.DrawText(x2, y, "#notsubset");
   y -= step ; l.DrawLatex(x1, y, "#oslash")         ; l.DrawText(x2, y, "#oslash");
   y -= step ; l.DrawLatex(x1, y, "#vee")            ; l.DrawText(x2, y, "#vee");
   y -= step ; l.DrawLatex(x1, y, "#void1")          ; l.DrawText(x2, y, "#void1");
   y -= step ; l.DrawLatex(x1, y, "#aa")             ; l.DrawText(x2, y, "#aa");
   y -= step ; l.DrawLatex(x1, y, "#/")              ; l.DrawText(x2, y, "#/");
   y -= step ; l.DrawLatex(x1, y, "#upoint")         ; l.DrawText(x2, y, "#upoint");
   y -= step ; l.DrawLatex(x1, y, "#partial")        ; l.DrawText(x2, y, "#partial");
   y -= step ; l.DrawLatex(x1, y, "#corner")         ; l.DrawText(x2, y, "#corner");
   y -= step ; l.DrawLatex(x1, y, "#ltbar")          ; l.DrawText(x2, y, "#ltbar");
   y -= step ; l.DrawLatex(x1, y, "#bottombar")      ; l.DrawText(x2, y, "#bottombar");
   y -= step ; l.DrawLatex(x1, y, "#rightarrow")     ; l.DrawText(x2, y, "#rightarrow");
   y -= step ; l.DrawLatex(x1, y, "#surd")           ; l.DrawText(x2, y, "#surd");
   y -= step ; l.DrawLatex(x1, y, "#Rightarrow")     ; l.DrawText(x2, y, "#Rightarrow");
   y -= step ; l.DrawLatex(x1, y-0.015, "#int")      ; l.DrawText(x2, y, "#int");
   y -= step ; l.DrawLatex(x1, y, "#odot")           ; l.DrawText(x2, y, "#odot");

   EndTest("TLatex 5 (Mathematical Symbols)");
}


//______________________________________________________________________________
void tgaxis1()
{
   // 1st TGaxis test.

   StartTest(700,500);

   gC->Range(-10,-1,10,1);
   TGaxis *axis1 = new TGaxis(-4.5,-0.2,5.5,-0.2,-6,8,510,"");
   axis1->SetName("axis1");
   axis1->Draw();
   TGaxis *axis2 = new TGaxis(-4.5,0.2,5.5,0.2,0.001,10000,510,"G");
   axis2->SetName("axis2");
   axis2->Draw();
   TGaxis *axis3 = new TGaxis(-9,-0.8,-9,0.8,-8,8,50510,"");
   axis3->SetName("axis3");
   axis3->SetTitle("axis3");
   axis3->SetTitleOffset(0.5);
   axis3->Draw();
   TGaxis *axis4 = new TGaxis(-7,-0.8,-7,0.8,1,10000,50510,"G");
   axis4->SetName("axis4");
   axis4->SetTitle("axis4");
   axis4->Draw();
   TGaxis *axis5 = new TGaxis(-4.5,-0.6,5.5,-0.6,1.2,1.32,80506,"-+");
   axis5->SetName("axis5");
   axis5->SetLabelSize(0.03);
   axis5->SetTextFont(72);
   axis5->Draw();
   TGaxis *axis6 = new TGaxis(-4.5,0.5,5.5,0.5,100,900,50510,"-");
   axis6->SetName("axis6");
   axis6->Draw();
   TGaxis *axis6a = new TGaxis(-5.5,0.85,5.5,0.85,0,4.3e-6,510,"");
   axis6a->SetName("axis6a");
   axis6a->Draw();
   TGaxis *axis7 = new TGaxis(8,-0.8,8,0.8,0,9000,50510,"+L");
   axis7->SetName("axis7");
   axis7->Draw();
   TGaxis *axis8 = new TGaxis(6.5,0.8,6.499,-0.8,0,90,50510,"-");
   axis8->SetName("axis8");
   axis8->Draw();

   EndTest("TGaxis 1");
}


//______________________________________________________________________________
void tgaxis2()
{
   // 2nd TGaxis test.

   StartTest(600,700);

   gC->Range(-10,-1,10,1);
   TGaxis *axis1 = new TGaxis(-5,-0.2,6,-0.2,-6,8,510,"");
   axis1->SetName("axis1");
   axis1->Draw();
   TGaxis *axis2 = new TGaxis(-5,0.2,6,0.2,0.001,10000,510,"G");
   axis2->SetName("axis2");
   axis2->Draw();
   TGaxis *axis3 = new TGaxis(-9,-0.8,-9,0.8,-8,8,50510,"");
   axis3->SetName("axis3");
   axis3->Draw();
   TGaxis *axis4 = new TGaxis(-7,-0.8,-7,0.8,1,10000,50510,"G");
   axis4->SetName("axis4");
   axis4->Draw();
   TGaxis *axis5 = new TGaxis(-5,-0.6,6,-0.6,1.2,1.32,80506,"-+");
   axis5->SetName("axis5");
   axis5->SetLabelSize(0.03);
   axis5->SetTextFont(72);
   axis5->SetLabelOffset(0.025);
   axis5->Draw();
   TGaxis *axis6 = new TGaxis(-5,0.6,6,0.6,100,900,50510,"-");
   axis6->SetName("axis6");
   axis6->Draw();
   TGaxis *axis7 = new TGaxis(8,-0.8,8,0.8,0,9000,50510,"+L");
   axis7->SetName("axis7");
   axis7->SetLabelOffset(0.01);
   axis7->Draw();

   EndTest("TGaxis 2");
}


//______________________________________________________________________________
void tgaxis3()
{
   // 3rd TGaxis test.

   StartTest(700,900);

   time_t script_time;
   script_time = time(0);
   script_time = 3600*(int)(script_time/3600);
   gStyle->SetTimeOffset(script_time);
   gC->Divide(1,3);
   gC->SetFillColor(28);
   int i;
   gStyle->SetTitleH(0.08);
   float noise;
   TH1F *ht = new TH1F("ht","Love at first sight",3000,0.,2000.);
   for (i=1;i<3000;i++) {
      noise = gRandom->Gaus(0,120);
      if (i>700) {
         noise += 1000*sin((i-700)*6.28/30)*exp((double)(700-i)/300);
      }
      ht->SetBinContent(i,noise);
   }
   gC->cd(1);
   ht->SetLineColor(2);
   ht->GetXaxis()->SetLabelSize(0.05);
   ht->Draw();
   ht->GetXaxis()->SetTimeDisplay(1);
   float x[100], t[100];
   for (i=0;i<100;i++) {
      x[i] = sin(i*4*3.1415926/50)*exp(-(double)i/20);
      t[i] = 6000+(double)i/20;
   }
   TGraph *gt = new TGraph(100,t,x);
   gt->SetTitle("Politics");
   gC->cd(2);
   gt->SetFillColor(19);
   gt->SetLineColor(5);
   gt->SetLineWidth(2);
   gt->Draw("AL");
   gt->GetXaxis()->SetLabelSize(0.05);
   gt->GetXaxis()->SetTimeDisplay(1);
   gPad->Modified();
   float x2[10], t2[10];
   for (i=0;i<10;i++) {
      x2[i] = gRandom->Gaus(500,100)*i;
      t2[i] = i*365*86400;
   }
   TGraph *gt2 = new TGraph(10,t2,x2);
   gt2->SetTitle("Number of monkeys on the moon");
   gC->cd(3);
   gt2->SetFillColor(19);
   gt2->SetMarkerColor(4);
   gt2->SetMarkerStyle(29);
   gt2->SetMarkerSize(1.3);
   gt2->Draw("AP");
   gt2->GetXaxis()->SetLabelSize(0.05);
   gt2->GetXaxis()->SetTimeDisplay(1);
   gt2->GetXaxis()->SetTimeFormat("y. %Y");
   
   EndTest("TGaxis 3 (Time on axis)");
}


//______________________________________________________________________________
void tgaxis4()
{
   // 4th TGaxis test.

   StartTest(600,700);

   TDatime T0(2003,1,1,0,0,0);
   int X0 = T0.Convert();
   gStyle->SetTimeOffset(X0);
   TDatime T1(2002,9,23,0,0,0);
   int X1 = T1.Convert()-X0;
   TDatime T2(2003,3,7,0,0,0);
   int X2 = T2.Convert()-X0;
   TH1F * h1 = new TH1F("h1","test",100,X1,X2);
   TRandom r;
   for (Int_t i=0;i<30000;i++) {
      Double_t noise = r.Gaus(0.5*(X1+X2),0.1*(X2-X1));
      h1->Fill(noise);
   }
   h1->GetXaxis()->SetTimeDisplay(1);
   h1->GetXaxis()->SetLabelSize(0.03);
   h1->GetXaxis()->SetTimeFormat("%Y:%m:%d");
   h1->Draw();

   EndTest("TGaxis 4 (Time on axis)");
   delete h1;
}


//______________________________________________________________________________
void tellipse()
{
   // TEllipse test.

   StartTest(700,800);

   gC->Range(0,0,1,1);
   TPaveLabel pel(0.1,0.8,0.9,0.95,"Examples of Ellipses");
   pel.SetFillColor(42);
   pel.Draw();
   TEllipse el1(0.25,0.25,.1,.2);
   el1.Draw();
   TEllipse el2(0.25,0.6,.2,.1);
   el2.SetFillColor(6);
   el2.SetFillStyle(3008);
   el2.Draw();
   TEllipse el3(0.75,0.6,.2,.1,45,315);
   el3.SetFillColor(2);
   el3.SetFillStyle(1001);
   el3.SetLineColor(4);
   el3.Draw();
   TEllipse el4(0.75,0.25,.2,.15,45,315,62);
   el4.SetFillColor(5);
   el4.SetFillStyle(1001);
   el4.SetLineColor(4);
   el4.SetLineWidth(6);
   el4.Draw();

   EndTest("TEllipse");
}


//______________________________________________________________________________
void feynman()
{
   // Feynman diagrams test.

   StartTest(600,300);

   gC->Range(0, 0, 140, 60);
   Int_t linsav = gStyle->GetLineWidth();
   gStyle->SetLineWidth(3);
   TLatex t;
   t.SetTextAlign(22);
   t.SetTextSize(0.1);
   TLine * l;
   l = new TLine(10, 10, 30, 30); l->Draw();
   l = new TLine(10, 50, 30, 30); l->Draw();
   TCurlyArc *ginit = new TCurlyArc(30, 30, 12.5*TMath::Sqrt(2), 135, 225);
   ginit->SetWavy();
   ginit->Draw();
   t.DrawLatex(7,6,"e^{-}");
   t.DrawLatex(7,55,"e^{+}");
   t.DrawLatex(7,30,"#gamma");
   TCurlyLine *gamma = new TCurlyLine(30, 30, 55, 30);
   gamma->SetWavy();
   gamma->Draw();
   t.DrawLatex(42.5,37.7,"#gamma");
   TArc *a = new TArc(70, 30, 15);
   a->Draw();
   t.DrawLatex(55, 45,"#bar{q}");
   t.DrawLatex(85, 15,"q");
   TCurlyLine *gluon = new TCurlyLine(70, 45, 70, 15);
   gluon->Draw();
   t.DrawLatex(77.5,30,"g");
   TCurlyLine *z0 = new TCurlyLine(85, 30, 110, 30);
   z0->SetWavy();
   z0->Draw();
   t.DrawLatex(100, 37.5,"Z^{0}");
   l = new TLine(110, 30, 130, 10); l->Draw();
   l = new TLine(110, 30, 130, 50); l->Draw();
   TCurlyArc *gluon1 = new TCurlyArc(110, 30, 12.5*TMath::Sqrt(2), 315, 45);
   gluon1->Draw();
   t.DrawLatex(135,6,"#bar{q}");
   t.DrawLatex(135,55,"q");
   t.DrawLatex(135,30,"g");
   gC->Update();
   gStyle->SetLineWidth(linsav);

   EndTest("Feynman diagrams");
}


//______________________________________________________________________________
void tgraph1()
{
   // 1st TGraph test.

   StartTest(700,500);

   gC->SetFillColor(42);
   gC->SetGrid();
   const Int_t n = 20;
   Double_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i*0.1;
     y[i] = 10*sin(x[i]+0.2);
   }
   TGraph *gr = new TGraph(n,x,y);
   gr->SetLineColor(2);
   gr->SetLineWidth(4);
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);
   gr->SetTitle("a simple graph");
   gr->GetXaxis()->SetTitle("X title");
   gr->GetYaxis()->SetTitle("Y title");
   gr->Draw("ACP");
   gC->Update();
   gC->GetFrame()->SetFillColor(21);
   gC->GetFrame()->SetBorderSize(12);

   EndTest("TGraph 1");
}


//______________________________________________________________________________
void tgraph2()
{
   // 2nd TGraph test.
   
   StartTest(700,500);

   gC->SetGrid();
   TMultiGraph *mg = new TMultiGraph();
   mg->SetTitle("Exclusion graphs");
   const Int_t n = 35;
   Double_t x1[n], x2[n], x3[n], y1[n], y2[n], y3[n];
   for (Int_t i=0;i<n;i++) {
     x1[i]  = i*0.1;
     x2[i]  = x1[i];
     x3[i]  = x1[i]+.5;
     y1[i] = 10*sin(x1[i]);
     y2[i] = 10*cos(x1[i]);
     y3[i] = 10*sin(x1[i])-2;
   }
   TGraph *gr1 = new TGraph(n,x1,y1);
   gr1->SetLineColor(2);
   gr1->SetLineWidth(1504);
   gr1->SetFillStyle(3005);
   TGraph *gr2 = new TGraph(n,x2,y2);
   gr2->SetLineColor(4);
   gr2->SetLineWidth(-2002);
   gr2->SetFillStyle(3004);
   gr2->SetFillColor(9);
   TGraph *gr3 = new TGraph(n,x3,y3);
   gr3->SetLineColor(5);
   gr3->SetLineWidth(-802);
   gr3->SetFillStyle(3002);
   gr3->SetFillColor(2);
   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);
   mg->Draw("AC");

   EndTest("TGraph 2 (Exclusion Zone)");
}


//______________________________________________________________________________
void tmultigraph()
{
   // TMultigraph and TGraphErrors test

   StartTest(700,500);

   gStyle->SetOptFit();
   gC->SetGrid();
   TMultiGraph *mg = new TMultiGraph();
   Int_t n1 = 10;
   Double_t x1[]  = {-0.1, 0.05, 0.25, 0.35, 0.5, 0.61,0.7,0.85,0.89,0.95};
   Double_t y1[]  = {-1,2.9,5.6,7.4,9,9.6,8.7,6.3,4.5,1};
   Double_t ex1[] = {.05,.1,.07,.07,.04,.05,.06,.07,.08,.05};
   Double_t ey1[] = {.8,.7,.6,.5,.4,.4,.5,.6,.7,.8};
   TGraphErrors *gr1 = new TGraphErrors(n1,x1,y1,ex1,ey1);
   gr1->SetMarkerColor(kBlue);
   gr1->SetMarkerStyle(21);
   gr1->Fit("pol6","q");
   mg->Add(gr1);
   Int_t n2 = 10;
   Float_t x2[]  = {-0.28, 0.005, 0.19, 0.29, 0.45, 0.56,0.65,0.80,0.90,1.01};
   Float_t y2[]  = {2.1,3.86,7,9,10,10.55,9.64,7.26,5.42,2};
   Float_t ex2[] = {.04,.12,.08,.06,.05,.04,.07,.06,.08,.04};
   Float_t ey2[] = {.6,.8,.7,.4,.3,.3,.4,.5,.6,.7};
   TGraphErrors *gr2 = new TGraphErrors(n2,x2,y2,ex2,ey2);
   gr2->SetMarkerColor(kRed);
   gr2->SetMarkerStyle(20);
   gr2->Fit("pol5","q");
   mg->Add(gr2);
   mg->Draw("ap");
   gC->Update();
   TPaveStats *stats1 = (TPaveStats*)gr1->GetListOfFunctions()->FindObject("stats");
   TPaveStats *stats2 = (TPaveStats*)gr2->GetListOfFunctions()->FindObject("stats");
   stats1->SetTextColor(kBlue); 
   stats2->SetTextColor(kRed); 
   stats1->SetX1NDC(0.12); stats1->SetX2NDC(0.32); stats1->SetY1NDC(0.75);
   stats2->SetX1NDC(0.72); stats2->SetX2NDC(0.92); stats2->SetY1NDC(0.78);
   gC->Modified();

   EndTest("TMultigraph and TGraphErrors");
}


//______________________________________________________________________________
void options2d1()
{
   // 1st 2D options Test 
   
   StartTest(800,600);

   gStyle->SetOptStat(0);
   gStyle->SetPalette(1);
   gStyle->SetCanvasColor(33);
   gStyle->SetFrameFillColor(18);
   TF2 *f2 = new TF2("f2","xygaus + xygaus(5) + xylandau(10)",-4,4,-4,4);
   Double_t params[] = {130,-1.4,1.8,1.5,1, 150,2,0.5,-2,0.5, 3600,-2,0.7,-3,0.3};
   f2->SetParameters(params);
   gH2 = new TH2F("h2","xygaus + xygaus(5) + xylandau(10)",20,-4,4,20,-4,4);
   gH2->SetFillColor(46);
   gH2->FillRandom("f2",40000);

   TPaveLabel pl;
   Float_t x1=0.67, y1=0.875, x2=0.85, y2=0.95;
   gC->Divide(2,2);
   gC->SetFillColor(17);
   gC->cd(1);
   gH2->Draw();       pl.DrawPaveLabel(x1,y1,x2,y2,"SCAT","brNDC");
   gC->cd(2);
   gH2->Draw("box");  pl.DrawPaveLabel(x1,y1,x2,y2,"BOX","brNDC");
   gC->cd(3);
   gH2->Draw("arr");  pl.DrawPaveLabel(x1,y1,x2,y2,"ARR","brNDC");
   gC->cd(4);
   gH2->Draw("colz"); pl.DrawPaveLabel(x1,y1,x2,y2,"COLZ","brNDC");
   
   EndTest("Basic 2D options");
}


//______________________________________________________________________________
void options2d2()
{
   // 2nd 2D options Test 
   
   StartTest(800,600);

   TPaveLabel pl;
   Float_t x1=0.67, y1=0.875, x2=0.85, y2=0.95;
   gPad->SetGrid();
   gC->SetFillColor(17);
   gC->SetGrid();
   gH2->Draw("text"); pl.DrawPaveLabel(x1,y1,x2,y2,"TEXT","brNDC");
   
   EndTest("Text option");
}


//______________________________________________________________________________
void options2d3()
{
   // 3rd 2D options Test 
   
   StartTest(800,600);

   TPaveLabel pl;
   Float_t x1=0.67, y1=0.875, x2=0.85, y2=0.95;
   gC->Divide(2,2);
   gPad->SetGrid();
   gC->SetFillColor(17);
   gC->cd(1);
   gH2->Draw("contz"); pl.DrawPaveLabel(x1,y1,x2,y2,"CONTZ","brNDC");
   gC->cd(2);
   gPad->SetGrid();
   gH2->Draw("cont1"); pl.DrawPaveLabel(x1,y1,x2,y2,"CONT1","brNDC");
   gC->cd(3);
   gPad->SetGrid();
   gH2->Draw("cont2"); pl.DrawPaveLabel(x1,y1,x2,y2,"CONT2","brNDC");
   gC->cd(4);
   gPad->SetGrid();
   gH2->Draw("cont3"); pl.DrawPaveLabel(x1,y1,x2,y2,"CONT3","brNDC");
   
   EndTest("Contour options");
}


//______________________________________________________________________________
void options2d4()
{
   // 4th 2D options Test 
   
   StartTest(800,600);

   TPaveLabel pl;
   Float_t x1=0.67, y1=0.875, x2=0.85, y2=0.95;
   gC->Divide(2,2);
   gC->SetFillColor(17);
   gC->cd(1);
   gH2->Draw("lego");     pl.DrawPaveLabel(x1,y1,x2,y2,"LEGO","brNDC");
   gC->cd(2);
   gH2->Draw("lego1");    pl.DrawPaveLabel(x1,y1,x2,y2,"LEGO1","brNDC");
   gC->cd(3);
   gPad->SetTheta(61); gPad->SetPhi(-82);
   gH2->Draw("surf1pol"); pl.DrawPaveLabel(x1,y1,x2+0.05,y2,"SURF1POL","brNDC");
   gC->cd(4);
   gPad->SetTheta(21); gPad->SetPhi(-90);
   gH2->Draw("surf1cyl"); pl.DrawPaveLabel(x1,y1,x2+0.05,y2,"SURF1CYL","brNDC");
   
   EndTest("Lego options");
}


//______________________________________________________________________________
void options2d5()
{
   // 5th 2D options Test 
   
   StartTest(800,600);

   TPaveLabel pl;
   Float_t x1=0.67, y1=0.875, x2=0.85, y2=0.95;
   gC->Divide(2,2);
   gC->SetFillColor(17);
   gC->cd(1);
   gH2->Draw("surf1");   pl.DrawPaveLabel(x1,y1,x2,y2,"SURF1","brNDC");
   gC->cd(2);
   gH2->Draw("surf2z");  pl.DrawPaveLabel(x1,y1,x2,y2,"SURF2Z","brNDC");
   gC->cd(3);
   gH2->Draw("surf3");   pl.DrawPaveLabel(x1,y1,x2,y2,"SURF3","brNDC");
   gC->cd(4);
   gH2->Draw("surf4");   pl.DrawPaveLabel(x1,y1,x2,y2,"SURF4","brNDC");

   EndTest("Surface options");
   delete gH2;
}


//______________________________________________________________________________
void earth()
{
   // 5th 2D options Test 
   
   StartTest(1000,800);
   
   gStyle->SetPalette(1);
   gStyle->SetOptTitle(1);
   gStyle->SetOptStat(0);
   gC->Divide(2,2);
   TH2F *h1 = new TH2F("h1","Aitoff",    50, -180, 180, 50, -89.5, 89.5);
   TH2F *h2 = new TH2F("h2","Mercator",  50, -180, 180, 50, -80.5, 80.5);
   TH2F *h3 = new TH2F("h3","Sinusoidal",50, -180, 180, 50, -90.5, 90.5);
   TH2F *h4 = new TH2F("h4","Parabolic", 50, -180, 180, 50, -90.5, 90.5);
   ifstream in;
   in.open("../tutorials/earth.dat");
   Float_t x,y;
   while (1) {
     in >> x >> y;
     if (!in.good()) break;
     h1->Fill(x,y, 1);
     h2->Fill(x,y, 1);
     h3->Fill(x,y, 1);
     h4->Fill(x,y, 1);
   }
   in.close();
   gC->cd(1); h1->Draw("z aitoff");
   gC->cd(2); h2->Draw("z mercator");
   gC->cd(3); h3->Draw("z sinusoidal");
   gC->cd(4); h4->Draw("z parabolic");

   EndTest("Special contour options (AITOFF etc.)");
   delete h1;
   delete h2;
   delete h3;
   delete h4;
}


//______________________________________________________________________________
void tgraph2d1()
{
   // 1st TGraph2D Test 
   
   StartTest(600,600);

   Double_t P = 5.;
   Int_t npx  = 20 ;
   Int_t npy  = 20 ;
   Double_t x = -P;
   Double_t y = -P;
   Double_t z;
   Int_t k = 0;
   Double_t dx = (2*P)/npx;
   Double_t dy = (2*P)/npy;
   TGraph2D *dt = new TGraph2D(npx*npy);
   dt->SetNpy(41);
   dt->SetNpx(40);
   for (Int_t i=0; i<npx; i++) {
      for (Int_t j=0; j<npy; j++) {
         z = sin(sqrt(x*x+y*y))+1;
         dt->SetPoint(k,x,y,z);
         k++;
         y = y+dy;
      }
       x = x+dx;
       y = -P;
   }
   gStyle->SetPalette(1);
   dt->SetFillColor(0);
   dt->SetLineColor(1);
   dt->SetMarkerSize(1);
   dt->Draw("tri2p0Z  ");

   EndTest("TGraph2D 1 (TRI2 and P0)");
   delete dt;
}


//______________________________________________________________________________
void tgraph2d2()
{
   // 2nd TGraph2D Test 
   
   StartTest(600,600);

   gStyle->SetPadBorderMode(0);
   gStyle->SetFrameBorderMode(0);
   gStyle->SetCanvasBorderMode(0);
   Double_t Px = 6.;
   Double_t Py = 6.;
   Int_t np    = 1000;
   Double_t *rx=0, *ry=0, *rz=0;
   rx = new Double_t[np];
   ry = new Double_t[np];
   rz = new Double_t[np];
   TRandom *r = new TRandom();
   for (Int_t N=0; N<np; N++) {
      rx[N]=2*Px*(r->Rndm(N))-Px;
      ry[N]=2*Py*(r->Rndm(N))-Py;
      rz[N]=sin(sqrt(rx[N]*rx[N]+ry[N]*ry[N]))+1;
   }
   gStyle->SetPalette(1);
   TGraph2D *dt = new TGraph2D( np, rx, ry, rz);
   dt->SetFillColor(0);
   dt->SetMarkerStyle(20);
   dt->Draw("PCOL");
   
   EndTest("TGraph2D 2 (COL and P)");
   delete dt;
}


//______________________________________________________________________________
void tgraph2d3()
{
   // 3rd TGraph2D Test 
   
   StartTest(600,600);

   gStyle->SetPadBorderMode(0);
   gStyle->SetFrameBorderMode(0);
   gStyle->SetCanvasBorderMode(0);
   Double_t Px = 6.;
   Double_t Py = 6.;
   Int_t np    = 200;
   Double_t *rx=0, *ry=0, *rz=0;
   rx = new Double_t[np];
   ry = new Double_t[np];
   rz = new Double_t[np];
   TRandom *r = new TRandom();
   for (Int_t N=0; N<np; N++) {
      rx[N]=2*Px*(r->Rndm(N))-Px;
      ry[N]=2*Py*(r->Rndm(N))-Py;
      rz[N]=sin(sqrt(rx[N]*rx[N]+ry[N]*ry[N]))+1;
   }
   gStyle->SetPalette(1);
   TGraph2D *dt = new TGraph2D( np, rx, ry, rz);
   dt->SetFillColor(0);
   dt->Draw("CONT5  ");

   EndTest("TGraph2D 3 (CONT5)");
   delete dt;
}


//______________________________________________________________________________
void ntuple1()
{
   // 1st complex drawing and TPad test
   
   StartTest(700,780);

   TPad *pad1 = new TPad("pad1","This is pad1",0.02,0.52,0.48,0.98,21);
   TPad *pad2 = new TPad("pad2","This is pad2",0.52,0.52,0.98,0.98,21);
   TPad *pad3 = new TPad("pad3","This is pad3",0.02,0.02,0.48,0.48,21);
   TPad *pad4 = new TPad("pad4","This is pad4",0.52,0.02,0.98,0.48,1);
   pad1->Draw();
   pad2->Draw();
   pad3->Draw();
   pad4->Draw();
   gStyle->SetStatW(0.30);
   gStyle->SetStatH(0.20);
   gStyle->SetStatColor(42);
   pad1->cd();
   pad1->SetGrid();
   pad1->SetLogy();
   pad1->GetFrame()->SetFillColor(15);
   TNtuple *ntuple = (TNtuple*)gFile->Get("ntuple");
   ntuple->SetLineColor(1);
   ntuple->SetFillStyle(1001);
   ntuple->SetFillColor(45);
   ntuple->Draw("3*px+2","px**2+py**2>1");
   ntuple->SetFillColor(38);
   ntuple->Draw("2*px+2","pz>2","same");
   ntuple->SetFillColor(5);
   ntuple->Draw("1.3*px+2","(px^2+py^2>4) && py>0","same");
   pad1->RedrawAxis();
   pad2->cd();
   pad2->SetGrid();
   pad2->GetFrame()->SetFillColor(32);
   ntuple->Draw("pz:px>>hprofs","","goffprofs");
   TProfile *hprofs = (TProfile*)gDirectory->Get("hprofs");
   hprofs->SetMarkerColor(5);
   hprofs->SetMarkerSize(0.7);
   hprofs->SetMarkerStyle(21);
   hprofs->Fit("pol2","q");
   TF1 *fpol2 = hprofs->GetFunction("pol2");
   fpol2->SetLineWidth(4);
   fpol2->SetLineColor(2);
   pad3->cd();
   pad3->GetFrame()->SetFillColor(38);
   pad3->GetFrame()->SetBorderSize(8);
   ntuple->SetMarkerColor(1);
   ntuple->Draw("py:px","pz>1");
   ntuple->SetMarkerColor(2);
   ntuple->Draw("py:px","pz<1","same");
   pad4->cd();
   ntuple->Draw("pz:py:px","(pz<10 && pz>6)+(pz<4 && pz>3)");
   ntuple->SetMarkerColor(4);
   ntuple->Draw("pz:py:px","pz<6 && pz>4","same");
   ntuple->SetMarkerColor(5);
   ntuple->Draw("pz:py:px","pz<4 && pz>3","same");
   TPaveText *l4 = new TPaveText(-0.9,0.5,0.9,0.95);
   l4->SetFillColor(42);
   l4->SetTextAlign(12);
   l4->AddText("You can interactively rotate this view in 2 ways:");
   l4->AddText("  - With the RotateCube in clicking in this pad");
   l4->AddText("  - Selecting View with x3d in the View menu");
   l4->Draw();
   gStyle->SetStatColor(19);

   EndTest("Ntuple drawing and TPad");
}


//______________________________________________________________________________
void quarks()
{
   // 2nd complex drawing and TPad test
   
   StartTest(630,760);

   gC->SetFillColor(kBlack);
   Int_t quarkColor  = 50;
   Int_t leptonColor = 16;
   Int_t forceColor  = 38;
   Int_t titleColor  = kYellow;
   Int_t border = 8;
   TLatex *texf = new TLatex(0.90,0.455,"Force Carriers");
   texf->SetTextColor(forceColor);
   texf->SetTextAlign(22); texf->SetTextSize(0.07); texf->SetTextAngle(90);
   texf->Draw();
   TLatex *texl = new TLatex(0.11,0.288,"Leptons");
   texl->SetTextColor(leptonColor);
   texl->SetTextAlign(22); texl->SetTextSize(0.07); texl->SetTextAngle(90);
   texl->Draw();
   TLatex *texq = new TLatex(0.11,0.624,"Quarks");
   texq->SetTextColor(quarkColor);
   texq->SetTextAlign(22); texq->SetTextSize(0.07); texq->SetTextAngle(90);
   texq->Draw();
   TLatex tex(0.5,0.5,"u");
   tex.SetTextColor(titleColor); tex.SetTextFont(32); tex.SetTextAlign(22);
   tex.SetTextSize(0.14); tex.DrawLatex(0.5,0.93,"Elementary");
   tex.SetTextSize(0.12); tex.DrawLatex(0.5,0.84,"Particles");
   tex.SetTextSize(0.05); tex.DrawLatex(0.5,0.067,"Three Generations of Matter");
   tex.SetTextColor(kBlack); tex.SetTextSize(0.8);
   TPad *pad = new TPad("pad", "pad",0.15,0.11,0.85,0.79);
   pad->Draw();
   pad->cd();
   pad->Divide(4,4,0.0003,0.0003);
   pad->cd(1); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"u");
   pad->cd(2); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"c");
   pad->cd(3); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"t");
   pad->cd(4); gPad->SetFillColor(forceColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.55,"#gamma");
   pad->cd(5); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"d");
   pad->cd(6); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"s");
   pad->cd(7); gPad->SetFillColor(quarkColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"b");
   pad->cd(8); gPad->SetFillColor(forceColor);   gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.55,"g");
   pad->cd(9); gPad->SetFillColor(leptonColor);  gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{e}");
   pad->cd(10); gPad->SetFillColor(leptonColor); gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{#mu}");
   pad->cd(11); gPad->SetFillColor(leptonColor); gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#nu_{#tau}");
   pad->cd(12); gPad->SetFillColor(forceColor);  gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"Z");
   pad->cd(13); gPad->SetFillColor(leptonColor); gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"e");
   pad->cd(14); gPad->SetFillColor(leptonColor); gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.56,"#mu");
   pad->cd(15); gPad->SetFillColor(leptonColor); gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"#tau");
   pad->cd(16); gPad->SetFillColor(forceColor);  gPad->SetBorderSize(border);
   tex.DrawLatex(.5,.5,"W");
   gC->cd();

   EndTest("Divided pads and TLatex");
}


//______________________________________________________________________________
void timage()
{
   // TImage test
   
   StartTest(800,800);

   TImage *img = TImage::Open("../tutorials/rose512.jpg");
   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }
   TImage *i1 = TImage::Open("../tutorials/rose512.jpg");
   i1->SetConstRatio(kFALSE);
   i1->Flip(90);
   TImage *i2 = TImage::Open("../tutorials/rose512.jpg");
   i2->SetConstRatio(kFALSE);
   i2->Flip(180);
   TImage *i3 = TImage::Open("../tutorials/rose512.jpg");
   i3->SetConstRatio(kFALSE);
   i3->Flip(270);
   TImage *i4 = TImage::Open("../tutorials/rose512.jpg");
   i4->SetConstRatio(kFALSE);
   i4->Mirror(kTRUE);
   float d = 0.40;
   TPad *p1 = new TPad("i1", "i1", 0.05, 0.55, 0.05+d*i1->GetWidth()/i1->GetHeight(), 0.95);
   TPad *p2 = new TPad("i2", "i2", 0.55, 0.55, 0.95, 0.55+d*i2->GetHeight()/i2->GetWidth());
   TPad *p3 = new TPad("i3", "i3", 0.55, 0.05, 0.55+d*i3->GetWidth()/i3->GetHeight(), 0.45);
   TPad *p4 = new TPad("i4", "i4", 0.05, 0.05, 0.45, 0.05+d*i4->GetHeight()/i4->GetWidth());
   p1->Draw();
   p1->cd();
   i1->Draw();
   gC->cd();
   p2->Draw();
   p2->cd();
   i2->Draw();
   gC->cd();
   p3->Draw();
   p3->cd();
   i3->Draw();
   gC->cd();
   p4->Draw();
   p4->cd();
   i4->Draw();
   gC->cd();

   EndTest("TImage");
}


//______________________________________________________________________________
void cleanup()
{
}
