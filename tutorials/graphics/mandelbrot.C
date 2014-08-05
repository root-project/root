#include <TStyle.h>
#include <TROOT.h>
#include <TH2.h>
#include <TComplex.h>
#include <TVirtualPad.h>
#include <TCanvas.h>

//==================================================================
//
// Using TExec to handle keyboard events and TComplex to draw the Mandelbrot set.
// Author : Luigi Bardelli [ bardelli@fi.infn.it ]
//
// Pressing the keys 'z' and 'u' will zoom and unzoom the picture
// near the mouse location, 'r' will reset to the default view.
//
// Try it (in compiled mode!) with:   root mandelbrot.C+
//
// Details:
//    when a mouse event occurs the myexec() function is called (by
//    using AddExec). Depending on the pressed key, the mygenerate()
//    function is called, with the proper arguments. Note the
//    last_x and last_y variables that are used in myexec() to store
//    the last pointer coordinates (px is not a pointer position in
//    kKeyPress events).
//==================================================================

TH2F *last_histo=NULL;

void mygenerate(double factor, double cen_x, double cen_y)
{
  printf("Regenerating...\n");
  // resize histo:
  if(factor>0)
    {
      double dx=last_histo->GetXaxis()->GetXmax()-last_histo->GetXaxis()->GetXmin();
      double dy=last_histo->GetYaxis()->GetXmax()-last_histo->GetYaxis()->GetXmin();
      last_histo->SetBins(
                          last_histo->GetNbinsX(),
                          cen_x-factor*dx/2,
                          cen_x+factor*dx/2,
                          last_histo->GetNbinsY(),
                          cen_y-factor*dy/2,
                          cen_y+factor*dy/2
                          );
      last_histo->Reset();
    }
  else
    {
      if(last_histo!=NULL) delete last_histo;
      // allocate first view...
      last_histo= new TH2F("h2",
         "Mandelbrot [move mouse and  press z to zoom, u to unzoom, r to reset]",
                           200,-2,2,200,-2,2);
      last_histo->SetStats(0);
    }
  const int max_iter=50;
  for(int bx=1;bx<=last_histo->GetNbinsX();bx++)
    for(int by=1;by<=last_histo->GetNbinsY();by++)
      {
         double x=last_histo->GetXaxis()->GetBinCenter(bx);
         double y=last_histo->GetYaxis()->GetBinCenter(by);
         TComplex point( x,y);
         TComplex z=point;
         int iter=0;
         while (z.Rho()<2){
            z=z*z+point;
            last_histo->Fill(x,y);
            iter++;
            if(iter>max_iter) break;
         }
      }
  last_histo->Draw("colz");
  gPad->Modified();
  gPad->Update();
  printf("Done.\n");
}

void myexec()
{
  // get event information
  int event = gPad->GetEvent();
  int px = gPad->GetEventX();
  int py = gPad->GetEventY();

  // some magic to get the coordinates...
  double xd = gPad->AbsPixeltoX(px);
  double yd = gPad->AbsPixeltoY(py);
  float x = gPad->PadtoX(xd);
  float y = gPad->PadtoY(yd);

  static float last_x;
  static float last_y;

  if(event!=kKeyPress)
    {
      last_x=x;
      last_y=y;
      return;
    }

  const double Z=2.;
  switch(px){
  case 'z': // ZOOM
    mygenerate(1./Z, last_x, last_y);
    break;
  case 'u': // UNZOOM
    mygenerate(Z   , last_x, last_y);
    break;
  case 'r': // RESET
    mygenerate(-1   , last_x, last_y);
    break;
  };
}

void mandelbrot()
{
  // cosmetics...
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1,0);
  gStyle->SetPadGridX(kTRUE);
  gStyle->SetPadGridY(kTRUE);
  new TCanvas("canvas","View Mandelbrot set");
  gPad->SetCrosshair();
  // this generates and draws the first view...
  mygenerate(-1,0,0);

  // add exec
  gPad->AddExec("myexec","myexec()");
}
