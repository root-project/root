/// \file
/// \ingroup tutorial_hist
/// Echo object at mouse position and show a graphics line.
/// Example of macro called when a mouse event occurs in a pad.
///
/// Example:
/// ~~~{.cpp}
/// Root > TFile f("hsimple.root");
/// Root > hpxpy.Draw();
/// Root > c1.AddExec("ex2",".x exec2.C");
/// ~~~
/// When moving the mouse in the canvas, a second canvas shows the
/// projection along X of the bin corresponding to the Y position
/// of the mouse. The resulting histogram is fitted with a gaussian.
/// A "dynamic" line shows the current bin position in Y.
/// This more elaborated example can be used as a starting point
/// to develop more powerful interactive applications exploiting CINT
/// as a development engine.
///
/// \macro_code
///
/// \author Rene Brun

void exec2()
{

   if (!gPad) {
      Error("exec2", "gPad is null, you are not supposed to run this macro");
      return;
   }


   TObject *select = gPad->GetSelected();
   if(!select) return;
   if (!select->InheritsFrom(TH2::Class())) {gPad->SetUniqueID(0); return;}
   gPad->GetCanvas()->FeedbackMode(kTRUE);

   //erase old position and draw a line at current position
   int pyold = gPad->GetUniqueID();
   int px = gPad->GetEventX();
   int py = gPad->GetEventY();
   float uxmin = gPad->GetUxmin();
   float uxmax = gPad->GetUxmax();
   int pxmin = gPad->XtoAbsPixel(uxmin);
   int pxmax = gPad->XtoAbsPixel(uxmax);
   if(pyold) gVirtualX->DrawLine(pxmin,pyold,pxmax,pyold);
   gVirtualX->DrawLine(pxmin,py,pxmax,py);
   gPad->SetUniqueID(py);
   Float_t upy = gPad->AbsPixeltoY(py);
   Float_t y = gPad->PadtoY(upy);

   //create or set the new canvas c2
   TVirtualPad *padsav = gPad;
   TCanvas *c2 = (TCanvas*)gROOT->GetListOfCanvases()->FindObject("c2");
   if(c2) delete c2->GetPrimitive("Projection");
   else   c2 = new TCanvas("c2");
   c2->cd();

   //draw slice corresponding to mouse position
   TH2 *h = (TH2*)select;
   Int_t biny = h->GetYaxis()->FindBin(y);
   TH1D *hp = h->ProjectionX("",biny,biny);
   char title[80];
   sprintf(title,"Projection of biny=%d",biny);
   hp->SetName("Projection");
   hp->SetTitle(title);
   hp->Fit("gaus","ql");
   c2->Update();
   padsav->cd();
}
