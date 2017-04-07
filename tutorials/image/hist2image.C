/// \file
/// \ingroup tutorial_image
/// \notebook
/// Create an image from a 2-D histogram and manipulate it.
///
/// \image html image_hist2image.png
/// \macro_code
///
/// \author Valeriy Onuchin

#include <TAttImage.h>

void hist2image()
{

   TCanvas *canv = new TCanvas("image", "xygaus + xygaus(5) + xylandau(10)");
   canv->ToggleEventStatus();
   canv->SetRightMargin(0.2);
   canv->SetLeftMargin(0.01);
   canv->SetTopMargin(0.01);
   canv->SetBottomMargin(0.01);

   // histogram as image (hist taken from draw2dopt.C)
   TImage *img = TImage::Create();

   TF2 *f2 = new TF2("f2","(xygaus + xygaus(5) + xylandau(10))",-4,4,-4,4);
   Double_t params[] = {130,-1.4,1.8,1.5,1, 150,2,0.5,-2,0.5, 3600,-2,0.7,-3,0.3};
   f2->SetParameters(params);
   TH2D *h2 = new TH2D("h2","xygaus + xygaus(5) + xylandau(10)",100,-4,4,100,-4,4);
   h2->FillRandom("f2",40000);
   img->SetImage((const Double_t *)h2->GetArray(), h2->GetNbinsX() + 2,
                  h2->GetNbinsY() + 2, gHistImagePalette);
   img->Draw();
   img->StartPaletteEditor();
}
