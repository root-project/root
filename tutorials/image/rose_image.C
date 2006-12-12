#include "TImage.h"
#include "TCanvas.h"
#include "TArrayD.h"
#include "TROOT.h"
#include "TColor.h"
#include "TAttImage.h"
#include "TEnv.h"

TCanvas *c1;

void rose_image()
{
   // Display image in a new canvas and pad.
   //Author: Valeriy Onuchin
   
   TImage *img = TImage::Open("rose512.jpg");

   if (!img) {
      printf("Could not create an image... exit\n");
      return;
   }

   img->SetConstRatio(0);
   img->SetImageQuality(TAttImage::kImgBest);

   TString fp = gEnv->GetValue("Root.TTFontPath", "");
   TString bc = fp + "/BlackChancery.ttf";
   TString ar = fp + "/arial.ttf";

   // draw text over image with funny font
   img->DrawText(120, 160, "Hello World!", 32, 
                 gROOT->GetColor(4)->AsHexString(), 
                 bc, TImage::kShadeBelow);

   // draw text over image with foreground specified by pixmap
   img->DrawText(250, 350, "goodbye cruel world ...", 24, 0, 
                 ar, TImage::kPlain, "fore.xpm");

   TImage *img2 = TImage::Open("mditestbg.xpm");

   // tile image
   img2->Tile(img->GetWidth(), img->GetHeight());

   c1 = new TCanvas("rose512", "examples of image manipulations", 760, 900);
   c1->Divide(2, 3);
   c1->cd(1);
   img->Draw("xxx");
   img->SetEditable(kTRUE);

   c1->cd(2);
   // averaging with mditestbg.xpm image
   TImage *img3 = (TImage*)img->Clone("img3");
   img3->Merge(img2, "allanon");
   img3->Draw();

   // contrasting (tint with itself)
   c1->cd(3);
   TImage *img4 = (TImage*)img->Clone("img4");
   img4->Merge(img4, "tint");

   // draw filled rectangle with magenta color
   img4->FillRectangle("#FF00FF", 20, 220, 40, 40);

   // Render multipoint alpha-blended gradient (R->G->B)
   img4->Gradient(0, "#FF0000 #00FF00 #220000FF", 0, 50, 50, 100, 100);

   // draw semi-transparent 3D button
   img4->Bevel(300, 20, 160, 40, "#ffffffff", "#fe000000", 3, 0);
   img4->DrawLine(10, 100, 100, 10, "#0000ff", 4);
   img4->Draw();

   // vectorize image. Reduce palette to 256 colors
   c1->cd(4);
   TImage *img5 = (TImage*)img->Clone("img5");
   img5->Vectorize(256);
   img5->Draw();

   // quantization of the image
   c1->cd(5);
   TImage *img6 = (TImage*)img->Clone("img6");
   TImagePalette *pal = (TImagePalette *)&img5->GetPalette();
   TArrayD *arr = img6->GetArray(50, 40, pal);
   img6->SetImage(arr->GetArray(), 50, 40, pal);
   img6->Draw();

   // HSV adjustment (convert red to yellow)
   c1->cd(6);
   TImage *img7 = (TImage*)img->Clone("img7");
   img7->HSV(0, 40, 40);
   img7->Draw();
}
