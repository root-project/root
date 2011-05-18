//  Demonstrates how to access and manipulate ARGB pixel values of an image +...
//  - how to make a part of an image to be transparent.
//  - how to merge/alphablend an image with transparent colors
//    with some background image.
//Author: Valeriy Onuchin

#include "TColor.h"
#include "TImage.h"
#include "TImageDump.h"
#include "TVirtualPad.h"
#include "TROOT.h"
#include "TFrame.h"

static UInt_t color2rgb(TColor *col)
{
   // returns RGB value of color

   return ((UInt_t(col->GetRed()*255) << 16) +
           (UInt_t(col->GetGreen()*255) << 8) +
            UInt_t(col->GetBlue()*255));
}


void trans_graph()
{
   // remember if  we are in batch mode
   Bool_t batch = gROOT->IsBatch();

   // switch to batch mode
   gROOT->SetBatch(kTRUE);

   // execute graph.C macro
   gROOT->Macro("$ROOTSYS/tutorials/graphs/graph.C");

   // create gVirtualPS object
   TImageDump dmp("dummy.png");
   TImage *fore = dmp.GetImage();  // image associated with image_dump

   // resize canvas
   gPad->SetCanvasSize(400, 300);
   gPad->Paint(); // paint gPad on fore image associated with TImageDump

   // open background image
   TImage *back = TImage::Open("$ROOTSYS/tutorials/image/rose512.jpg");

   // choose colors to be transparent
   TColor *bk1 = gROOT->GetColor(gPad->GetFillColor());
   TColor *bk2 = gROOT->GetColor(gPad->GetFrame()->GetFillColor());
   UInt_t rgb1 = color2rgb(bk1);
   UInt_t rgb2 = color2rgb(bk2);

   // get directly accessible ARGB array
   UInt_t *argb = fore->GetArgbArray();
   UInt_t w = fore->GetWidth();
   UInt_t h = fore->GetHeight();

   // scan all pixels in fore image and
   // make rgb1, rgb2 colors transparent.
   for (UInt_t i = 0; i < h; i++) {
      for (UInt_t j = 0; j < w; j++) {
         Int_t idx = i*w + j;

         // RGB part of ARGB color
         UInt_t col = argb[idx] & 0xffffff;

         // 24..31 bits define transparency of the color in the range 0 - 0xff
         // for example, 0x00000000 - black color with 100% transparency
         //              0xff000000 - non-transparent black color

         if ((col == rgb1) || (col == rgb2)) { //
            argb[idx] = 0; // 100% transparent
         } else {
            argb[idx] = 0xff000000 + col;  // make other pixels non-transparent
         }
      }
   }

   // alphablend back and fore images
   back->Merge(fore, "alphablend", 20, 20);

   // write result image in PNG format
   back->WriteImage("trans_graph.png");
   printf("*************** File trans_graph.png created ***************\n");

   delete back;

   // switch back to GUI mode
   if (!batch) gROOT->SetBatch(kFALSE);
}
