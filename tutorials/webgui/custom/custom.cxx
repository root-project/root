/// \file
/// \ingroup tutorial_webgui
/// This is example how custom user class can be used in the TWebCanvas
/// Custom class is just triangle which drawn on the frame with NDC coordinates
/// `custom.mjs` provides JavaScript code for object painting and interactivity
/// It is also possible to use such "simple" class without loading of custom JS code,
/// but then it requires appropriate Paint() method and will miss interactivity in browser
///
/// This macro must be executed with ACLiC as follows:
///
/// ~~~{.cpp}
/// .x custom.cxx+
/// ~~~
///
/// \macro_code
///
/// \author Sergey Linev

#include <iostream>

#include "TNamed.h"
#include "TAttLine.h"
#include "TAttFill.h"
#include "TWebCanvas.h"
#include "TCanvas.h"
#include "TROOT.h"

#include "ROOT/RWebWindowsManager.hxx"

class TTriangle : public TNamed, public TAttLine, public TAttFill {
   double fX[3] = { 0, 0, 0 };
   double fY[3] = { 0, 0, 0 };

   public:
      TTriangle()  {} // =default not allowed !!!

      TTriangle(const char *name, const char *title = "") : TNamed(name, title) {}

      void SetPoints(double x1, double y1, double x2, double y2, double x3, double y3)
      {
         fX[0] = x1;
         fY[0] = y1;
         fX[1] = x2;
         fY[1] = y2;
         fX[2] = x3;
         fY[2] = y3;
      }

      /** In old graphics just provide line drawing */
      void Paint(Option_t *opt) override
      {
         if (!gPad) return;
         TAttLine::Modify();  //Change line attributes only if necessary
         TAttFill::Modify();  //Change fill area attributes only if necessary

         if (*opt == 'f')
            gPad->PaintFillAreaNDC(3, fX, fY, opt);
         else
            gPad->PaintPolyLineNDC(3, fX, fY, opt);
      }

   ClassDefOverride(TTriangle, 1);   // Example of triangle drawing in web canvas
};

void custom(bool ignore_jsmodule = false)
{
   if (ignore_jsmodule) {
      printf("Custom JS module will NOT be provided for TTriangle class\n");
      printf("No interactive features will be available\n");
      printf("TTriangle::Paint() method will be used for object painting - also in web case\n");
   } else {
      #ifdef __CLING__
         printf("Please run this script in compiled mode by running \".x custom.cxx+\"\n");
         printf("Requires to properly generate dictionary for TTriangle class\n");
         return;
      #endif

      std::string fdir = __FILE__;
      auto pos = fdir.find("custom.cxx");
      if (pos > 0)
         fdir.resize(pos);
      else
         fdir = gROOT->GetTutorialsDir() + std::string("/webgui/custom/");

      // location required to load files
      // also it is name of modules path used in importmap
      ROOT::RWebWindowsManager::AddServerLocation("triangle", fdir);

      // mark TTriangle as supported on the client
      TWebCanvas::AddCustomClass("TTriangle");

      // specify which extra module should be loaded,
      // "triangle/" is registered path from server locations
      TWebCanvas::SetCustomScripts("modules:triangle/custom.mjs");
   }

   auto tr1 = new TTriangle("tr1", "first triangle");
   tr1->SetPoints(0.4, 0.5, 0.8, 0.3, 0.8, 0.7);
   tr1->SetLineColor(kRed);
   tr1->SetLineWidth(3);
   tr1->SetFillColor(kBlue);

   auto tr2 = new TTriangle("tr2", "second triangle");
   tr2->SetPoints(0.2, 0.2, 0.2, 0.6, 0.5, 0.4);
   tr2->SetLineColor(kGreen);
   tr2->SetLineStyle(2);
   tr2->SetFillColor(kYellow);

   auto tr3 = new TTriangle("tr3", "third triangle");
   tr3->SetPoints(0.4, 0.8, 0.6, 0.8, 0.5, 0.2);
   tr3->SetLineColor(kCyan);
   tr3->SetFillColor(kMagenta);

   auto c1 = new TCanvas("c1", "Triangles");
   c1->Add(tr1, "f");
   c1->Add(tr2, "f");
   c1->Add(tr3, "f");

   // test image saving with web browser, chrome or firefox are required
   // c1->Update();
   // c1->SaveAs("triangle.png");
}
