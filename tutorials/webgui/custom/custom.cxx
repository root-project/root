/// \file
/// \ingroup tutorial_webgui
///  This is example how custom user class can be used in the TWebCanvas
/// Custom class is just triangle which drawn on the frame with NDC coordinates
///  custom.mjs includes JavaScript code for painting
///  And demo shows how custom class can be registered to the TWebCanvas
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

void custom()
{
   std::string fdir = __FILE__;
   auto pos = fdir.find("custom.cxx");
   if (pos > 0)
      fdir.resize(pos);
   else
      fdir = gROOT->GetTutorialsDir() + std::string("/webgui/custom/");

   printf("fdir = %s\n", fdir.c_str());

   // location required to load files
   // also it is name of modules path used in importmap
   ROOT::RWebWindowsManager::AddServerLocation("triangle", fdir);

   // mark TTriangle as supported on the client
   TWebCanvas::AddCustomClass("TTriangle");

   // specify which extra module should be loaded,
   // "triangle/" is registered path from server locations
   TWebCanvas::SetCustomScripts("modules:triangle/custom.mjs");


   auto tr = new TTriangle("tr1", "title of triangle");
   tr->SetPoints(0.3, 0.5, 0.7, 0.3, 0.7, 0.7);

   tr->SetLineColor(kRed);
   tr->SetLineWidth(3);
   tr->SetFillColor(kBlue);

   auto c1 = new TCanvas("c1", "Triangle");
   c1->Add(tr, "f");
}
