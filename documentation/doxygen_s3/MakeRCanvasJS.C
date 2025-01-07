/// Generates the json file output of the macro MacroName

#include "ROOT/RCanvas.hxx"
#include "TString.h"
#include "TSystem.h"
#include "TROOT.h"
#include <fstream>

void MakeRCanvasJS(const char *MacroName, const char *IN, const char *OutDir, bool cp, bool py)
{
   using namespace ROOT::Experimental;

   // Execute the macro as a C++ one or a Python one.
   if (!py) gROOT->ProcessLine(TString::Format(".x %s",MacroName));
   else     gROOT->ProcessLine(TString::Format("TPython::ExecScript(\"%s\");",MacroName));

   // If needed, copy the macro in the documentation directory.
   if (cp) {
      TString MN = MacroName;
      Int_t i = MN.Index("(");
      Int_t l = MN.Length();
      if (i>0) MN.Remove(i, l);
      gSystem->Exec(TString::Format("cp %s %s/macros", MN.Data(), OutDir));
   }

   // Retrieve the RCanvas produced by the macro execution and produce the json file.
   auto canv = RCanvas::GetCanvases()[0];
   TString json_file = TString::Format("%s/html/%s.json",OutDir,IN);
   canv->SaveAs(json_file.Data());

   std::string json_str;
   {
       std::ifstream fjson(json_file.Data());
      json_str = std::string((std::istreambuf_iterator<char>(fjson)), std::istreambuf_iterator<char>());
   }
   gSystem->Unlink(json_file.Data());

   // Build the html file inlining the json picture
   FILE *fh = fopen(TString::Format("%s/macros/%s.html",OutDir,IN), "w");
   fprintf(fh,"<div id=\"draw_json_%s\" style=\"position: relative; width: 700px; height: 500px;\"></div>\n", IN);
   fprintf(fh,"<script type=\"module\">\n");
   fprintf(fh,"   import { settings, parse, draw } from './js/modules/main.mjs';\n");
   fprintf(fh,"   settings.HandleKeys = false;\n");
   fprintf(fh,"   let obj = parse(%s);\n", json_str.c_str());
   fprintf(fh,"   draw('draw_json_%s', obj);\n", IN);
   fprintf(fh,"</script>\n");
   fclose(fh);
}
