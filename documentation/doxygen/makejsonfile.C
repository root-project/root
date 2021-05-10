/// Generates the json file output of the macro MacroName

#include "ROOT/RCanvas.hxx"
#include "ROOT/RColor.hxx"
#include "ROOT/RText.hxx"
#include "ROOT/RPadPos.hxx"

void makejsonfile(const char *MacroName, const char *IN, const char *OutDir, bool cp, bool py)
{
   using namespace ROOT::Experimental;

   // Execute the macro as a C++ one or a Python one.
   if (!py) gROOT->ProcessLine(Form(".x %s",MacroName));
   else     gROOT->ProcessLine(Form("TPython::ExecScript(\"%s\");",MacroName));

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

   // Build the html file inlining the json picture
   FILE *fh = fopen(TString::Format("%s/macros/%s.html",OutDir,IN), "w");
   fprintf(fh,"<script src=\"https://root.cern/js/dev/scripts/JSRoot.core.min.js\" type=\"text/javascript\"></script>\n");
   fprintf(fh,"<div id=\"draw_json_%s\" style=\"width:700px; height:500px\"></div>\n", IN);
   fprintf(fh,"<script type='text/javascript'>JSROOT.httpRequest(\"./%s.json\",\"object\")",IN);
   fprintf(fh,".then(obj => JSROOT.draw(\"draw_json_%s\", obj))", IN);
   fprintf(fh,";</script>\n");
   fclose(fh);
}
