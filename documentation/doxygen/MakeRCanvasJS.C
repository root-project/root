/// Generates the json file output of the macro MacroName

#include "ROOT/RCanvas.hxx"

void MakeRCanvasJS(const char *MacroName, const char *IN, const char *OutDir, bool cp, bool py)
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

   std::string json_str;
   {
       std::ifstream fjson(json_file.Data());
      json_str = std::string((std::istreambuf_iterator<char>(fjson)), std::istreambuf_iterator<char>());
   }
   gSystem->Unlink(json_file.Data());

   // Build the html file inlining the json picture
   FILE *fh = fopen(TString::Format("%s/macros/%s.html",OutDir,IN), "w");
   fprintf(fh,"<div id=\"draw_json_%s\" style=\"width:700px; height:500px\"></div>\n", IN);
   fprintf(fh,"<script type=\"text/javascript\">\n");
   fprintf(fh,"   function load_jsroot_%s() {\n", IN);
   fprintf(fh,"      return new Promise(resolveFunc => {\n");
   fprintf(fh,"         if (typeof JSROOT != 'undefined') return resolveFunc(true);\n");
   fprintf(fh,"         let script = document.createElement('script');\n");
   fprintf(fh,"         script.src = 'https://root.cern/js/dev/scripts/JSRoot.core.min.js';\n");
   fprintf(fh,"         script.onload = resolveFunc;\n");
   fprintf(fh,"         document.head.appendChild(script);\n");
   fprintf(fh,"      });\n");
   fprintf(fh,"   }\n");
   fprintf(fh,"   load_jsroot_%s().then(() => { \n", IN);
   fprintf(fh,"      JSROOT.settings.HandleKeys = false;\n");
   fprintf(fh,"      let obj = JSROOT.parse(%s);\n", json_str.c_str());
   fprintf(fh,"      JSROOT.draw('draw_json_%s', obj);\n", IN);
   fprintf(fh,"   });\n");
   fprintf(fh,"</script>\n");
   fclose(fh);
}
