// Generates the ImageName output of the macro MacroName

void makeimage(const char *MacroName, const char *ImageName, const char *OutDir)
{
   gROOT->ProcessLine(Form(".x %s",MacroName));

   TIter iCanvas(gROOT->GetListOfCanvases());
   TVirtualPad* pad = 0;

   while ((pad = (TVirtualPad*) iCanvas())) {
      pad->SaveAs(TString::Format("%s/html/%s",OutDir,ImageName));
   }
}
