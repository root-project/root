// Generates the ImageName output of the macro MacroName

void makeimage(const char *MacroName, const char *ImageName)
{
   gROOT->ProcessLine(Form(".x %s",MacroName));

   TIter iCanvas(gROOT->GetListOfCanvases());
   TVirtualPad* pad = 0;

   while ((pad = (TVirtualPad*) iCanvas())) {
      pad->SaveAs(ImageName);
   }
}
