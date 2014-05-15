// Generates the png output of the macro "macroname" located in "dirname"

void makepng(const char *macroname, const char *dirname)
{
   gROOT->ProcessLine(Form(".x %s/%s.C",dirname,macroname));

   TIter iCanvas(gROOT->GetListOfCanvases());
   TVirtualPad* pad = 0;

   while ((pad = (TVirtualPad*) iCanvas())) {
      pad->SaveAs(TString::Format("%s.png", macroname));
   }
}
