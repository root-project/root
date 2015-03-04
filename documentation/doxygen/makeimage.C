// Generates the png output of the macro "macroname" located in "dirname"

void makeimage(const char *macroname, const char *classname, int id)
{
   gROOT->ProcessLine(Form(".x %s",macroname));

   TIter iCanvas(gROOT->GetListOfCanvases());
   TVirtualPad* pad = 0;

   while ((pad = (TVirtualPad*) iCanvas())) {
      pad->SaveAs(TString::Format("%s_%3.3d.png", classname,id));
   }
}
