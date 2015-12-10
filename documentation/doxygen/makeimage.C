/// Generates the ImageName output of the macro MacroName

#include <stdio.h>

void makeimage(const char *MacroName, const char *ImageName, const char *OutDir, bool cp, bool py)
{
   if (!py) gROOT->ProcessLine(Form(".x %s",MacroName));
   else     gROOT->ProcessLine(Form("TPython::ExecScript(\"%s\");",MacroName));
   if (cp) {
      TString MN = MacroName;
      Int_t i = MN.Index("(");
      Int_t l = MN.Length();
      if (i>0) MN.Remove(i, l);
      gSystem->Exec(TString::Format("cp %s %s/macros", MN.Data(), OutDir));
   }

   TIter iCanvas(gROOT->GetListOfCanvases());
   TVirtualPad* pad = 0;
   int ImageNum = 0;
   while ((pad = (TVirtualPad*) iCanvas())) {
      ImageNum++;
      pad->SaveAs(TString::Format("%s/html/pict%d_%s",OutDir,ImageNum,ImageName));
   }

   FILE *f = fopen("NumberOfImages.dat", "w");
   fprintf(f,"%d\n",ImageNum);
   fclose(f);
}
