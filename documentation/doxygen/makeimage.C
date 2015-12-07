/// Generates the ImageName output of the macro MacroName

#include <stdio.h>

void makeimage(const char *MacroName, const char *ImageName, const char *OutDir, bool cp)
{
   gROOT->ProcessLine(Form(".x %s",MacroName));
   if (cp) gSystem->Exec(TString::Format("cp %s %s/macros", MacroName, OutDir));

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
