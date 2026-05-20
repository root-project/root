#include <TH1.h>
#include <TFile.h>

int main()
{
   TFile file("hsimple.root", "READ");
   if (!file.IsOpen())
      return 1;

   TH1 *histo = file.Get<TH1>("hpx");
   if (!histo)
      return 2;

   if (histo->GetEntries() != 25000)
      return 3;
   histo->Print();

   return 0;
}
