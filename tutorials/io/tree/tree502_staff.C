/// \file
/// \ingroup tutorial_tree
/// \notebook -nodraw
/// Create a plot of the data in `cernstaff.root`
/// To create `cernstaff.root`, execute tutorial `$ROOTSYS/tutorials/io/tree/tree500_cernbuild.C`
///
/// \macro_code
///
/// \author Rene Brun

void tree502_staff()
{
   auto f = TFile::Open("cernstaff.root");
   auto T = f->Get<TTree>("T");
   T->Draw("Grade:Age:Cost:Division:Nation", "", "gl5d");
   if (gPad)
      gPad->Print("staff.C.png");
}
