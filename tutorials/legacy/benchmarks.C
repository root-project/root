/// \file
/// \ingroup tutorial_legacy
/// This macro run several tests and produces an benchmark report.
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Rene Brun

TCanvas* bench1 = 0;

void bexec(TString &dir,const char *macro)
{
   if (gROOT->IsBatch()) printf("Processing benchmark: %s%s\n",dir.Data(),macro);
   TPaveText *summary = (TPaveText*)bench1->GetPrimitive("TPave");
   TText *tmacro = summary->GetLineWith(macro);
   if (tmacro) tmacro->SetTextColor(4);
   bench1->Modified(); bench1->Update();

   gROOT->Macro(Form("%s%s",dir.Data(),macro));

   TPaveText *summary2 = (TPaveText*)bench1->GetPrimitive("TPave");
   TText *tmacro2 = summary2->GetLineWith(macro);
   if (tmacro2) tmacro2->SetTextColor(2);
   bench1->Modified(); bench1->Update(); gSystem->ProcessEvents();

}

void benchmarks() {
   TString dir = gSystem->UnixPathName(__FILE__);
   dir.ReplaceAll("benchmarks.C","");
   dir.ReplaceAll("/./","/");
   bench1 = new TCanvas("bench1","Benchmarks Summary",-1000,50,200,500);
   TPaveText *summary = new TPaveText(0,0,1,1);
   summary->SetTextAlign(12);
   summary->SetTextSize(0.08);
   summary->Draw();
   summary->AddText("  graphics/framework.C");
   summary->AddText("  hsimple.C");
   summary->AddText("  hist/hsum.C");
   summary->AddText("  graphics/formula1.C");
   summary->AddText("  hist/fillrandom.C");
   summary->AddText("  fit/fit1.C");
   summary->AddText("  hist/h1draw.C");
   summary->AddText("  graphs/graph.C");
   summary->AddText("  graphs/gerrors.C");
   summary->AddText("  graphics/tornado.C");
   summary->AddText("  graphs/surfaces.C");
   summary->AddText("  graphs/zdemo.C");
   summary->AddText("  geom/geometry.C");
   summary->AddText("  geom/na49view.C");
   summary->AddText("  tree/ntuple1.C");
   summary->AddText("  ");
   bexec(dir,"graphics/framework.C");
   bexec(dir,"hsimple.C");
   bexec(dir,"hist/hsum.C");
   bexec(dir,"graphics/formula1.C");
   bexec(dir,"hist/fillrandom.C");
   bexec(dir,"fit/fit1.C");
   bexec(dir,"hist/h1draw.C");
   bexec(dir,"graphs/graph.C");
   bexec(dir,"graphs/gerrors.C");
   bexec(dir,"graphics/tornado.C");
   bexec(dir,"graphs/surfaces.C");
   bexec(dir,"graphs/zdemo.C");
   bexec(dir,"geom/geometry.C");
   bexec(dir,"geom/na49view.C");
   bexec(dir,"tree/ntuple1.C");
   bexec(dir,"rootmarks.C");
}
