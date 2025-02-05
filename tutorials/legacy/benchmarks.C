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
   dir.ReplaceAll("legacy","");
   bench1 = new TCanvas("bench1","Benchmarks Summary",-1000,50,250,500);
   TPaveText *summary = new TPaveText(0,0,1,1);
   summary->SetTextAlign(12);
   summary->SetTextSize(0.06);
   summary->Draw();
   summary->AddText("  visualisation/graphics/framework.C");
   summary->AddText("  hsimple.C");
   summary->AddText("  hist/hist007_TH1_liveupdate.C");
   summary->AddText("  visualisation/graphics/formula1.C");
   summary->AddText("  hist/hist001_TH1_fillrandom.C");
   summary->AddText("  math/fit/fit1.C");
   summary->AddText("  hist/hist015_TH1_read_and_draw.C");
   summary->AddText("  visualisation/graphs/gr001_simple.C");
   summary->AddText("  visualisation/graphs/gr002_errors.C");
   summary->AddText("  visualisation/graphics/tornado.C");
   summary->AddText("  visualisation/graphics/surfaces.C");
   summary->AddText("  visualisation/graphs/gr303_zdemo.C");
   summary->AddText("  legacy/g3d/geometry.C");
   summary->AddText("  legacy/g3d/na49view.C");
   summary->AddText("  io/tree/tree120_ntuple.C");
   summary->AddText("  ");
   bexec(dir,"visualisation/graphics/framework.C");
   bexec(dir,"hsimple.C");
   bexec(dir,"hist/hist007_TH1_liveupdate.C");
   bexec(dir,"visualisation/graphics/formula1.C");
   bexec(dir,"hist/hist001_TH1_fillrandom.C");
   bexec(dir,"math/fit/fit1.C");
   bexec(dir,"hist/hist015_TH1_read_and_draw.C");
   bexec(dir,"visualisation/graphs/gr001_simple.C");
   bexec(dir,"visualisation/graphs/gr002_errors.C");
   bexec(dir,"visualisation/graphics/tornado.C");
   bexec(dir,"visualisation/graphics/surfaces.C");
   bexec(dir,"visualisation/graphs/gr303_zdemo.C");
   bexec(dir,"legacy/g3d/geometry.C");
   bexec(dir,"legacy/g3d/na49view.C");
   bexec(dir,"io/tree/tree120_ntuple.C");
   bexec(dir,"legacy/rootmarks.C");
}
