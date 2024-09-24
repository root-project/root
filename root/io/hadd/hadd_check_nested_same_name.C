// https://github.com/root-project/root/issues/16190
#include "TFileMerger.h"
#include "TFile.h"
#include "TH1D.h"
#include <iostream>

void createSrcFile(const char* name)
{
  TFile f(name, "RECREATE");
  f.mkdir("A", "");
  f.mkdir("B/A", "");
  f.mkdir("C/D/A", "");
  f.cd("A");
  TH1D h1("H1", "H1", 1, 0., 1.);
  h1.SetBinContent(1, 1);
  h1.Write();
  f.cd("B/A");
  TH1D h2("H2", "H2", 1, 0., 1.);
  h2.SetBinContent(1, 10);
  h2.Write();
  f.cd("C/D/A");
  TH1D h3("H3", "H3", 1, 0., 1.);
  h3.SetBinContent(1, 100);
  h3.Write();
  f.Close();
}


void check(const char* filename, const char* deco)
{
  TFile f(filename, "READ");
  std::cout << deco
	    << f.Get<TH1D>("A/H1")->GetBinContent(1)
	    << " "
            << f.Get<TH1D>("B/A/H2")->GetBinContent(1)
	    << " "
	    << f.Get<TH1D>("C/D/A/H3")->GetBinContent(1)
	    << std::endl;
  f.Close();
}


void hadd_check_nested_same_name()
{
  createSrcFile("src1.root");
  createSrcFile("src2.root");
  TFileMerger mg;
  mg.AddFile("src1.root", kFALSE);
  mg.AddFile("src2.root", kFALSE);
  mg.OutputFile("dest.root", kTRUE);
  mg.Merge();
  check("src1.root", "  ");
  check("src2.root", "+ ");
  std::cout << "------------\n";
  check("dest.root", "= ");
}
