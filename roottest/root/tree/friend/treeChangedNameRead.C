#include "TROOT.h"

// https://its.cern.ch/jira/browse/ROOT-7743
void treeChangedNameRead()
{
   gROOT->ProcessLine("auto _file0 = TFile::Open(\"ftest7743.root\");");
   gROOT->ProcessLine(".ls");
   gROOT->ProcessLine("treeW->AddFriend(\"tree1\");");
   gROOT->ProcessLine("treeW->GetListOfFriends()->Print();");
   gROOT->ProcessLine("treeW->Draw(\"tree1.x\");");
   gROOT->ProcessLine("htemp->Integral()");
}
