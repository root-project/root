{
fprintf(stdout,"loading libraries and script needed for the new TTree::Draw\n");
gROOT->ProcessLine(".L TProxy.cxx+");
gSystem->Load("libTreePlayer");
gROOT->ProcessLine(".L GenerateProxy.C+");
#include <string>
fprintf(stdout,"You can call draw(tree,filename) to try the new TTree::Draw\n");
gInterpreter->SaveContext();
}
