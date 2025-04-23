#include <vector>

void runmapvector(const char  *filename = "mapvector.root") 
{
   gInterpreter->UnloadLibraryMap("mapvector_C");
   TFile * f = TFile::Open(filename,"READ");
   TTree *tree; f->GetObject("T",tree);
   tree->Scan("data.first");
   tree->Scan("data.second");
}
