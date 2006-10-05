

void runmapvector(const char  *filename = "mapvector.root") 
{
   TFile * f = TFile::Open(filename,"READ");
   TTree *tree; f->GetObject("T",tree);
   tree->Scan("data.first");
   tree->Scan("data.second");
}
