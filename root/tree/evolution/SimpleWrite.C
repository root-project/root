void Write(const char *filename)
{
   TFile *file = new TFile(filename,"RECREATE");
   TTree *tree = new TTree("tree","simple tree");
   Simple *s = new Simple;
   tree->Branch("simple.","Simple",&s);
   for(int i=0; i<3; ++i) {
      s->fData = i;
      tree->Fill();
   }
   file->Write();
   file->Close();
   delete file;
}

void SimpleWriteOne() 
{
   gROOT->ProcessLine(".L SimpleOne.C+");
   Write("SimpleOne.root");
}

void SimpleWriteTwo() 
{
   gROOT->ProcessLine(".L SimpleTwo.C+");
   Write("SimpleTwo.root");
}

void SimpleWrite(int version) 
{
   switch(version) {
      case 1: SimpleWriteOne(); break;
      case 2: SimpleWriteTwo(); break;
   }
}