void Write(const char *filename)
{
   TFile *file = new TFile(filename,"RECREATE");
   TTree *tree = new TTree("tree","simple tree");
   Simple *s = new Simple;
   tree->Branch("simple","Simple",&s,32000,0);
   tree->Branch("simpleSplit.","Simple",&s,32000,99);
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
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L SimpleOne.C+");
#endif
   Write("SimpleOne.root");
}

void SimpleWriteTwo() 
{
#ifndef ClingWorkAroundMissingDynamicScope
   gROOT->ProcessLine(".L SimpleTwo.C+");
#endif
   Write("SimpleTwo.root");
}

void SimpleWrite(int version) 
{
   switch(version) {
      case 1: SimpleWriteOne(); break;
      case 2: SimpleWriteTwo(); break;
   }
}
