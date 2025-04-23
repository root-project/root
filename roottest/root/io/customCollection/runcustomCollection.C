{
   // Fill out the code of the actual test
   printf("Reading file with just a TTree\n");
   TFile *file = TFile::Open("tcoll.root","READ");
   TTree *tree = 0;
   file->GetObject("T",tree);
   if (tree) {
      tree->Scan("coll.fName.Length()");
      tree->Scan();
   }
   delete file;

   printf("Reading TTree\n");
   file = TFile::Open("coll.root","READ");
   tree = 0;
   file->GetObject("T",tree);
   if (tree) {
      tree->Scan("coll.fName.Length()");
      tree->Scan();
   }
   delete file;
   return 0;
}
