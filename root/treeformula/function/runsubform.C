TTree *GetTree() {
   TTree *tree = new TTree("tree","tree");
   int i = 0;
   tree->Branch("i",&i);
   for (i=0; i<10; ++i) {
      tree->Fill();
   }
   tree->ResetBranchAddresses();
   return tree;
}

int test(TTree *tree) {
   TTreeFormula *f1 = new TTreeFormula("myfunc()","(i>5)",tree);
   gROOT->GetListOfFunctions()->Add(f1);
   TTreeFormula *f2 = new TTreeFormula("f2","myfunc()",tree);
   int result = 0;
   for(int j = 0; j < tree->GetEntries(); ++j) {
      tree->GetEntry(j);
      double val1 = f1->EvalInstance(0);
      double val2 = f2->EvalInstance(0);
      if ( TMath::Abs(val1 - val2) > 0.1 ) {
         fprintf(stderr,"For entry %d the 2 formula disagree f1 = %f f2 = %f\n",j,val1,val2);
         result = 1;
      }
      if ( TMath::Abs(val2 - (j>5)) > 0.1 ) {
         fprintf(stderr,"For entry %d the 2nd formula is not as expected = %d f2 = %f\n",j,(j>5),val2);
         result = 1;
      }
   }
   return result;
}
int runsubform() {
   return test(GetTree());
}
