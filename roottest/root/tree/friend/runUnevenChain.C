string chainOneStem = "chainOne";
size_t numFilesInChainOne = 2;
string chainTwoStem = "chainTwo";
size_t numFilesInChainTwo = 3;

void generateFile(string nameOfChain, size_t numberOfFiles, size_t numberOfEntriesInFile) {

   Int_t valueToPlace = numberOfFiles*1000+numberOfEntriesInFile*100;
   for (size_t i=0;i<numberOfFiles;i++) {
      TFile* newfile = new TFile(Form("%s_%ld.root", nameOfChain.c_str(), (long)i), "RECREATE"); 
      TTree*  outputTree = new TTree(nameOfChain.c_str(), nameOfChain.c_str());
      TString branchname; branchname.Form("value%ld_%ld",(long)numberOfFiles,(long)numberOfEntriesInFile);
      TString leafname; leafname.Form("leafvalue%ld_%ld/I",(long)numberOfFiles,(long)numberOfEntriesInFile);
      outputTree->Branch(branchname, &valueToPlace,leafname);
      for (size_t j=0;j<numberOfEntriesInFile;j++) {
         outputTree->Fill();
         valueToPlace++;
      }
      cout << nameOfChain << ", file " << i+1 << ", Entry number: " << outputTree->GetEntries() << endl;
      outputTree->Write();
      delete outputTree;
      delete newfile;
   } 
   
}

void generateFiles() {
  /* We put the number of events in this chain that equals the number of
   * files in the other chain so that we end up with equal numbers of entries.*/
  generateFile(chainOneStem, numFilesInChainOne, numFilesInChainTwo);
  generateFile(chainTwoStem, numFilesInChainTwo, numFilesInChainOne);
}

void runUnevenChain() {

   generateFiles();
   
   TChain firstChain(chainOneStem.c_str());
   firstChain.Add(Form("%s*.root", chainOneStem.c_str()));
   TChain secondChain(chainTwoStem.c_str());
   secondChain.Add(Form("%s*.root", chainTwoStem.c_str()));
   
   Int_t first, second;
   firstChain.SetBranchAddress("value2_3", &first);
   secondChain.SetBranchAddress("value3_2", &second);
   
   cout << firstChain.GetName() << " has " << firstChain.GetEntries() << " entries\n";
   cout << secondChain.GetName() << " has " << secondChain.GetEntries() << " entries\n";
   
   /* First output the trees to make sure they are correct. */
   for(Int_t i=0;i<secondChain.GetEntries();i++) {
      firstChain.GetEntry(i);
      secondChain.GetEntry(i);
      cout << first << " " << second << endl;
   }
   
   /* Now add the friends and test friendship. */
   cout << "Testing Friendship..." << endl;
   firstChain.AddFriend(&secondChain);
   secondChain.GetStatus()->Delete();
   first = second = 0;
   Int_t fr_second = 0;
   firstChain.SetBranchAddress("value3_2",&fr_second);
   for(Int_t i=0;i<firstChain.GetEntries();i++) {
      firstChain.GetEntry(i);
      if (first%100 != fr_second%100) {
         cout << "Chains not lined up, entry: " << i << endl;
      }
      cout << first << " " << fr_second << endl;
   }
   firstChain.Scan("value2_3:value3_2");
   
} 
