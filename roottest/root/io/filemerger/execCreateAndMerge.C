void execCreateAndMerge() { 
//-------------------------------------------------------------------------- 
for ( Int_t ifile = 0 ; ifile < 4 ; ifile++ ) { 
   TFile* outputFile = TFile::Open(Form("a_file_%d.root",ifile),"RECREATE"); 
   TNamed* namedObj = new TNamed(Form("namedObj%d",ifile),Form("namedObj%d",ifile)); 
   namedObj->Write(); 
   outputFile->Close(); 
} 
cout << "4 a files created" << endl; 
//-------------------------------------------------------------------------- 
//-------------------------------------------------------------------------- 
for ( Int_t ifile = 0 ; ifile < 4 ; ifile++ ) { 
   TFile* outputFile = TFile::Open(Form("b_file_%d.root",ifile),"RECREATE"); 
   TNamed* namedObj = new TNamed("namedObj","namedObj"); 
   namedObj->Write(); 
   outputFile->Close(); 
} 
cout << "4 b files created" << endl; 
//-------------------------------------------------------------------------- 
//-------------------------------------------------------------------------- 
TFileMerger* aMerger = new TFileMerger(kFALSE); 
 for ( Int_t ifile = 0 ; ifile < 4 ; ifile++ ) { 
    TFile* inFile = TFile::Open(Form("a_file_%d.root",ifile)); 
    aMerger->AddFile(inFile); 
 } 
 aMerger->OutputFile("a_file.root"); 
 aMerger->Merge(); 
 cout << "4 a files merged" << endl; 
//-------------------------------------------------------------------------- 
//-------------------------------------------------------------------------- 
 TFileMerger* bMerger = new TFileMerger(kFALSE); 
 for ( Int_t ifile = 0 ; ifile < 4 ; ifile++ ) { 
    TFile* inFile = TFile::Open(Form("b_file_%d.root",ifile)); 
    bMerger->AddFile(inFile); 
 } 
 bMerger->OutputFile("b_file.root"); 
 bMerger->Merge(); 
cout << "4 b files merged" << endl; 
//-------------------------------------------------------------------------- 
//-------------------------------------------------------------------------- 
TFile* inAFile = TFile::Open("a_file.root"); 
cout << "******* MERGED A FILE ******** 4 objects with different names" << endl; 
inAFile->ls(); 
cout << "******************************" << endl; 
//-------------------------------------------------------------------------- 
//-------------------------------------------------------------------------- 
TFile* inBFile = TFile::Open("b_file.root"); 
cout << "******* MERGED B FILE ******** 4 objects with the same names" << endl; 
inBFile->ls(); 
cout << "******************************" << endl; 
//-------------------------------------------------------------------------- 
} 
