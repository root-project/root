{
// Fill out the code of the actual test
   TFile *file = TFile::Open("RefTest.root");
   TTree *MetaData; file->GetObject("MetaData",MetaData);
   MetaData->Scan("ProductDescription.productList_.first.friendlyClassName_.c_str()","","colsize=12");
   return 0;
}
