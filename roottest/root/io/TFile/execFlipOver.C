int execFlipOver()
{
   std::vector<int> data;
   data.resize(  (100*1024*1024+408620)/4);
   std::vector<float> smalldata;
   smalldata.resize( 3 );

   fprintf(stderr,"Creating input file\n");
   TString dataName;
   TMemFile *f = new TMemFile("willbelarge.root","CREATE");
   f->SetCompressionLevel(0);

   // Get to close to 2Gb.
   for(int i=0; i < 19; ++i) {
      smalldata.resize(2*i);
      dataName.Form("smalldata%d",i);
      f->WriteObject(&smalldata,dataName);

      dataName.Form("largedata%d",i);
      f->WriteObject(&data,dataName);

      smalldata.resize(10*i);
      dataName.Form("smalldata%d",i);
      f->WriteObject(&smalldata,dataName,"overwrite");

      // f->GetListOfFree()->ls();
   }

   fprintf(stderr,"Write input file header\n");
   f->Write();

   constexpr Long64_t kBufferLen = 2.1*1024*1024*1024;
   char *buffer = new char[kBufferLen];

   // fprintf(stderr,"Size is %lld and buffer is %p\n",kBufferLen,buffer);
   fprintf(stderr,"Copying input file into char buffer\n");
   f->CopyTo(buffer,kBufferLen);


   fprintf(stderr,"Reading TMemFile\n");
   TFile *f2 = new TMemFile("copyof.root", buffer, kBufferLen, "UPDATE");

   if (!f2->GetListOfFree()) {
      Error("flipOver","Missing list of free block in copy of large file.");
      return 1;
   }
   TFree *block = (TFree*)f2->GetListOfFree()->Last();
   if (!block) {
      Error("flipOver","Missing the last item in the list of free block in copy of large file.");
      return 2;
   }
   if (block->GetLast() != TFile::kStartBigFile+1000000000LL) {
      Error("flipOver","The last item in the list of free block in copy of large file is wrong.");
      block->ls("");
      return 3;
   }

   // Now let's see if we can safely go over 3GB.

   fprintf(stderr,"Filling input file with 1 more GB.\n");
   data.resize(  (100*1024*1024-4857697-60)/4);
   f->Delete("*;*");
   for(int i=0; i < 30; ++i) {
      dataName.Form("secondlargedata%d",i);
      f->WriteObject(&data,dataName);

      // f->GetListOfFree()->ls();
   }

   // Set the last free block to just fit the list of keys, so we can test
   // this boundary condition.
   auto be = (TFree*)f->GetListOfFree()->Last();
   be->SetFirst(2999997803-7);
   f->SetEND(2999997803-7);
   //f->GetListOfFree()->ls();

   fprintf(stderr,"Write input file header\n");
   f->Write();

   block = (TFree*)f->GetListOfFree()->Last();
   if (!block) {
      Error("flipOver","Missing the last item in the list of free block in copy of large file.");
      return 4;
   }
   if (block->GetLast() != TFile::kStartBigFile+2000000000LL) {
      Error("flipOver","The last item in the list of free block in copy of large file is wrong.  It should end at 4000000000.");
      block->ls("");
      return 5;
   }

   return 0;
}
