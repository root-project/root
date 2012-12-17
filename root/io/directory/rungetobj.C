void rungetobj() {
   // make a subdir /a/b/a
   TFile *f=TFile::Open("TDirGetObj.root","RECREATE");
   gDirectory=f->mkdir("a")->mkdir("b");
   TNamed* named=new TNamed("a","this is not gDir->GetDir(\"a\")!");
   named->Write();
   f->ls();
   delete f;
   
   f = TFile::Open("TDirGetObj.root");
   TDirectory* dirRead=0;
#ifdef ClingReinstateImplicitDynamicCast
   TDirectory* dirReadOld = gDirectory->Get("a/b/a");
#else
   TDirectory* dirReadOld = (TDirectory*)gDirectory->Get("a/b/a");
#endif
   gDirectory->GetObject("a/b/a", dirRead);

   if (dirRead) {
      printf("Ooops, obviously, gDirectory->GetObj(\"/a/b/a\") should be a TNamed, not a %s.\n",
          dirRead->IsA()->GetName());
      printf("We've picked up %p, which is /a=%p, not /a/b/a.\n\n", dirRead, f->Get("a"));
   }
   if (dirReadOld->IsA()!=TNamed::Class()) {
      printf("Ooops, obviously, gDirectory->Get(\"/a/b/a\") should be a TNamed, not a %s.\n",
          dirReadOld->IsA()->GetName());
      printf("We've picked up %p, which is /a=%p, not /a/b/a.\n\n", dirReadOld, f->Get("a"));
   }
   gDirectory->GetObject("a/b/a",named);   
   if (named->IsA()!=TNamed::Class()) {
      printf("Even though that's wrong, once we have read it, we can \"cast\" it to anything, e.g. even read back our TNamed, even though it's still a TDir:\n");
      printf("named is now %p, as is the TDir /a=%p\n", named, f->Get("a"));
   }
   delete f;
}
