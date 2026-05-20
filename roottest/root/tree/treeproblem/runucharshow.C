{
   UChar_t x = 200;
   TTree tree("tree","tree");
   tree.Branch("x",&x,"x/b");
   tree.Branch("y",&x,"y/B");
   tree.Fill();
   x = 250;
   tree.Fill();
   tree.Show(0);
   tree.Show(1);
   tree.Scan("x:y");
#ifdef ClingWorkAroundBrokenUnnamedReturn
   int res = 0;
#else
   return 0;
#endif
}

