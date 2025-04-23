void CreateTreeWithStdArray(){
   TFile f("test.root", "RECREATE");
   TTree t("tree", "tree");
   std::array<Double_t,3> arr;
   arr[0] = 10.1;
   arr[1] = 9.2;
   arr[2] = 8.3;
   t.Branch("arr1", &arr, "arr1[3]/D");
   t.Branch("arr2", &arr);
   t.Fill();
   t.Write();
}

int execStdArray()
{
   CreateTreeWithStdArray();
   TFile f("test.root", "READ");
   TTree *t = f.Get<TTree>("tree");
   if (!t) {
      Error("execStdArray","Could not find the TTree 'tree'");
      return 1;
   }
   t->Scan("arr1:arr2");
   return 0;
}
