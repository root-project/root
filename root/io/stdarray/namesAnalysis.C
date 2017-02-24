void namesAnalysis() {
   string typenamebuf;
   int ndim;
   std::array<int,5> maxi {0,0,0,0,0};
   auto names = {"array<int,3>",
                 "array<array<T,4>,3>",
                 "array<array<array<L,16>,4>,3>",
                 "array<array<array<array<double,13>,16>,4>,3>",
                 "array<array<array<array<array<c,18>,13>,16>,4>,3>",
                 "array<int,3>"};
   for (auto& name : names) {
      if(! TClassEdit::GetStdArrayProperties(name,typenamebuf,maxi,ndim)){
         cerr << "ERROR: class \"" << name << "\" is not an STL array\n";
      }
      cout << name << ":" << endl
           << " - typename: " << typenamebuf << endl
           << " - ndim: " << ndim << endl
           << " - max index:" << endl;
      for (int i=0; i<5; ++i) {
         cout << "  o Dim " << i << ": "<< maxi[i] << endl;
      }
   }
}
