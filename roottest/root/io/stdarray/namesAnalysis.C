void namesAnalysis() {
   string typenamebuf;
   int ndim;
   std::array<int,5> maxi {0,0,0,0,0};
   auto names = {"C<array<bool,4>,array<int,5>>",
                 "array<int,3>",
                 "array<array<T,4>,3>",
                 "array<array<array<L,16>,4>,3>",
                 "array<array<array<array<double,13>,16>,4>,3>",
                 "array<array<array<array<array<c,18>,13>,16>,4>,3>",
                 "array<A<int>,12>",
                 "array<array<A<array<int,6>>,4>,3>"};
   for (auto& name : names) {
      maxi = {0,0,0,0,0};
      if(! TClassEdit::GetStdArrayProperties(name,typenamebuf,maxi,ndim)){
         cout << "Class \"" << name << "\" is not an STL array\n";
         continue;
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
