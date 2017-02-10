class B{int a; std::array<int,3> b; int c[3];};

void checkRealData() {
   auto c = TClass::GetClass("B");
   c->BuildRealData();
   auto lrd = c->GetListOfRealData();
   for (auto i : ROOT::TSeqI(3)) {
      cout << "B Real Data " << lrd->At(i)->GetName() << endl;
   }
}
