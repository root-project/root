int runROOT6440() {
   gSystem->Load("libROOT6440_dictrflx");
   return (TClass::GetClass("TheTemplTempl<SomeTemplate>") ? 0 : 1);
}
