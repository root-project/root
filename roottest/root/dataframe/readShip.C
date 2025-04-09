// This is a test for ROOT-9674
int readShip()
{
   ROOT::RDataFrame f("cbmsim","ship_ROOT_9674.root");
   auto name = f.GetColumnType("MCTrack.fPdgCode");
   if(name != "ROOT::VecOps::RVec<Int_t>") {
      std::cerr << "ERROR: expecting the type name ROOT::VecOps::RVec<Int_t> for column MCTrack.fPdgCode and got '" << name << "' instead!" << std::endl;
      return 1;
   }
   return 0;
}
