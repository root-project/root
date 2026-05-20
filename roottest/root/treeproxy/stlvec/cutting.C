bool cutting(){

//   cout<<"Event properties: "<<eventNumber->size()<<eventNumber->front()<<endl;
    
   TStlPx_TLorentzVector myjet4mom=jet4mom;
//   for (UInt_t jets=0; jets < myjet4mom->size(); ++jets ) {
//      cout<<"Jet No: "<<jets<<endl;
//      myjet4mom[jets].Dump();
//   }
   if (myjet4mom->size()!=2){
      return false;
   }
   else {
//      cout<<"Number of Jets-Cut passed... "<<myjet4mom->size()<<endl;
   }
   if (myjet4mom[0].Pt() < 100){
      return false;
   }
   else {
//      cout<<"Pt of Jets-Cut (100) passed... "<<myjet4mom[0].Pt()<<endl;
   }

   return true;
}
