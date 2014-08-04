
void FeldmanCousins()
{
 // Example macro of using the TFeldmanCousins class in root.
 //
 // get a FeldmanCousins calculation object with the default limits
 // of calculating a 90% CL with the minimum signal value scanned
 // = 0.0 and the maximum signal value scanned of 50.0
 //Author : Adrian John Bevan <bevan@SLAC.Stanford.EDU>

 if (!gROOT->GetClass("TFeldmanCousins")) gSystem->Load("libPhysics");

 TFeldmanCousins f;

  // calculate either the upper or lower limit for 10 observerd
  // events with an estimated background of 3.  The calculation of
  // either upper or lower limit will return that limit and fill
  // data members with both the upper and lower limit for you.
  Double_t Nobserved   = 10.0;
  Double_t Nbackground = 3.0;

  Double_t ul = f.CalculateUpperLimit(Nobserved, Nbackground);
  Double_t ll = f.GetLowerLimit();

  cout << "For " <<  Nobserved << " data observed with and estimated background"<<endl;
  cout << "of " << Nbackground << " candidates, the Feldman-Cousins method of "<<endl;
  cout << "calculating confidence limits gives:"<<endl;
  cout << "\tUpper Limit = " <<  ul << endl;
  cout << "\tLower Limit = " <<  ll << endl;
  cout << "at the 90% CL"<< endl;
}

