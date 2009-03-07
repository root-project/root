#include <iostream>
#include <TROOT.h>
#include <TSystem.h>
#include <TF1.h>

using namespace std;

int main(int argc, char **argv)
{

// gSystem->ListLibraries();

cout << "Now loading libXMLIO:" << endl;
cout << "Loading library = " << gSystem->Load("libXMLIO.so") << endl;

//cout << "\nListing libraries after loading RFIO:" << endl;
//gSystem->ListLibraries();

cout << "\nInitializing function TF1... " << endl;
TF1* function2 = new TF1("fa2","TMath::Exp(-(x-[0])/[1])",0,10);
cout << "...initialized." << endl;

return (function2==0);

}
