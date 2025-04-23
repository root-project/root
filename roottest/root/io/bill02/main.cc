#include <iostream>
#include "TFile.h"
#include "MyObject.h"
main(int argc, char **argv)
{
	TFile f("mytest.root","RECREATE", "", 0);
	MyObject a;
	a.Write();
	f.Close();
	return(0);
}

ClassImp(MyObject)
