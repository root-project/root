#include <iostream>
#include "TFile.h"
#include "MyObject.h"
main(int argc, char **argv)
{
	int allocated = 0;
	int filled = 0;
	switch (argc) {
	case 1:
		break;
	case 2:
		allocated = atoi(argv[1]);
		break;
	case 3:
		allocated = atoi(argv[1]);
		filled = atoi(argv[2]);
		if (filled > allocated) {
			allocated = filled;
		}
		break;
	default:
		cerr << "too many arguments\n";
		return(1);
	}
	TFile f("mytest.root","RECREATE", "", 0);
	MyObject a(allocated, filled);
	a.Write();
	f.Close();
	return(0);
}

ClassImp(MyObject)
