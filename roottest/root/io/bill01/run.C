#include <iostream>
#include "TFile.h"

int run()
{
   Long64_t size_small = 0, size_big = 0;

	{
	   TFile fsmall("mytest_small.root", "RECREATE", "small object", 0);
	   MyObject asmall(100, 50);
	   asmall.Write();
	   size_small = fsmall.GetSize();
	}

   {
	   TFile fbig("mytest_big.root", "RECREATE", "big object", 0);
	   MyObject abig(10000, 5000);
	   abig.Write();
	   size_big = fbig.GetSize();
	}

	if (size_small * 10 >= size_big) {
	   std::cerr << "Error, file with small object " << size_small
		          << " too big compared with file with large object " << size_big << std::endl;
		return 1;
	}

	return 0;
}
