#include <iostream>
#include "TFile.h"

int run()
{
	TFile f("mytest.root", "RECREATE", "", 0);
	MyObject a;
	a.Write();
	auto sz = f.GetSize();
	f.Close();

	if (sz < 10000) {
	   std::cerr << "Error, file too small " << sz
		          << " to store object with 10000 integers" << std::endl;
      return 1;
	}

	return 0;
}
