#include "aHeader.h"

void execAutoloadOnInclusion(){

 if ( 1 == gSystem->Load("libHist.so") )
    std::cout << "libHist.so already loaded through header inclusion!\n";
 else
    std::cout << "Error: libHist.so should have been already up!\n";

}
