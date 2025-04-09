#include "Riostream.h"
void semicolon(const char *string = "/star/simu/simu/jwebb/11-12-2010-pp500-pileup/rcf10100_1000_5evts_Wplus_enu.fzd;  gfile b /star/simu/simu/jwebb/11-12-2010-pp500-pileup/rcf10000_1000_250evts_minb.fzd; mode TPCE back 4001400; gback 400 400 0.1 106.6") {
  cout << string << endl;
}

#if !defined(__CINT__) && !defined(__CLING__)
int main(int /* argc */, char **argv) {
  semicolon(argv[1]);
  return 0;
}
#endif
