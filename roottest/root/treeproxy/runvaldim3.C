#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>

#if defined(__CINT__) && !defined(__MAKECINT__)
#include "Track.C+"
#elif defined(__CLING__) && !defined(__MAKECLING__) && !defined(ClingWorkAroundMissingSmartInclude)
#include "Track.C+"
#else
#include "Track.h"
#endif

void runvaldim3() 
{
   TFile file("forproxy.root");
   TTree *t; file.GetObject("t",t);
   t->Process("val3dimSel.h+");
}