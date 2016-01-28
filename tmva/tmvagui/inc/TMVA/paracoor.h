#ifndef paracoor__HH
#define paracoor__HH
#include "tmvaglob.h"
#include "TROOT.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TParallelCoord.h"
#include "TParallelCoordVar.h"
#include "TParallelCoordRange.h"
namespace TMVA{


   // plot parallel coordinates

   void paracoor( TString fin = "TMVA.root", TString fDatasetName = "dataset", Bool_t useTMVAStyle = kTRUE );
}
#endif
