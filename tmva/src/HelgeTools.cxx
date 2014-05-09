#include <algorithm>
#include <cstdlib>

#include "TObjString.h"
#include "TMath.h"
#include "TString.h"
#include "TTree.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TH2.h"
#include "TList.h"
#include "TSpline.h"
#include "TVector.h"
#include "TMatrixD.h"
#include "TMatrixDSymEigen.h"
#include "TVectorD.h"
#include "TTreeFormula.h"
#include "TXMLEngine.h"
#include "TROOT.h"
#include "TMatrixDSymEigen.h"

#ifndef ROOT_TMVA_HelgeTools
#include "TMVA/HelgeTools.h"
#endif
#ifndef ROOT_TMVA_Config
#include "TMVA/Config.h"
#endif
#ifndef ROOT_TMVA_Event
#include "TMVA/Event.h"
#endif
#ifndef ROOT_TMVA_Version
#include "TMVA/Version.h"
#endif
#ifndef ROOT_TMVA_PDF
#include "TMVA/PDF.h"
#endif
#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

using namespace std;

