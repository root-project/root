#include <cstdio>
#include <iostream>

#include "TStyle.h"
#include "TLatex.h"
#include "TLine.h"
#include "TPolyLine.h"
#include "TMarker.h"
#include "TLegend.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TROOT.h"
#include "TObject.h"
#include "TLegendEntry.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"
#include "THStack.h"

TFeynmanEntry::TFeynmanEntry(const TObject* particle) {
    SetObject((TObject*)particle);
}
void TFeynmanEntry::SetObject(TObject *obj) {
  fObject = obj;
}
