#include <cstdio>
#include <iostream>

#include "TStyle.h"
#include "TList.h"
#include "TVirtualPad.h"
#include "TMath.h"
#include "TObject.h"


TFeynmanEntry::TFeynmanEntry(const TObject* particle, const char *label) {
    SetObject((TObject*)particle);
    fParticle = label;
}
