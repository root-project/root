#include "TFeynman.h"

void feynman()
{
    TCanvas *c1 = new TCanvas();
    TFeynman *f = new TFeynman();

    f->AddItem("fermion", 10, 10, 30, 30);

    f->Paint();

    f->Draw();
}
