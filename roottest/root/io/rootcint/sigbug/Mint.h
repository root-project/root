#ifndef MINT_H
#define MINT_H

#include <Rtypes.h>

#include "sigc++/sigc++.h"

#include "Jint.h"

class Mint
{

public:
    Mint();
    virtual ~Mint();

    Jint& GetJint() { return fJint; }

    SigC::Signal0<void> digits_selected;
    SigC::Signal0<void> strips_selected;
    SigC::Signal0<void> digit_picked;
    SigC::Signal0<void> strip_picked;

private:
    Jint fJint;

ClassDef(Mint,0)
};                              // end of class Mint

extern Mint* gMint;             // global pointer to most recent
                                // (normaly only) Mint object.
#endif  // MINT_H
