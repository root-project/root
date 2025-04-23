#include "Mint.h"

#include "sigc++/class_slot.h"

Mint* gMint = 0;

ClassImp(Mint)

Mint::Mint()
{
    gMint = this;
}

Mint::~Mint() 
{
}
