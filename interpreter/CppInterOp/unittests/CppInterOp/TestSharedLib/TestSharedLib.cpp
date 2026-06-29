#include "TestSharedLib.h"

int ret_zero() { return 0; }

OverlayBase::OverlayBase() {}
OverlayBase::~OverlayBase() {}
int OverlayBase::frob(int x) { return x; }

int OverlayDispatchOnce(OverlayBase* b, int x) { return b->frob(x); }
