#include "TVirtualPadPainter.h"
#include "TPluginManager.h"

ClassImp(TVirtualPadPainter)

////////////////////////////////////////////////////////////////////////////////
///Virtual dtor.

TVirtualPadPainter::~TVirtualPadPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::InitPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::InvalidateCS()
{
}

////////////////////////////////////////////////////////////////////////////////
///Empty definition.

void TVirtualPadPainter::LockPainter()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create a pad painter of specified type.

TVirtualPadPainter *TVirtualPadPainter::PadPainter(Option_t *type)
{
   TVirtualPadPainter *painter = 0;
   TPluginHandler *h = gPluginMgr->FindHandler("TVirtualPadPainter", type);

   if (h && h->LoadPlugin() != -1)
      painter = (TVirtualPadPainter *) h->ExecPlugin(0);

   return painter;
}
