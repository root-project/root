#include "TVirtualPadPainter.h"
#include "TPluginManager.h"

ClassImp(TVirtualPadPainter)

//______________________________________________________________________________
TVirtualPadPainter::~TVirtualPadPainter()
{
   //Virtual dtor.
}

//______________________________________________________________________________
void TVirtualPadPainter::InitPainter()
{
   //Empty definition.
}

//______________________________________________________________________________
void TVirtualPadPainter::InvalidateCS()
{
   //Empty definition.
}

//______________________________________________________________________________
void TVirtualPadPainter::LockPainter()
{
   //Empty definition.
}

//______________________________________________________________________________
TVirtualPadPainter *TVirtualPadPainter::PadPainter(Option_t *type)
{
   // Create a pad painter of specified type.

   TVirtualPadPainter *painter = 0;
   TPluginHandler *h = gPluginMgr->FindHandler("TVirtualPadPainter", type);
   
   if (h && h->LoadPlugin() != -1)
      painter = (TVirtualPadPainter *) h->ExecPlugin(0);

   return painter;
}
