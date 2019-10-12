#include "RooFitMoreLib.h"
#include "RooMsgService.h"

// Load automatically RooFitMore library that automatically will register the
// integrator classes
void RooFitMoreLib::Load() {
   oocoutI((TObject*)nullptr, InputArguments) << "libRooFitMore has been loaded " << std::endl;
}
