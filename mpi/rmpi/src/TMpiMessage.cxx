#include<Mpi/TMpiMessage.h>
#include<TClass.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TMpiMessage::TMpiMessage(Char_t *buffer, Int_t size): TMessage(buffer, size)
{
   SetReadMode();
   Reset();
}

//______________________________________________________________________________
TMpiMessage::TMpiMessage(UInt_t what, Int_t bufsiz): TMessage(what, bufsiz) { }
