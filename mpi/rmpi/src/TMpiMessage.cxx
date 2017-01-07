#include<Mpi/TMpiMessage.h>
#include<TClass.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TMpiMessageInfo::TMpiMessageInfo(const TMpiMessageInfo &msgi): TObject(msgi)
{
   fBuffer = msgi.fBuffer;
   fDataTypeName = msgi.fDataTypeName;
   fSource = msgi.fSource;
   fDestination = msgi.fDestination;
   fTag = msgi.fTag;
}

//______________________________________________________________________________
TMpiMessageInfo::TMpiMessageInfo(const Char_t *buffer, UInt_t size)
{
   fBuffer = TString(buffer, size);
}

//______________________________________________________________________________
TMpiMessage::TMpiMessage(Char_t *buffer, Int_t size): TMessage(buffer, size)
{
   SetReadMode();
   Reset();
}

//______________________________________________________________________________
TMpiMessage::TMpiMessage(UInt_t what, Int_t bufsiz): TMessage(what, bufsiz) { }

