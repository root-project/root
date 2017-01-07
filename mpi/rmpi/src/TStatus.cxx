// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#include<Mpi/TStatus.h>

using namespace ROOT::Mpi;

//______________________________________________________________________________
TStatus::TStatus(): fStatus() {}

//______________________________________________________________________________
TStatus::TStatus(const TStatus &data): TObject(data), fStatus(data.fStatus) {}

//______________________________________________________________________________
Bool_t TStatus::IsCancelled() const
{
   return fStatus.Is_cancelled();
}


//______________________________________________________________________________
Int_t TStatus::GetSource() const
{
   return fStatus.Get_source();
}

//______________________________________________________________________________
void TStatus::SetSource(Int_t source)
{
   fStatus.Set_source(source);
}

//______________________________________________________________________________
Int_t TStatus::GetTag() const
{
   return fStatus.Get_tag();
}

//______________________________________________________________________________
void TStatus::SetTag(Int_t tag)
{
   fStatus.Set_tag(tag);
}

//______________________________________________________________________________
Int_t TStatus::GetError() const
{
   return fStatus.Get_error();
}

//______________________________________________________________________________
void TStatus::SetError(Int_t error)
{
   fStatus.Set_error(error);
}

//______________________________________________________________________________
void TStatus::SetCancelled(Bool_t flag)
{
   fStatus.Set_cancelled(flag);
}
