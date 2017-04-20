// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#include<Mpi/TStatus.h>
#include<Mpi/TErrorHandler.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TStatus::TStatus(): fStatus() {}

//______________________________________________________________________________
TStatus::TStatus(const TStatus &data): TObject(data), fStatus(data.fStatus) {}

//______________________________________________________________________________
Bool_t TStatus::IsCancelled() const
{
   Int_t t;
   ROOT_MPI_CHECK_CALL(MPI_Test_cancelled,(const_cast<MPI_Status *>(&fStatus), &t),TStatus::Class_Name());
   return (Bool_t)t;
}


//______________________________________________________________________________
Int_t TStatus::GetSource() const
{
   return  fStatus.MPI_SOURCE;
}

//______________________________________________________________________________
void TStatus::SetSource(Int_t source)
{
   fStatus.MPI_SOURCE = source;
}

//______________________________________________________________________________
Int_t TStatus::GetTag() const
{
   return fStatus.MPI_TAG;
}

//______________________________________________________________________________
void TStatus::SetTag(Int_t tag)
{
   fStatus.MPI_TAG = tag;
}

//______________________________________________________________________________
Int_t TStatus::GetError() const
{
   return fStatus.MPI_ERROR;
}

//______________________________________________________________________________
void TStatus::SetError(Int_t error)
{
   fStatus.MPI_ERROR = error;
}

//______________________________________________________________________________
void TStatus::SetCancelled(Bool_t flag)
{
   ROOT_MPI_CHECK_CALL(MPI_Status_set_cancelled,(&fStatus, (Int_t) flag),TStatus::Class_Name());
}
