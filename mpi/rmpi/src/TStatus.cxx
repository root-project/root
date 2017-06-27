// @(#)root/mpi / Author: Omar.Zapata@cern.ch 2017 http://oproject.org
#include<Mpi/TStatus.h>
#include<Mpi/TErrorHandler.h>
using namespace ROOT::Mpi;

//______________________________________________________________________________
TStatus::TStatus(): fStatus() {}

//______________________________________________________________________________
TStatus::TStatus(const TStatus &data): TObject(data), fStatus(data.fStatus) {}

//______________________________________________________________________________
/**
 * Method to get if the current process was cancelled
 * \return integer with the status value
 */
Bool_t TStatus::IsCancelled() const
{
   Int_t t;
   ROOT_MPI_CHECK_CALL(MPI_Test_cancelled, (const_cast<MPI_Status *>(&fStatus), &t), TStatus::Class_Name());
   return (Bool_t)t;
}


//______________________________________________________________________________
/**
 * Method to get the source of the process
 * \return integer with the rank or process id value
 */
Int_t TStatus::GetSource() const
{
   return  fStatus.MPI_SOURCE;
}

//______________________________________________________________________________
/**
* Method to set the source to the status message
* \param source integer with the rank or process id value
*/
void TStatus::SetSource(Int_t source)
{
   fStatus.MPI_SOURCE = source;
}

//______________________________________________________________________________
/**
* Method to get the tag id of the process
* \return integer with the tag id
*/
Int_t TStatus::GetTag() const
{
   return fStatus.MPI_TAG;
}

//______________________________________________________________________________
/**
* Method to set the tag id to the status of the process
* \param tag integer with the tag id
*/
void TStatus::SetTag(Int_t tag)
{
   fStatus.MPI_TAG = tag;
}

//______________________________________________________________________________
/**
 * Method to get the error id of the process
 * \return integer with the error id
 */
Int_t TStatus::GetError() const
{
   return fStatus.MPI_ERROR;
}

//______________________________________________________________________________
/**
 * Method to set the error id to the status message
 * \param error integer with the error id
 */

void TStatus::SetError(Int_t error)
{
   fStatus.MPI_ERROR = error;
}

//______________________________________________________________________________
/**
 * Method to set the cancelled flag to the status message
 * \param flag boolean to set the cancelled flag to the message
 */

void TStatus::SetCancelled(Bool_t flag)
{
   ROOT_MPI_CHECK_CALL(MPI_Status_set_cancelled, (&fStatus, (Int_t) flag), TStatus::Class_Name());
}

//______________________________________________________________________________
/**
 * Returns the size of the message in bytes
 */
Int_t TStatus::GetMsgSize() const
{
   return fMsgSize;
}

//______________________________________________________________________________
/**
 * Method to set the size of the message in bytes
 * \param size size of the message in bytes
 */
void TStatus::SetMsgSize(Int_t size)
{
   fMsgSize = size;
}

