#include<Mpi/TErrorHandler.h>

// namespace ROOT {
//    namespace Mpi {
//       const TErrorHandler  ERRORS_ARE_FATAL = MPI::ERRORS_ARE_FATAL;
//       const TErrorHandler  ERRORS_RETURN = MPI::ERRORS_RETURN;
//       const TErrorHandler  ERRORS_THROW_EXCEPTIONS = MPI::ERRORS_THROW_EXCEPTIONS;
//    }
// }

using namespace ROOT::Mpi;

//______________________________________________________________________________
TErrorHandler::TErrorHandler(): fErrorHandler(MPI_ERRORS_ARE_FATAL) {}

//______________________________________________________________________________
TErrorHandler::TErrorHandler(const TErrorHandler &err) : TObject(err), fErrorHandler(err.fErrorHandler) { }

//______________________________________________________________________________
Int_t TErrorHandler::GetErrorClass(Int_t errcode)
{
   Int_t eclass;
   MPI_Error_class(errcode, &eclass);
   return eclass;
}

//______________________________________________________________________________
TString TErrorHandler::GetErrorString(Int_t errcode)
{
   Char_t *estring = new Char_t[MAX_ERROR_STRING];
   Int_t size;
   MPI_Error_string(errcode, estring, &size);
   return TString(estring, size);
}

//______________________________________________________________________________
void  TErrorHandler::SetErrorString(Int_t errcode, const TString msg)
{
   MPI_Add_error_string(errcode, const_cast<Char_t *>(msg.Data()));
}

//______________________________________________________________________________
Int_t TErrorHandler::CreateErrorClass()
{
   Int_t errcls;
   MPI_Add_error_class(&errcls);
   return errcls;
}

//______________________________________________________________________________
Int_t TErrorHandler::CreateErrorCode(Int_t errclass)
{
   Int_t errcode;
   MPI_Add_error_code(errclass, &errcode);
   return errcode;
}

//______________________________________________________________________________
void TErrorHandler::Free()
{
   MPI_Errhandler_free(&fErrorHandler);
}


