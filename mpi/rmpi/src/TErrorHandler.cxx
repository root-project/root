#include<Mpi/TErrorHandler.h>
#include<Mpi/TGroup.h>
#include<Mpi/TIntraCommunicator.h>

using namespace ROOT::Mpi;
Bool_t TErrorHandler::fVerbose = kFALSE;
//______________________________________________________________________________
TErrorHandler::TErrorHandler(): fErrorHandler(MPI_ERRORS_RETURN) {}

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

//______________________________________________________________________________
template<> void TErrorHandler::TraceBack(const TGroup *group, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *_msg)
{
   TString msg;
   if (TErrorHandler::IsVerbose()) {
      msg += Form("\nRank   = %d", group->GetRank());
      msg += Form("\nSize   = %d", group->GetSize());
      msg += Form("\nObject = %s", group->ClassName());
      msg += Form("\nHost   = %s", TEnvironment::GetProcessorName().Data());
   }

   msg += Form("\nCode = %d", errcode);
   msg += Form("\nName = %s", GetErrorString(errcode).Data());
   msg += Form("\nMessage = %s", _msg);
   msg += "\nAborting, finishing the remaining processes.";
   msg += "\n--------------------------------------------------------------------------\n";

   group->Error(Form("%s(...) %s[%d]", function, file, line), "%s", msg.Data());
   COMM_WORLD.Abort(errcode, kTRUE);
}

//______________________________________________________________________________
template<> void TErrorHandler::TraceBack(const Char_t *class_name, const Char_t *function, const Char_t *file, Int_t line, Int_t errcode, const Char_t *_msg)
{
   TString msg;
   if (TErrorHandler::IsVerbose()) {
      msg += "\nUsing COMM_WORLD";
      msg += Form("\nRank   = %d", COMM_WORLD.GetRank());
      msg += Form("\nSize   = %d", COMM_WORLD.GetSize());
      msg += Form("\nObject = %s", class_name);
      msg += Form("\nHost   = %s", TEnvironment::GetProcessorName().Data());
   }

   msg += Form("\nCode = %d", errcode);
   msg += Form("\nName = %s", GetErrorString(errcode).Data());
   msg += Form("\nMessage = %s", _msg);
   msg += "\nAborting, finishing the remaining processes.";
   msg += "\n--------------------------------------------------------------------------\n";

   COMM_WORLD.Error(Form("%s(...) %s[%d]", function, file, line), "%s", msg.Data());
   COMM_WORLD.Abort(errcode, kTRUE);
}
