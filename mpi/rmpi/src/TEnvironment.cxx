#include<Mpi/TEnvironment.h>
#include<Mpi/TIntraCommunicator.h>
using namespace ROOT::Mpi;
//TODO: enable thread level and thread-safe for ROOT

//______________________________________________________________________________
TEnvironment::TEnvironment(): fBuffer(new Char_t[MAX_IO_BUFFER + 1])
{
   MPI_Init(NULL, NULL);

   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
   } else {
      //TODO: added error handling here
   }
}

//______________________________________________________________________________
TEnvironment::TEnvironment(Int_t &argc, Char_t ** &argv)
{
   MPI_Init(&argc, &argv);
   if (IsInitialized()) {
      Int_t result;
      MPI_Comm_compare((MPI_Comm)COMM_WORLD, MPI_COMM_WORLD, &result);
      if (result == IDENT) COMM_WORLD.SetCommName("ROOT::Mpi::COMM_WORLD");
   } else {
      //TODO: added error handling here
   }
}

//______________________________________________________________________________
TEnvironment::~TEnvironment()
{
   //if mpi's environment is initialized then finalize it
   MPI_Barrier(MPI_COMM_WORLD);
   if (!IsFinalized()) {
      Finalize();
   }
   EndCapture();
   Flush();
   fBuffer = nullptr;
}

//______________________________________________________________________________
void TEnvironment::InitCapture()
{
   if (!fSyncOutput) {
      /* save stdout/stderr for display later */
      fSavedStdOut = dup(STDOUT_FILENO);
      fSavedStdErr = dup(STDERR_FILENO);
      if (pipe(fStdOutPipe) != 0) {           /* make a pipe for stdout*/
         return;
      }
      if (pipe(fStdErrPipe) != 0) {           /* make a pipe for stdout*/
         return;
      }

      Long_t flags = fcntl(fStdOutPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdOutPipe[0], F_SETFL, flags);

      flags = fcntl(fStdErrPipe[0], F_GETFL);
      flags |= O_NONBLOCK;
      fcntl(fStdErrPipe[0], F_SETFL, flags);

      dup2(fStdOutPipe[1], STDOUT_FILENO);   /* redirect stdout to the pipe */
      close(fStdOutPipe[1]);

      dup2(fStdErrPipe[1], STDERR_FILENO);   /* redirect stderr to the pipe */
      close(fStdErrPipe[1]);

      fSyncOutput = true;
   }
}

//______________________________________________________________________________
void TEnvironment::EndCapture()
{
   if (fSyncOutput) {
      fflush(stdout);
      fflush(stderr);
      Int_t buf_readed;

      while (true) { /* read from pipe into buffer */
         buf_readed = read(fStdOutPipe[0], fBuffer.get(), MAX_IO_BUFFER);
         if (buf_readed <= 0) break;
         fStdOut += fBuffer.get();
         memset(fBuffer.get(), 0, MAX_IO_BUFFER + 1);
      }

      while (true) { /* read from pipe into buffer */
         buf_readed = read(fStdErrPipe[0], fBuffer.get(), MAX_IO_BUFFER);
         if (buf_readed <= 0) break;
         fStdErr += fBuffer.get();
         memset(fBuffer.get(), 0, MAX_IO_BUFFER + 1);
      }

      dup2(fSavedStdOut, STDOUT_FILENO);  /* reconnect stdout*/
      dup2(fSavedStdErr, STDERR_FILENO);  /* reconnect stderr*/
      fSyncOutput = false;
   }
}

//______________________________________________________________________________
void TEnvironment::Flush()
{
   if (fSyncOutput) {
      write(fSavedStdOut, fStdOut.Data(), fStdOut.Length());
      write(fSavedStdErr, fStdErr.Data(), fStdErr.Length());
      fsync(fStdOutPipe[0]);
      fsync(fStdErrPipe[0]);
   } else {
      fprintf(stdout, "%s", fStdOut.Data());
      fprintf(stderr, "%s", fStdErr.Data());
      fflush(stdout);
      fflush(stderr);
    }
   ClearBuffers();
}

//______________________________________________________________________________
void TEnvironment::ClearBuffers()
{
   fStdOut = "";
   fStdErr = "";
}

//______________________________________________________________________________
void TEnvironment::SyncOutput(Bool_t status)
{
   fSyncOutput = status;
   InitCapture();
}

//______________________________________________________________________________
void TEnvironment::Init()
{
   MPI_Init(NULL, NULL);
}

//______________________________________________________________________________
Bool_t TEnvironment::IsFinalized()
{
   Int_t t;
   MPI_Finalized(&t);
   return Bool_t(t);
}

//______________________________________________________________________________
Bool_t TEnvironment::IsInitialized()
{
   Int_t t;
   MPI_Initialized(&t);
   return (Bool_t)(t);
}


//______________________________________________________________________________
void TEnvironment::Finalize()
{
   //Finalize the mpi's environment
   MPI_Finalize();
}

//______________________________________________________________________________
TString TEnvironment::GetProcessorName()
{
   Char_t name[MAX_PROCESSOR_NAME];
   Int_t size;
   MPI_Get_processor_name(name, &size);
   return TString(name, size);
}
