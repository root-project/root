#include <TMpi.h>
using namespace ROOT::Mpi;

#define count 10
#define root 1
#define source 1
#define dest 1
#define tag 1

void serialization(Bool_t stressTest = kTRUE)
{
   TEnvironment env;                    // environment to start communication system
   TIntraCommunicator comm(COMM_WORLD); // Communicator to send/recv messages

   std::map<std::string, std::string> smap[count]; // std oebjct
   TMatrixD smat[count];                           // ROOT object
   TMpiMessage msgs[count];

   for (auto i = 0; i < count; i++) {
      smap[i]["key"] = "hola";
      smat[i].ResizeTo(count, count);
      smat[i][0][0] = 0.1;
      smat[i][0][1] = 0.2;
      smat[i][1][0] = 0.3;
      smat[i][1][1] = 0.4;
      msgs[i].WriteObject(smat[0]);
   }
   Char_t *buffer;
   Int_t size;
   TCommunicator::Serialize(&buffer, size, smat, count, &comm, dest, source, tag, root);

   TMatrixD umat[count]; // ROOT object

   TCommunicator::Unserialize(buffer, size, umat, count, &comm, dest, source, tag, root); //
   for (auto i = 0; i < count; i++) {
      ROOT_MPI_ASSERT(umat[i][0][0] == smat[i][0][0]);
      ROOT_MPI_ASSERT(umat[i][0][1] == smat[i][0][1]);
      ROOT_MPI_ASSERT(umat[i][1][0] == smat[i][1][0]);
      ROOT_MPI_ASSERT(umat[i][1][1] == smat[i][1][1]);
   }

   ///
   TCommunicator::Serialize(&buffer, size, smap, count, &comm, dest, source, tag, root);
   std::map<std::string, std::string> umap[count];                                        // std oebjct
   TCommunicator::Unserialize(buffer, size, umap, count, &comm, dest, source, tag, root); //
   for (auto i = 0; i < count; i++) {
      ROOT_MPI_ASSERT(umap[i]["key"] == smap[i]["key"]);
   }

   ///
   TCommunicator::Serialize(&buffer, size, msgs, count, &comm, dest, source, tag, root);
   TMpiMessage umsgs[count];                                                               // std oebjct
   TCommunicator::Unserialize(buffer, size, umsgs, count, &comm, dest, source, tag, root); //
   for (auto i = 0; i < count; i++) {
      auto mat = (TMatrixD *)umsgs[i].ReadObjectAny(gROOT->GetClass(typeid(TMatrixD)));
      if (mat == NULL)
         comm.Abort(ERR_BUFFER);
      ROOT_MPI_ASSERT((*mat)[0][0] == smat[i][0][0]);
      ROOT_MPI_ASSERT((*mat)[0][1] == smat[i][0][1]);
      ROOT_MPI_ASSERT((*mat)[1][0] == smat[i][1][0]);
      ROOT_MPI_ASSERT((*mat)[1][1] == smat[i][1][1]);
   }
}
