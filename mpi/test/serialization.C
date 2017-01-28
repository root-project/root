#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

#define count 10
#define root 1
#define source 1
#define dest 1
#define tag 1

void serialization(Bool_t stressTest = kTRUE)
{
   TEnvironment env;          //environment to start communication system
   TCommunicator comm;   //Communicator to send/recv messages


   std::map<std::string, std::string> smap[count]; //std oebjct
   TMatrixD smat[count];                           //ROOT object

   for (auto i = 0; i < count; i++) {
      smap[i]["key"] = "hola";
      smat[i].ResizeTo(count, count);
      smat[i][0][0] = 0.1;
      smat[i][0][1] = 0.2;
      smat[i][1][0] = 0.3;
      smat[i][1][1] = 0.4;
   }
   Char_t *buffer;
   Int_t size;
   Serialize(&buffer, size, smat, count, &comm, dest, source, tag, root);

   TMatrixD umat[count];                           //ROOT object

   Unserialize(buffer, size, umat, count, &comm, dest, source, tag, root);//
   for (auto i = 0; i < count; i++) {
      assert(umat[i][0][0] == smat[i][0][0]);
      assert(umat[i][0][1] == smat[i][0][1]);
      assert(umat[i][1][0] == smat[i][1][0]);
      assert(umat[i][1][1] == smat[i][1][1]);
   }

   Serialize(&buffer, size, smap, count, &comm, dest, source, tag, root);
   std::map<std::string, std::string> umap[count]; //std oebjct
   Unserialize(buffer, size, umap, count, &comm, dest, source, tag, root);//
   for (auto i = 0; i < count; i++) {
      assert(umap[i]["key"] == smap[i]["key"]);
   }

}

