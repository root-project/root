#include <cassert>
using namespace ROOT::Mpi;
#define debug std::cout << __LINE__ << std::endl;

void communicators()
{
   TEnvironment env; // environment to start communication system

   env.SetVerbose();

   if (COMM_WORLD.GetSize() != 4) {
      Error(__FUNCTION__, "needs 4 proccessors");
      COMM_WORLD.Abort(1); // needs 4 ranks
   }
   TNullCommunicator ncomm;              // null comm
   TIntraCommunicator comm;              // default must be null
   TInterCommunicator itcomm;            // default must be null
   TIntraCommunicator gcomm(COMM_WORLD); // default

   ROOT_MPI_ASSERT(ncomm == comm);
   ROOT_MPI_ASSERT(ncomm == COMM_NULL);
   ROOT_MPI_ASSERT(comm == COMM_NULL);
   ROOT_MPI_CHECK_COMM(gcomm, &COMM_WORLD);

   ROOT_MPI_ASSERT(itcomm == COMM_NULL);

   comm = gcomm;
   ROOT_MPI_ASSERT(comm == COMM_WORLD);
   ROOT_MPI_ASSERT(comm.IsInter() == kFALSE); // is an intra comm not an inter comm

   TGroup group = comm.GetGroup();

   Int_t n = 2;
   const Int_t ranks[2] = {0, 2};

   // pair ranks
   auto pgroup = group.Include(2, ranks);
   ROOT_MPI_ASSERT(pgroup != GROUP_NULL, &COMM_WORLD);

   // odd ranks
   auto ogroup = group.Exclude(2, ranks);
   ROOT_MPI_ASSERT(ogroup != GROUP_NULL, &COMM_WORLD);

   auto rank = comm.GetRank();

   if (rank % 2 == 0) {
      auto pcomm = comm.Create(pgroup); // Internal inter comm that only works in pair ranks

      ROOT_MPI_CHECK_COMM(pcomm, &COMM_WORLD);
      ROOT_MPI_ASSERT(pcomm.IsInter() == kFALSE, &pcomm); // is an intra comm not an inter comm

      // performing a allreduce for pair group
      Int_t results = 0;
      auto lrank = pcomm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      ROOT_MPI_ASSERT(results == 1, &pcomm);

      lrank = comm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      ROOT_MPI_ASSERT(results == 2, &pcomm);

   } else {
      auto ocomm = comm.Create(ogroup);
      ROOT_MPI_CHECK_COMM(ocomm, &COMM_WORLD);
      ROOT_MPI_ASSERT(ocomm.IsInter() == kFALSE, &ocomm); // is an intra comm not an inter comm

      // performing a allreduce for odd group
      Int_t results = 0;
      auto lrank = ocomm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      ROOT_MPI_ASSERT(results == 1, &ocomm);

      lrank = comm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      ROOT_MPI_ASSERT(results == 4, &ocomm);
   }
   //////////////////////////////
   // testing TInterCommunicator//
   //////////////////////////////
   auto key = comm.GetRank();
   auto color = comm.GetRank() % 2;
   gcomm = comm.Split(color, key);

   ROOT_MPI_CHECK_COMM(gcomm, &COMM_WORLD);
   ROOT_MPI_ASSERT(gcomm.IsInter() == kFALSE, &gcomm); // is an intra comm not and inter comm

   // local
   auto icomm = gcomm.CreateIntercomm(0, comm, 1 - color, 0); // creating an intercomm to communicate with odd ranks

   ROOT_MPI_CHECK_COMM(icomm, &COMM_WORLD);
   ROOT_MPI_ASSERT(icomm.IsInter() == kTRUE, &icomm); // is an intra comm not and inter comm

   // std::cout<<icomm.GetRank()<<std::endl;
   // std::cout<<icomm.GetSize()<<std::endl;

   if (rank % 2 == 0) { // pair ranks
      ROOT_MPI_ASSERT(icomm.GetRank() == 0 || icomm.GetRank() == 1, &icomm);
      ROOT_MPI_ASSERT(icomm.GetSize() == 2, &icomm);

      ROOT_MPI_ASSERT(icomm.GetRemoteSize() == 2, &icomm);

      if (icomm.GetRank() == 0)
         icomm.Send(comm.GetSize(), 1, 12); // sending value to the second gruop

   } else { // odd ranks
      ROOT_MPI_ASSERT(icomm.GetRank() == 0 || icomm.GetRank() == 1, &icomm);
      ROOT_MPI_ASSERT(icomm.GetSize() == 2, &icomm);
      ROOT_MPI_ASSERT(icomm.GetRemoteSize() == 2, &icomm);

      Int_t value;
      if (icomm.GetRank() == 1) {
         icomm.Recv(value, 0, 12); // recv value from the first gruop
         ROOT_MPI_ASSERT(value == comm.GetSize(), &icomm);
      }
   }

   auto intracomm = icomm.Merge(rank % 2); // mergin first the pairs ranks must be {0,2,1,3}

   ROOT_MPI_CHECK_COMM(intracomm, &COMM_WORLD);
   ROOT_MPI_ASSERT(intracomm.GetSize() == 4, &intracomm); // must be 4 (the two groups together)

   // testing merge the two groups in pair group
   if (rank % 2 == 0) {
      ROOT_MPI_ASSERT(intracomm.GetRank() == 0 ||
                      intracomm.GetRank() == 1); // the ranks pairs must have 0 and 1 ranks respect a COMM_WORLD
   } else {
      ROOT_MPI_ASSERT(intracomm.GetRank() == 2 || intracomm.GetRank() == 3);
   }

   intracomm = icomm.Merge(
      rank % 2 == 0 ? 1 : 0); // mergin first the odd ranks must be {2,0,3,1} NOTE: in pair positions odd ranks

   ROOT_MPI_CHECK_COMM(intracomm, &COMM_WORLD);
   ROOT_MPI_ASSERT(intracomm.GetSize() == 4, &intracomm); // must be 4 (the two groups together)

   // testing merge the two groups in pair group
   if (rank % 2 == 0) {
      ROOT_MPI_ASSERT(intracomm.GetRank() == 2 || intracomm.GetRank() == 3,
                      &intracomm); // the ranks pairs must have 3 and 2 ranks respect a COMM_WORLD
   } else {
      ROOT_MPI_ASSERT(intracomm.GetRank() == 0 || intracomm.GetRank() == 1, &intracomm);
   }

   // clening the objects
   icomm.Free();
   pgroup.Free();
   ogroup.Free();

   ROOT_MPI_ASSERT(icomm == COMM_NULL, &COMM_WORLD);

   ROOT_MPI_ASSERT(pgroup == GROUP_NULL, &COMM_WORLD);
   ROOT_MPI_ASSERT(ogroup == GROUP_NULL, &COMM_WORLD);

   //////////////////////////////
   // testing TCartCommunicator//
   //////////////////////////////
   // creating a 2x2 grid
   Int_t dim[2], reorder;
   Bool_t period[2];
   Int_t coord[2], id;

   dim[0] = 2;
   dim[1] = 2;
   period[0] = 1;
   period[1] = 1;
   reorder = 0;

   auto cart2x2 = COMM_WORLD.CreateCartcomm(2, dim, period, reorder);

   ROOT_MPI_CHECK_COMM(cart2x2, &COMM_WORLD);

   ROOT_MPI_ASSERT(cart2x2.GetDim() == 2, &cart2x2);
   cart2x2.GetCoords(cart2x2.GetRank(), 2, coord);
   if (cart2x2.GetRank() == 0) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart2x2);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart2x2);
   }
   if (cart2x2.GetRank() == 1) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart2x2);
      ROOT_MPI_ASSERT(coord[1] == 1, &cart2x2);
   }
   if (cart2x2.GetRank() == 2) {
      ROOT_MPI_ASSERT(coord[0] == 1, &cart2x2);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart2x2);
   }
   if (cart2x2.GetRank() == 3) {
      ROOT_MPI_ASSERT(coord[0] == 1, &cart2x2);
      ROOT_MPI_ASSERT(coord[1] == 1, &cart2x2);
   }
   // get rank associated to the coords
   ROOT_MPI_ASSERT(cart2x2.GetCartRank({0, 0}) == 0, &cart2x2);
   ROOT_MPI_ASSERT(cart2x2.GetCartRank({0, 1}) == 1, &cart2x2);
   ROOT_MPI_ASSERT(cart2x2.GetCartRank({1, 0}) == 2, &cart2x2);
   ROOT_MPI_ASSERT(cart2x2.GetCartRank({1, 1}) == 3, &cart2x2);
   //    std::cout<<"rank = "<<cart2x2.GetRank()<<" "<<coord[0]<<" "<<coord[1]<<std::endl;
   //    if(cart2x2.GetRank() == 0) std::cout<<"rank = "<<cart2x2.GetCartRank({0,0})<<" coords[0][0] \n";
   //    if(cart2x2.GetRank() == 1) std::cout<<"rank = "<<cart2x2.GetCartRank({0,1})<<" coords[0][1] \n";
   //    if(cart2x2.GetRank() == 2) std::cout<<"rank = "<<cart2x2.GetCartRank({1,0})<<" coords[1][0] \n";
   //    if(cart2x2.GetRank() == 3) std::cout<<"rank = "<<cart2x2.GetCartRank({1,1})<<" coords[1][1] \n";

   dim[0] = 4; // four components in x
   dim[1] = 1; // one component in y
   auto cart4x1 = COMM_WORLD.CreateCartcomm(2, dim, period, reorder);
   ROOT_MPI_CHECK_COMM(cart4x1, &COMM_WORLD);
   ROOT_MPI_ASSERT(cart4x1.GetDim() == 2, &cart4x1);
   cart4x1.GetCoords(cart4x1.GetRank(), 2, coord);
   if (cart4x1.GetRank() == 0) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart4x1);
   }
   if (cart4x1.GetRank() == 1) {
      ROOT_MPI_ASSERT(coord[0] == 1, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart4x1);
   }
   if (cart4x1.GetRank() == 2) {
      ROOT_MPI_ASSERT(coord[0] == 2, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart4x1);
   }
   if (cart4x1.GetRank() == 3) {
      ROOT_MPI_ASSERT(coord[0] == 3, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart4x1);
   }

   //    std::cout<<"rank = "<<cart4x1.GetRank()<<" "<<coord[0]<<" "<<coord[1]<<std::endl;

   // test shift
   if (cart4x1.GetRank() == 1) { // i am in (0,1) I will to see left and  right ranks that must be left 0 right 2
      Int_t left, right;
      cart4x1.Shift(0, 1, left, right);
      ROOT_MPI_ASSERT(left == 0, &cart4x1);
      ROOT_MPI_ASSERT(right == 2, &cart4x1);
      //     std::cout<<"rank left= "<<left<<" rank right= "<<right<<" "<<std::endl;
   }

   dim[0] = 1; // one components in x
   dim[1] = 4; // four component in y
   auto cart1x4 = COMM_WORLD.CreateCartcomm(2, dim, period, reorder);
   ROOT_MPI_CHECK_COMM(cart1x4, &COMM_WORLD);

   ROOT_MPI_ASSERT(cart1x4.GetDim() == 2, &cart4x1);
   cart1x4.GetCoords(cart1x4.GetRank(), 2, coord);
   if (cart1x4.GetRank() == 0) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 0, &cart4x1);
   }
   if (cart1x4.GetRank() == 1) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 1, &cart4x1);
   }
   if (cart1x4.GetRank() == 2) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 2, &cart4x1);
   }
   if (cart1x4.GetRank() == 3) {
      ROOT_MPI_ASSERT(coord[0] == 0, &cart4x1);
      ROOT_MPI_ASSERT(coord[1] == 3, &cart4x1);
   }
   //    std::cout << "rank = " << cart1x4.GetRank() << " " << coord[0] << " " << coord[1] << std::endl;

   // test shift
   if (cart1x4.GetRank() == 1) { // i am in (0,1) I will to see up and  down ranks that must be up 0 down 2
      Int_t up, down;
      cart1x4.Shift(1, 1, up, down);
      ROOT_MPI_ASSERT(up == 0, &cart4x1);
      ROOT_MPI_ASSERT(down == 2, &cart4x1);
      //       std::cout << "rank up= " << up << " rank down= " << down << " " << std::endl;
   }

   // architecture of the graph
   //   0   3
   //  /     \
   // 1  <->  2
   // node     nneighbors    index      adges
   //  0          1            1          1
   //  1          2            3          0,2
   //  2          2            5          1,3
   //  3          1            6          2
   Int_t nnodes = 4;                    /* number of nodes */
   Int_t index[4] = {1, 3, 5, 6};       /* index definition */
   Int_t edges[6] = {1, 0, 2, 1, 3, 2}; /* edges definition */
   reorder = 1;                         /* allows processes reordered for efficiency */
   auto graph = COMM_WORLD.CreateGraphcomm(nnodes, index, edges, reorder);

   ROOT_MPI_CHECK_COMM(graph, &COMM_WORLD);

   rank = graph.GetRank();
   Int_t maxneighbors = 2;
   Int_t neighbors[maxneighbors];
   if (rank == 0) {
      ROOT_MPI_ASSERT(graph.GetNeighborsCount(rank) == 1, &graph);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      ROOT_MPI_ASSERT(neighbors[0] == 1, &graph);
      //       std::cout << "neighbor rank 0 = " << neighbors[0] << std::endl;
   }
   if (rank == 1) {
      ROOT_MPI_ASSERT(graph.GetNeighborsCount(1) == 2, &graph);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      ROOT_MPI_ASSERT(neighbors[0] == 0, &graph);
      ROOT_MPI_ASSERT(neighbors[1] == 2, &graph);
      //       std::cout << "neighbors rank 1 = " << neighbors[0] << " - " << neighbors[1] << std::endl;
   }
   if (rank == 2) {
      ROOT_MPI_ASSERT(graph.GetNeighborsCount(2) == 2, &graph);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      ROOT_MPI_ASSERT(neighbors[0] == 1, &graph);
      ROOT_MPI_ASSERT(neighbors[1] == 3, &graph);
      //       std::cout << "neighbors rank 2 = " << neighbors[0] << " - " << neighbors[1] << std::endl;
   }
   if (graph.GetRank() == 3) {
      ROOT_MPI_ASSERT(graph.GetNeighborsCount(3) == 1, &graph);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      ROOT_MPI_ASSERT(neighbors[0] == 2, &graph);
      //       std::cout << "neighbor rank 3 = " << neighbors[0] << std::endl;
   }
}
