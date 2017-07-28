#include <cassert>
using namespace ROOT::Mpi;
#define debug std::cout << __LINE__ << std::endl;

void communicators()
{
   TEnvironment env; // environment to start communication system

   if (COMM_WORLD.GetSize() != 4) {
      Error(__FUNCTION__, "needs 4 proccessors");
      COMM_WORLD.Abort(1); // needs 4 ranks
   }
   TNullCommunicator ncomm;              // null comm
   TIntraCommunicator comm;              // default must be null
   TInterCommunicator itcomm;            // default must be null
   TIntraCommunicator gcomm(COMM_WORLD); // default

   assert(ncomm == comm);
   assert(ncomm == COMM_NULL);
   assert(comm == COMM_NULL);
   assert(gcomm == COMM_WORLD);
   assert(itcomm == COMM_NULL);

   comm = gcomm;
   assert(comm == COMM_WORLD);
   assert(comm.IsInter() == kFALSE); // is an intra comm not an inter comm

   TGroup group = comm.GetGroup();

   Int_t n = 2;
   const Int_t ranks[2] = {0, 2};

   // pair ranks
   auto pgroup = group.Include(2, ranks);
   assert(pgroup != GROUP_NULL);

   // odd ranks
   auto ogroup = group.Exclude(2, ranks);
   assert(ogroup != GROUP_NULL);

   auto rank = comm.GetRank();

   if (rank % 2 == 0) {
      auto pcomm = comm.Create(pgroup); // Internal inter comm that only works in pair ranks

      assert(pcomm != COMM_NULL);
      assert(pcomm.IsInter() == kFALSE); // is an intra comm not an inter comm

      // performing a allreduce for pair group
      Int_t results = 0;
      auto lrank = pcomm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 1);

      lrank = comm.GetRank();
      pcomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 2);

   } else {
      auto ocomm = comm.Create(ogroup);
      assert(ocomm != COMM_NULL);
      assert(ocomm.IsInter() == kFALSE); // is an intra comm not an inter comm

      // performing a allreduce for odd group
      Int_t results = 0;
      auto lrank = ocomm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 1);

      lrank = comm.GetRank();
      ocomm.AllReduce(&lrank, &results, 1, SUM);
      assert(results == 4);
   }
   //////////////////////////////
   // testing TInterCommunicator//
   //////////////////////////////
   auto key = comm.GetRank();
   auto color = comm.GetRank() % 2;
   gcomm = comm.Split(color, key);

   assert(gcomm != COMM_NULL);
   assert(gcomm.IsInter() == kFALSE); // is an intra comm not and inter comm

   // local
   auto icomm = gcomm.CreateIntercomm(0, comm, 1 - color, 0); // creating an intercomm to communicate with odd ranks

   assert(icomm != COMM_NULL);
   assert(icomm.IsInter() == kTRUE); // is an intra comm not and inter comm

   // std::cout<<icomm.GetRank()<<std::endl;
   // std::cout<<icomm.GetSize()<<std::endl;

   if (rank % 2 == 0) { // pair ranks
      assert(icomm.GetRank() == 0 || icomm.GetRank() == 1);
      assert(icomm.GetSize() == 2);

      assert(icomm.GetRemoteSize() == 2);

      if (icomm.GetRank() == 0)
         icomm.Send(comm.GetSize(), 1, 12); // sending value to the second gruop

   } else { // odd ranks
      assert(icomm.GetRank() == 0 || icomm.GetRank() == 1);
      assert(icomm.GetSize() == 2);
      assert(icomm.GetRemoteSize() == 2);

      Int_t value;
      if (icomm.GetRank() == 1) {
         icomm.Recv(value, 0, 12); // recv value from the first gruop
         assert(value == comm.GetSize());
      }
   }

   auto intracomm = icomm.Merge(rank % 2); // mergin first the pairs ranks must be {0,2,1,3}

   assert(intracomm != COMM_NULL);
   assert(intracomm.GetSize() == 4); // must be 4 (the two groups together)

   // testing merge the two groups in pair group
   if (rank % 2 == 0) {
      assert(intracomm.GetRank() == 0 ||
             intracomm.GetRank() == 1); // the ranks pairs must have 0 and 1 ranks respect a COMM_WORLD
   } else {
      assert(intracomm.GetRank() == 2 || intracomm.GetRank() == 3);
   }

   intracomm = icomm.Merge(
      rank % 2 == 0 ? 1 : 0); // mergin first the odd ranks must be {2,0,3,1} NOTE: in pair positions odd ranks

   assert(intracomm != COMM_NULL);
   assert(intracomm.GetSize() == 4); // must be 4 (the two groups together)

   // testing merge the two groups in pair group
   if (rank % 2 == 0) {
      assert(intracomm.GetRank() == 2 ||
             intracomm.GetRank() == 3); // the ranks pairs must have 3 and 2 ranks respect a COMM_WORLD
   } else {
      assert(intracomm.GetRank() == 0 || intracomm.GetRank() == 1);
   }

   // clening the objects
   icomm.Free();
   pgroup.Free();
   ogroup.Free();

   assert(icomm == COMM_NULL);
   assert(pgroup == GROUP_NULL);
   assert(ogroup == GROUP_NULL);

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
   assert(cart2x2 == COMM_NULL);
   assert(cart2x2.getDim() == 2);
   cart2x2.GetCoords(cart2x2.GetRank(), 2, coord);
   if (cart2x2.GetRank() == 0) {
      assert(coord[0] == 0);
      assert(coord[1] == 0);
   }
   if (cart2x2.GetRank() == 1) {
      assert(coord[0] == 0);
      assert(coord[1] == 1);
   }
   if (cart2x2.GetRank() == 2) {
      assert(coord[0] == 1);
      assert(coord[1] == 0);
   }
   if (cart2x2.GetRank() == 3) {
      assert(coord[0] == 1);
      assert(coord[1] == 1);
   }
   // get rank associated to the coords
   assert(cart2x2.GetCartRank({0, 0}) == 0);
   assert(cart2x2.GetCartRank({0, 1}) == 1);
   assert(cart2x2.GetCartRank({1, 0}) == 2);
   assert(cart2x2.GetCartRank({1, 1}) == 3);
   //    std::cout<<"rank = "<<cart2x2.GetRank()<<" "<<coord[0]<<" "<<coord[1]<<std::endl;
   //    if(cart2x2.GetRank() == 0) std::cout<<"rank = "<<cart2x2.GetCartRank({0,0})<<" coords[0][0] \n";
   //    if(cart2x2.GetRank() == 1) std::cout<<"rank = "<<cart2x2.GetCartRank({0,1})<<" coords[0][1] \n";
   //    if(cart2x2.GetRank() == 2) std::cout<<"rank = "<<cart2x2.GetCartRank({1,0})<<" coords[1][0] \n";
   //    if(cart2x2.GetRank() == 3) std::cout<<"rank = "<<cart2x2.GetCartRank({1,1})<<" coords[1][1] \n";

   dim[0] = 4; // four components in x
   dim[1] = 1; // one component in y
   auto cart4x1 = COMM_WORLD.CreateCartcomm(2, dim, period, reorder);
   assert(cart4x1 == COMM_NULL);
   assert(cart4x1.getDim() == 2);
   cart4x1.GetCoords(cart4x1.GetRank(), 2, coord);
   if (cart4x1.GetRank() == 0) {
      assert(coord[0] == 0);
      assert(coord[1] == 0);
   }
   if (cart4x1.GetRank() == 1) {
      assert(coord[0] == 0);
      assert(coord[1] == 1);
   }
   if (cart4x1.GetRank() == 2) {
      assert(coord[0] == 0);
      assert(coord[1] == 2);
   }
   if (cart4x1.GetRank() == 3) {
      assert(coord[0] == 0);
      assert(coord[1] == 3);
   }
   //    std::cout<<"rank = "<<cart4x1.GetRank()<<" "<<coord[0]<<" "<<coord[1]<<std::endl;

   // test shift
   if (cart4x1.GetRank() == 1) { // i am in (0,1) I will to see left and  right ranks that must be left 0 right 2
      Int_t left, right;
      cart4x1.Shift(0, 1, left, right);
      assert(left == 0);
      assert(right == 2);
      //     std::cout<<"rank left= "<<left<<" rank right= "<<right<<" "<<std::endl;
   }

   dim[0] = 1; // one components in x
   dim[1] = 4; // four component in y
   auto cart1x4 = COMM_WORLD.CreateCartcomm(2, dim, period, reorder);
   assert(cart1x4 == COMM_NULL);
   assert(cart1x4.getDim() == 2);
   cart1x4.GetCoords(cart1x4.GetRank(), 2, coord);
   if (cart1x4.GetRank() == 0) {
      assert(coord[0] == 0);
      assert(coord[1] == 0);
   }
   if (cart1x4.GetRank() == 1) {
      assert(coord[0] == 1);
      assert(coord[1] == 0);
   }
   if (cart1x4.GetRank() == 2) {
      assert(coord[0] == 2);
      assert(coord[1] == 0);
   }
   if (cart1x4.GetRank() == 3) {
      assert(coord[0] == 3);
      assert(coord[1] == 0);
   }
   std::cout << "rank = " << cart1x4.GetRank() << " " << coord[0] << " " << coord[1] << std::endl;

   // test shift
   if (cart1x4.GetRank() == 1) { // i am in (0,1) I will to see up and  down ranks that must be up 0 down 2
      Int_t up, down;
      cart1x4.Shift(1, 1, up, down);
      assert(up == 0);
      assert(down == 2);
      std::cout << "rank up= " << up << " rank down= " << down << " " << std::endl;
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

   rank = graph.GetRank();
   Int_t maxneighbors = 2;
   Int_t neighbors[maxneighbors];
   if (rank == 0) {
      assert(graph.GetNeighborsCount(rank) == 1);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      assert(neighbors[0] == 1);
      std::cout << "neighbor rank 0 = " << neighbors[0] << std::endl;
   }
   if (rank == 1) {
      assert(graph.GetNeighborsCount(1) == 2);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      assert(neighbors[0] == 0);
      assert(neighbors[1] == 2);
      std::cout << "neighbors rank 1 = " << neighbors[0] << " - " << neighbors[1] << std::endl;
   }
   if (rank == 2) {
      assert(graph.GetNeighborsCount(2) == 2);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      assert(neighbors[0] == 1);
      assert(neighbors[1] == 3);
      std::cout << "neighbors rank 2 = " << neighbors[0] << " - " << neighbors[1] << std::endl;
   }
   if (graph.GetRank() == 3) {
      assert(graph.GetNeighborsCount(3) == 1);
      graph.GetNeighbors(rank, maxneighbors, neighbors);
      assert(neighbors[0] == 2);
      std::cout << "neighbor rank 3 = " << neighbors[0] << std::endl;
   }
}
