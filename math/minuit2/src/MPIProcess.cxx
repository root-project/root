// @(#)root/minuit2:$Id$
// Author: A. Lazzaro 2009
/***************************************************************************
 * Package: Minuit2                                                        *
 *    File: $Id$         *
 *  Author: Alfio Lazzaro, alfio.lazzaro@mi.infn.it                        *
 *                                                                         *
 * Copyright: (C) 2008 by Universita' and INFN, Milan                      *
 ***************************************************************************/

#include "Minuit2/MPIProcess.h"

#include <iostream>

namespace ROOT {

namespace Minuit2 {

   unsigned int MPIProcess::fgGlobalSize = 1;
   unsigned int MPIProcess::fgGlobalRank = 0;

   // By default all procs are for X
   unsigned int MPIProcess::fgCartSizeX = 0;
   unsigned int MPIProcess::fgCartSizeY = 0;
   unsigned int MPIProcess::fgCartDimension = 0;
   bool MPIProcess::fgNewCart = true;

#ifdef MPIPROC
   MPI_Comm MPIProcess::fgCommunicatorWorld = MPI_COMM_NULL;
   MPI_Comm* MPIProcess::fgCommunicator = 0;
   int MPIProcess::fgIndexComm = -1; // -1 for no-initialization
   MPI_Comm MPIProcess::fgCommunicators[2] = {MPI_COMM_NULL};
   unsigned int MPIProcess::fgIndecesComm[2] = {0};
#endif

   MPIProcess::MPIProcess(unsigned int nelements, unsigned int indexComm) :
      fNelements(nelements), fSize(1), fRank(0)
   {

      // check local requested index for communicator, valid values are 0 and 1
      indexComm = (indexComm==0) ? 0 : 1;

#ifdef MPIPROC

      StartMPI();

      if (fgGlobalSize==fgCartDimension &&
          fgCartSizeX!=fgCartDimension && fgCartSizeY!=fgCartDimension) {
         //declare the cartesian topology

         if (fgCommunicator==0 && fgIndexComm<0 && fgNewCart) {
            // first call, declare the topology
            std::cout << "Info --> MPIProcess::MPIProcess: Declare cartesian Topology ("
                      << fgCartSizeX << "x" << fgCartSizeY << ")" << std::endl;

            int color = fgGlobalRank / fgCartSizeY;
            int key = fgGlobalRank % fgCartSizeY;

            MPI_Comm_split(MPI_COMM_WORLD, key, color, &fgCommunicators[0]); // rows for Minuit
            MPI_Comm_split(MPI_COMM_WORLD, color, key, &fgCommunicators[1]); // columns for NLL

            fgNewCart = false;

         }

         fgIndexComm++;

         int comparison = 0;
         MPI_Comm_compare(MPI_COMM_WORLD, *fgCommunicator, &comparison);

         if (fgIndexComm>1 || comparison == MPI_IDENT) { // Remember, no more than 2 dimensions in the topology!
            std::cerr << "Error --> MPIProcess::MPIProcess: Requiring more than 2 dimensions in the topology!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
         }

         // requiring columns as first call. In this case use all nodes
         MPI_Comm_dup(MPI_COMM_WORLD, &fgCommunicatorWorld);
         if (((unsigned int)fgIndexComm)<indexComm)
            fgCommunicator = &fgCommunicatorWorld;
         else {
            fgIndecesComm[fgIndexComm] = indexComm;
            fgCommunicator = &fgCommunicators[fgIndecesComm[fgIndexComm]];
         }

      }
      else {
         // no cartesian topology
         if (fgCartDimension!=0 && fgGlobalSize!=fgCartDimension) {
            std::cout << "Warning --> MPIProcess::MPIProcess: Cartesian dimension doesn't correspond to # total procs!" << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Ignoring topology, use all procs for X." << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Resetting topology..." << std::endl;
            fgCartSizeX = fgGlobalSize;
            fgCartSizeY = 1;
            fgCartDimension = fgGlobalSize;
         }

         if (fgIndexComm<0) {
            if (fgCartSizeX==fgCartDimension) {
               MPI_Comm_dup(MPI_COMM_WORLD, &fgCommunicators[0]);
            }
            else {
               MPI_Comm_dup(MPI_COMM_WORLD, &fgCommunicators[1]);
            }
         }

         fgIndexComm++;

         if (fgIndexComm>1) { // Remember, no more than 2 nested MPI calls!
            std::cerr << "Error --> MPIProcess::MPIProcess: More than 2 nested MPI calls!" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
         }

         fgIndecesComm[fgIndexComm] = indexComm;

         // require 2 nested communicators
         if (fgCommunicator!=0 && fgCommunicators[indexComm]!=MPI_COMM_NULL) {
            std::cout << "Warning --> MPIProcess::MPIProcess: Requiring 2 nested MPI calls!" << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Ignoring second call." << std::endl;
            fgIndecesComm[fgIndexComm] = (indexComm==0) ? 1 : 0;
         }

         fgCommunicator = &fgCommunicators[fgIndecesComm[fgIndexComm]];

      }

      // set size and rank
      if (fgCommunicator!=0) {
         int size, rank;
         MPI_Comm_size(*fgCommunicator, &size);
         MPI_Comm_rank(*fgCommunicator, &rank);
         fSize = (unsigned int) size;
         fRank = (unsigned int) rank;
      }
      else {
         // no MPI calls
         fSize = 1;
         fRank = 0;
      }


      if (fSize>fNelements) {
         std::cerr << "Error --> MPIProcess::MPIProcess: more processors than elements!" << std::endl;
         MPI_Abort(MPI_COMM_WORLD, -1);
      }

#endif

      fNumElements4JobIn = fNelements / fSize;
      fNumElements4JobOut = fNelements % fSize;

   }

   MPIProcess::~MPIProcess()
   {
      // destructor
#ifdef MPIPROC
      fgCommunicator = 0;
      fgIndexComm--;
      if (fgIndexComm==0)
         fgCommunicator = &fgCommunicators[fgIndecesComm[fgIndexComm]];

#endif

   }

   bool MPIProcess::SyncVector(ROOT::Minuit2::MnAlgebraicVector &mnvector)
   {

      // In case of just one job, don't need sync, just go
      if (fSize<2)
         return false;

      if (mnvector.size()!=fNelements) {
         std::cerr << "Error --> MPIProcess::SyncVector: # defined elements different from # requested elements!" << std::endl;
         std::cerr << "Error --> MPIProcess::SyncVector: no MPI syncronization is possible!" << std::endl;
         exit(-1);
      }

#ifdef MPIPROC
      unsigned int numElements4ThisJob = NumElements4Job(fRank);
      unsigned int startElementIndex = StartElementIndex();
      unsigned int endElementIndex = EndElementIndex();

      double dvectorJob[numElements4ThisJob];
      for(unsigned int i = startElementIndex; i<endElementIndex; i++)
         dvectorJob[i-startElementIndex] = mnvector(i);

      double dvector[fNelements];
      MPISyncVector(dvectorJob,numElements4ThisJob,dvector);

      for (unsigned int i = 0; i<fNelements; i++) {
         mnvector(i) = dvector[i];
      }

      return true;

#else

      std::cerr << "Error --> MPIProcess::SyncVector: no MPI syncronization is possible!" << std::endl;
      exit(-1);

#endif

   }


   bool MPIProcess::SyncSymMatrixOffDiagonal(ROOT::Minuit2::MnAlgebraicSymMatrix &mnmatrix)
   {

      // In case of just one job, don't need sync, just go
      if (fSize<2)
         return false;

      if (mnmatrix.size()-mnmatrix.Nrow()!=fNelements) {
         std::cerr << "Error --> MPIProcess::SyncSymMatrixOffDiagonal: # defined elements different from # requested elements!" << std::endl;
         std::cerr << "Error --> MPIProcess::SyncSymMatrixOffDiagonal: no MPI syncronization is possible!" << std::endl;
         exit(-1);
      }

#ifdef MPIPROC
      unsigned int numElements4ThisJob = NumElements4Job(fRank);
      unsigned int startElementIndex = StartElementIndex();
      unsigned int endElementIndex = EndElementIndex();
      unsigned int nrow = mnmatrix.Nrow();

      unsigned int offsetVect = 0;
      for (unsigned int i = 0; i<startElementIndex; i++)
         if ((i+offsetVect)%(nrow-1)==0) offsetVect += (i+offsetVect)/(nrow-1);

      double dvectorJob[numElements4ThisJob];
      for(unsigned int i = startElementIndex; i<endElementIndex; i++) {

         int x = (i+offsetVect)/(nrow-1);
         if ((i+offsetVect)%(nrow-1)==0) offsetVect += x;
         int y = (i+offsetVect)%(nrow-1)+1;

         dvectorJob[i-startElementIndex] = mnmatrix(x,y);

      }

      double dvector[fNelements];
      MPISyncVector(dvectorJob,numElements4ThisJob,dvector);

      offsetVect = 0;
      for (unsigned int i = 0; i<fNelements; i++) {

         int x = (i+offsetVect)/(nrow-1);
         if ((i+offsetVect)%(nrow-1)==0) offsetVect += x;
         int y = (i+offsetVect)%(nrow-1)+1;

         mnmatrix(x,y) = dvector[i];

      }

      return true;

#else

      std::cerr << "Error --> MPIProcess::SyncMatrix: no MPI syncronization is possible!" << std::endl;
      exit(-1);

#endif

   }

#ifdef MPIPROC
   void MPIProcess::MPISyncVector(double *ivector, int svector, double *ovector)
   {
      int offsets[fSize];
      int nconts[fSize];
      nconts[0] = NumElements4Job(0);
      offsets[0] = 0;
      for (unsigned int i = 1; i<fSize; i++) {
         nconts[i] = NumElements4Job(i);
         offsets[i] = nconts[i-1] + offsets[i-1];
      }

      MPI_Allgatherv(ivector, svector, MPI_DOUBLE, ovector, nconts,
                     offsets, MPI_DOUBLE, *fgCommunicator);

   }

   bool MPIProcess::SetCartDimension(unsigned int dimX, unsigned int dimY)
   {
      if (fgCommunicator!=0 || fgIndexComm>=0) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: MPIProcess already declared! Ignoring command..." << std::endl;
         return false;
      }
      if (dimX*dimY<=0) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: Invalid topology! Ignoring command..." << std::endl;
         return false;
      }

      StartMPI();

      if (fgGlobalSize!=dimX*dimY) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: Cartesian dimension doesn't correspond to # total procs!" << std::endl;
         std::cout << "Warning --> MPIProcess::SetCartDimension: Ignoring command..." << std::endl;
         return false;
      }

      if (fgCartSizeX!=dimX || fgCartSizeY!=dimY) {
         fgCartSizeX = dimX; fgCartSizeY = dimY;
         fgCartDimension = fgCartSizeX * fgCartSizeY;
         fgNewCart = true;

      }

      return true;

   }

   bool MPIProcess::SetDoFirstMPICall(bool doFirstMPICall)
   {

      StartMPI();

      bool ret;
      if (doFirstMPICall)
         ret = SetCartDimension(fgGlobalSize,1);
      else
         ret = SetCartDimension(1,fgGlobalSize);

      return ret;

   }

#endif

#ifdef MPIPROC
   MPITerminate dummyMPITerminate = MPITerminate();
#endif

} // namespace Minuit2

} // namespace ROOT
