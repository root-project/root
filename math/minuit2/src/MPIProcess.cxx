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

   unsigned int MPIProcess::fGlobalSize = 1;
   unsigned int MPIProcess::fGlobalRank = 0;

   // By default all procs are for X
   unsigned int MPIProcess::fCartSizeX = 0;
   unsigned int MPIProcess::fCartSizeY = 0;
   unsigned int MPIProcess::fCartDimension = 0;
   bool MPIProcess::fNewCart = true;

#ifdef MPIPROC
   MPI::Intracomm* MPIProcess::fCommunicator = 0;
   int MPIProcess::fIndexComm = -1; // -1 for no-initialization
   MPI::Intracomm* MPIProcess::fCommunicators[2] = {0};
   unsigned int MPIProcess::fIndecesComm[2] = {0};
#endif

   MPIProcess::MPIProcess(unsigned int nelements, unsigned int indexComm) :
      fNelements(nelements), fSize(1), fRank(0)
   {

      // check local requested index for communicator, valid values are 0 and 1
      indexComm = (indexComm==0) ? 0 : 1;

#ifdef MPIPROC

      StartMPI();

      if (fGlobalSize==fCartDimension && 
          fCartSizeX!=fCartDimension && fCartSizeY!=fCartDimension) {
         //declare the cartesian topology

         if (fCommunicator==0 && fIndexComm<0 && fNewCart) {
            // first call, declare the topology
            std::cout << "Info --> MPIProcess::MPIProcess: Declare cartesian Topology (" 
                      << fCartSizeX << "x" << fCartSizeY << ")" << std::endl;
      
            int color = fGlobalRank / fCartSizeY;
            int key = fGlobalRank % fCartSizeY;

            fCommunicators[0] = new MPI::Intracomm(MPI::COMM_WORLD.Split(key,color)); // rows for Minuit
            fCommunicators[1] = new MPI::Intracomm(MPI::COMM_WORLD.Split(color,key)); // columns for NLL

            fNewCart = false;

         }

         fIndexComm++;

         if (fIndexComm>1 || fCommunicator==(&(MPI::COMM_WORLD))) { // Remember, no more than 2 dimensions in the topology!
            std::cerr << "Error --> MPIProcess::MPIProcess: Requiring more than 2 dimensions in the topology!" << std::endl;
            MPI::COMM_WORLD.Abort(-1);
         } 

         // requiring columns as first call. In this case use all nodes
         if (((unsigned int)fIndexComm)<indexComm)
            fCommunicator = &(MPI::COMM_WORLD);
         else {
            fIndecesComm[fIndexComm] = indexComm;
            fCommunicator = fCommunicators[fIndecesComm[fIndexComm]];
         }

      }
      else {
         // no cartesian topology
         if (fCartDimension!=0 && fGlobalSize!=fCartDimension) {
            std::cout << "Warning --> MPIProcess::MPIProcess: Cartesian dimension doesn't correspond to # total procs!" << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Ignoring topology, use all procs for X." << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Resetting topology..." << std::endl;
            fCartSizeX = fGlobalSize;
            fCartSizeY = 1;
            fCartDimension = fGlobalSize;
         }

         if (fIndexComm<0) {
            if (fCartSizeX==fCartDimension) {
               fCommunicators[0] = &(MPI::COMM_WORLD);
               fCommunicators[1] = 0;
            }
            else {
               fCommunicators[0] = 0;
               fCommunicators[1] = &(MPI::COMM_WORLD);
            }
         }

         fIndexComm++;

         if (fIndexComm>1) { // Remember, no more than 2 nested MPI calls!
            std::cerr << "Error --> MPIProcess::MPIProcess: More than 2 nested MPI calls!" << std::endl;
            MPI::COMM_WORLD.Abort(-1);
         }

         fIndecesComm[fIndexComm] = indexComm;

         // require 2 nested communicators
         if (fCommunicator!=0 && fCommunicators[indexComm]!=0) {
            std::cout << "Warning --> MPIProcess::MPIProcess: Requiring 2 nested MPI calls!" << std::endl;
            std::cout << "Warning --> MPIProcess::MPIProcess: Ignoring second call." << std::endl;
            fIndecesComm[fIndexComm] = (indexComm==0) ? 1 : 0;
         } 

         fCommunicator = fCommunicators[fIndecesComm[fIndexComm]];

      }

      // set size and rank
      if (fCommunicator!=0) {
         fSize = fCommunicator->Get_size();
         fRank = fCommunicator->Get_rank();
      }
      else {
         // no MPI calls
         fSize = 1;
         fRank = 0;
      }
    

      if (fSize>fNelements) {
         std::cerr << "Error --> MPIProcess::MPIProcess: more processors than elements!" << std::endl;
         MPI::COMM_WORLD.Abort(-1);
      }

#endif

      fNumElements4JobIn = fNelements / fSize;
      fNumElements4JobOut = fNelements % fSize;

   }

   MPIProcess::~MPIProcess()
   {
#ifdef MPIPROC  
      fCommunicator = 0;
      fIndexComm--; 
      if (fIndexComm==0)
         fCommunicator = fCommunicators[fIndecesComm[fIndexComm]];
    
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

      fCommunicator->Allgatherv(ivector,svector,MPI::DOUBLE,
                                ovector,nconts,offsets,MPI::DOUBLE); 
    
   }

   bool MPIProcess::SetCartDimension(unsigned int dimX, unsigned int dimY)  
   {
      if (fCommunicator!=0 || fIndexComm>=0) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: MPIProcess already declared! Ignoring command..." << std::endl;
         return false;
      }
      if (dimX*dimY<=0) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: Invalid topology! Ignoring command..." << std::endl;
         return false;
      }

      StartMPI();

      if (fGlobalSize!=dimX*dimY) {
         std::cout << "Warning --> MPIProcess::SetCartDimension: Cartesian dimension doesn't correspond to # total procs!" << std::endl;
         std::cout << "Warning --> MPIProcess::SetCartDimension: Ignoring command..." << std::endl;
         return false;
      }

      if (fCartSizeX!=dimX || fCartSizeY!=dimY) {
         fCartSizeX = dimX; fCartSizeY = dimY; 
         fCartDimension = fCartSizeX * fCartSizeY;
         fNewCart = true;

         if (fCommunicators[0]!=0 && fCommunicators[1]!=0) {
            delete fCommunicators[0]; fCommunicators[0] = 0; fIndecesComm[0] = 0;
            delete fCommunicators[1]; fCommunicators[1] = 0; fIndecesComm[1] = 0;
         }
      }

      return true;

   }

   bool MPIProcess::SetDoFirstMPICall(bool doFirstMPICall)
   { 

      StartMPI();

      bool ret;
      if (doFirstMPICall)
         ret = SetCartDimension(fGlobalSize,1);
      else
         ret = SetCartDimension(1,fGlobalSize);

      return ret;

   }

#endif

#ifdef MPIPROC
   MPITerminate dummyMPITerminate = MPITerminate();
#endif

} // namespace ROOT
