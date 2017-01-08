// @(#)root/mpi:$Id: LinkDef.h  -- :: $

/*************************************************************************
 * Copyright (C) 2016, Omar Andres Zapata Mesa           .               *
 * Omar.Zapata@cern.ch   http://oproject.org             .               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedclass;
#pragma link C++ nestedtypedef;

/*
 * Some raw MPI datatypes
 */
#pragma link C++ class MPI::Status+;
#pragma link C++ class MPI_Status+;
#pragma link C++ class MPI::Request+;
#pragma link C++ class MPI_Request+;

/*
 * ROOTMpi datatypes
 */
#pragma link C++ class ROOT::Mpi::TMpiMessage+;
#pragma link C++ class ROOT::Mpi::TMpiMessageInfo+;
#pragma link C++ class ROOT::Mpi::TStatus+;
#pragma link C++ class ROOT::Mpi::TRequest+;
#pragma link C++ class ROOT::Mpi::TEnvironment;
#pragma link C++ class ROOT::Mpi::TCommunicator;

/*
 * Global communicator
 */
// #pragma link C++ global ROOT::Mpi::COMM_WORLD;

#ifdef USE_FOR_AUTLOADING
#pragma link C++ class ROOT::Mpi;
#endif



#endif
