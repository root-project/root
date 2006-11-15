// @(#)root/tmva $Id: NodeID.h,v 1.6 2006/10/10 08:31:00 rdm Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class:   NodeID                                                                *
 *                                                                                *
 * Description:                                                                   *
 *      Node identification (NodeID) for the BinarySearch or Decision Trees nodes *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany                                                *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 **********************************************************************************/

#ifndef ROOT_TMVA_NodeID
#define ROOT_TMVA_NodeID

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA::NodeID                                                         //
//                                                                      //
// Node identifiaction (NodeID) for the BinarySearch or Decision Trees  //
//  nodes, needed for the recursive reading of the tree from a text file//
//  it is currently NOT a UNIQUE ID, ..                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Riosfwd
#include "Riosfwd.h"
#endif

namespace TMVA {

   // a class used to identify a Node; (needed for recursive reading from text file)
   // (currently it is NOT UNIQUE... but could eventually made it

   class NodeID {

   public:

      // Node constructor
      NodeID( Int_t d=0, std::string p="Root" ) : fDepth( d ), fPos( p ) {}
      // Node destructor
      ~NodeID() {}

      // Set depth, layer of the where the node is within the tree, seen from the top (root)
      void SetDepth(const Int_t d){fDepth=d;}

      //Return depth, layer of the where the node is within the tree, seen from the top (root)
      Int_t GetDepth() const {return fDepth;}

      //set node position, i.e, the node is a left (l) or right (r) daugther
      void SetPos(const std::string s) {fPos=s;}

      //Return the node position, i.e, the node is a left (l) or right (r) daugther
      std::string GetPos() const {return fPos;}

   private:

      Int_t        fDepth; // depth of the node within the tree (seen from root node)
      std::string  fPos;   // position, i.e. it is a left (l) or right (r) daughter

   };

}

#endif

