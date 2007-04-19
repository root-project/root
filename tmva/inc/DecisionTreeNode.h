// @(#)root/tmva $Id: DecisionTreeNode.h,v 1.12 2007/01/16 09:37:03 brun Exp $    
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : DecisionTreeNode                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Node for the Decision Tree                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      MPI-K Heidelberg, Germany ,                                               * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_DecisionTreeNode
#define ROOT_TMVA_DecisionTreeNode

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DecisionTreeNode                                                     //
//                                                                      //
// Node for the Decision Tree                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_Node
#include "TMVA/Node.h"
#endif


using std::string;

namespace TMVA {

   class Event;

   class DecisionTreeNode: public Node {

   public:

      // constructor of an essentially "empty" node floating in space
      DecisionTreeNode ();
      // constructor of a daughter node as a daughter of 'p'
      DecisionTreeNode (Node* p, char pos); 

      // copy constructor 
      DecisionTreeNode (const DecisionTreeNode &n, DecisionTreeNode* parent = NULL); 

      virtual ~DecisionTreeNode(){}

      // test event if it decends the tree at this node to the right  
      virtual Bool_t GoesRight( const Event & ) const;

      // test event if it decends the tree at this node to the left 
      virtual Bool_t GoesLeft ( const Event & ) const;

      // set index of variable used for discrimination at this node
      void SetSelector( Short_t i) { fSelector = i; }
      // return index of variable used for discrimination at this node 
      Short_t GetSelector() const { return fSelector; }


      // set the cut value applied at this node 
      void  SetCutValue ( Double_t c ) { fCutValue  = c; }
      // return the cut value applied at this node
      Double_t GetCutValue ( void ) const { return fCutValue;  }

      // set true: if event variable > cutValue ==> signal , false otherwise
      void SetCutType( Bool_t t   ) { fCutType = t; }
      // return kTRUE: Cuts select signal, kFALSE: Cuts select bkg
      Bool_t GetCutType( void ) const { return fCutType; }

      // set node type: 1 signal node, -1 bkg leave, 0 intermediate Node
      void  SetNodeType( Int_t t ) { fNodeType = t;} 
      // return node type: 1 signal node, -1 bkg leave, 0 intermediate Node 
      Int_t GetNodeType( void ) const { return fNodeType; }

      //return  S/(S+B) (purity) at this node (from  training)
      Double_t GetPurity( void ) const ;

      // set the sum of the signal weights in the node
      void SetNSigEvents( Double_t s ) { fNSigEvents = s; }

      // set the sum of the backgr weights in the node
      void SetNBkgEvents( Double_t b ) { fNBkgEvents = b; }

      // set the number of events that entered the node (during training)
      void SetNEvents( Double_t nev ){ fNEvents =nev ; }

      // set the sum of the unweighted signal events in the node
      void SetNSigEvents_unweighted( Double_t s ) { fNSigEvents_unweighted = s; }

      // set the sum of the unweighted backgr events in the node
      void SetNBkgEvents_unweighted( Double_t b ) { fNBkgEvents_unweighted = b; }

      // set the number of unweighted events that entered the node (during training)
      void SetNEvents_unweighted( Double_t nev ){ fNEvents_unweighted =nev ; }

      // increment the sum of the signal weights in the node
      void IncrementNSigEvents( Double_t s ) { fNSigEvents += s; }

      // increment the sum of the backgr weights in the node
      void IncrementNBkgEvents( Double_t b ) { fNBkgEvents += b; }

      // increment the number of events that entered the node (during training)
      void IncrementNEvents( Double_t nev ){ fNEvents +=nev ; }

      // increment the sum of the signal weights in the node
      void IncrementNSigEvents_unweighted( ) { fNSigEvents_unweighted += 1; }

      // increment the sum of the backgr weights in the node
      void IncrementNBkgEvents_unweighted( ) { fNBkgEvents_unweighted += 1; }

      // increment the number of events that entered the node (during training)
      void IncrementNEvents_unweighted( ){ fNEvents_unweighted +=1 ; }

      // return the sum of the signal weights in the node
      Double_t GetNSigEvents( void ) const  { return fNSigEvents; }

      // return the sum of the backgr weights in the node
      Double_t GetNBkgEvents( void ) const  { return fNBkgEvents; }

      // return  the number of events that entered the node (during training)
      Double_t GetNEvents( void ) const  { return fNEvents; }

      // return the sum of unweighted signal weights in the node
      Double_t GetNSigEvents_unweighted( void ) const  { return fNSigEvents_unweighted; }

      // return the sum of unweighted backgr weights in the node
      Double_t GetNBkgEvents_unweighted( void ) const  { return fNBkgEvents_unweighted; }

      // return  the number of unweighted events that entered the node (during training)
      Double_t GetNEvents_unweighted( void ) const  { return fNEvents_unweighted; }


      // set the choosen index, measure of "purity" (separation between S and B) AT this node
      void SetSeparationIndex( Double_t sep ){ fSeparationIndex =sep ; }
      // return the separation index AT this node
      Double_t GetSeparationIndex( void ) const  { return fSeparationIndex; }

      // set the separation, or information gained BY this nodes selection
      void SetSeparationGain( Double_t sep ){ fSeparationGain =sep ; }
      // return the gain in separation obtained by this nodes selection
      Double_t GetSeparationGain( void ) const  { return fSeparationGain; }

      // printout of the node
      virtual void Print( ostream& os ) const;

      //recursively print the node and its daughters (--> print the 'tree')
      virtual void PrintRec( ostream&  os ) const;

      //recursively read the node and its daughters (--> read the 'tree')
      virtual void ReadRec( istream& is, char &pos, 
                            UInt_t &depth, TMVA::Node* parent=NULL );
      
      //recursively clear the nodes content (S/N etc, but not the cut criteria) 
      void ClearNodeAndAllDaughters();

      ULong_t GetSequence() const {return fSequence;}

      void SetSequence(ULong_t s) {fSequence=s;}
 
   private:
  
      Bool_t ReadDataRecord( istream& is );

      Double_t fCutValue;        // cut value appplied on this node to discriminate bkg against sig
      Bool_t   fCutType;         // true: if event variable > cutValue ==> signal , false otherwise
      Short_t  fSelector;        // index of variable used in node selection (decision tree) 
  
      Double_t fNSigEvents;      // sum of weights of signal event in the node
      Double_t fNBkgEvents;      // sum of weights of backgr event in the node
      Double_t fNEvents;         // number of events in that entered the node (during training)

      Double_t fNSigEvents_unweighted;      // sum of signal event in the node
      Double_t fNBkgEvents_unweighted;      // sum of backgr event in the node
      Double_t fNEvents_unweighted;         // number of events in that entered the node (during training)

      Double_t fSeparationIndex; // measure of "purity" (separation between S and B) AT this node
      Double_t fSeparationGain;  // measure of "purity", separation, or information gained BY this nodes selection
      Int_t    fNodeType;        // Type of node: -1 == Bkg-leaf, 1 == Signal-leaf, 0 = internal 

      ULong_t  fSequence;        // bit coded left right sequence to reach the node
  
      static MsgLogger* fLogger;    // static because there is a huge number of nodes...

      ClassDef(DecisionTreeNode,0) //Node for the Decision Tree 

   };
} // namespace TMVA

#endif 
