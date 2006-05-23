// @(#)root/tmva $Id: MethodANNBase.cxx,v 1.4 2006/05/23 09:53:10 stelzer Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::MethodANNBase                                                   *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
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
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/
   
//_______________________________________________________________________
//                                                                      
// Base class for all MVA methods using artificial neural networks      
//                                                                      
//_______________________________________________________________________

#include "TMVA/MethodANNBase.h"
#include "TList.h"
#include "TMVA/Tools.h"
#include "TObjString.h"
#include "Riostream.h"

using std::vector;

ClassImp(TMVA::MethodANNBase)

//_______________________________________________________________________
TMVA::MethodANNBase::MethodANNBase( void )
{}

//_______________________________________________________________________
vector<Int_t>* TMVA::MethodANNBase::ParseOptionString( TString theOptions, Int_t nvar,
                                                       vector<Int_t>* nodes )
{
   // default settings (should be defined in theOption string)
   TList*  list  = TMVA::Tools::ParseFormatLine( theOptions );

   // format and syntax of option string: "3000:N:N+2:N-3:6"
   //
   // where:
   //        3000 - number of training cycles (epochs)
   //        N    - number of nodes in first hidden layer, where N is the number
   //               of discriminating variables used (note that the first ANN
   //               layer necessarily has N nodes, and hence is not given).
   //        N+2  - number of nodes in 2nd hidden layer (2 nodes more than
   //               number of variables)
   //        N-3  - number of nodes in 3rd hidden layer (3 nodes less than
   //               number of variables)
   //        6    - 6 nodes in last (4th) hidden layer (note that the last ANN
   //               layer in MVA has 2 nodes, each one for signal and background
   //               classes)

   // sanity check
   if (list->GetSize() < 1) {
      cout << "--- Fatal error in NN parser (1): unrecognized option string: " << theOptions
           << " ==> exit(1)" << endl;
      exit(1);
   }

   // add number of cycles
   nodes->push_back( atoi( ((TObjString*)list->At(0))->GetString() ) );

   Int_t a;
   if (list->GetSize() > 1) {
      for (Int_t i=1; i<list->GetSize(); i++) {
         TString s = ((TObjString*)list->At(i))->GetString();
         s.ToUpper();
         if (s(0) == 'N')  {
            if (s.Length() > 1) nodes->push_back( nvar + atoi(&s[1]) );
            else                nodes->push_back( nvar );
         }
         else if ((a = atoi( s )) > 0) nodes->push_back( atoi(s ) );
         else {
            cout << "--- Fatal error in NN parser (1): unrecognized option string: " << theOptions
                 << " ==> exit(1)" << endl;
            exit(1);
         }
      }
   }

   return nodes;
}

