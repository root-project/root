// @(#)root/tmva $\Id$
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MetricEuler                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer <speckmay@mail.cern.ch> - CERN, Switzerland              *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// interface for a metric
//
//_______________________________________________________________________

#include "TMVA/MetricEuler.h"
#include <cmath>

ClassImp(TMVA::MetricEuler)

//_______________________________________________________________________
TMVA::MetricEuler::MetricEuler() 
   : IMetric()
{
   // constructor
}            


//_______________________________________________________________________
Double_t TMVA::MetricEuler::Distance( std::vector<Double_t>& pointA, std::vector<Double_t>& pointB )
{
   Double_t distance = 0.0;
   Double_t val = 0.0;
   std::vector<Double_t>::iterator itA;
   std::vector<Double_t>::iterator itB;
   if( fParameters == NULL ){
      itA = pointA.begin();
      for( itB = pointB.begin(); itB != pointB.end(); itB++ ){
         if( itA == pointA.end() ){
            break;
         }
         val = (*itA)-(*itB);
         distance += std::pow( val, 2 );
         itA++;
      }
   }else{
      std::vector<Double_t>::iterator itPar;
      itA   = pointA.begin();
      itPar = fParameters->begin();
      for( itB = pointB.begin(); itB != pointB.end(); itB++ ){
         if( itA == pointA.end() ){
            break;
         }
         if( itPar == fParameters->end() ){
            break;
         }
         val = (*itPar)*( (*itA)-(*itB) );
         distance += std::pow( val, 2 );
         itA++;
         itPar++;
      }
      if( itA != pointA.end() ){
         distance *= std::pow( (*itA),2 );
      }
   }
   return sqrt( distance );
}


