// @(#)root/tmva $Id$
// Authors: Simone Azeglio, Lorenzo Moneta , Stefan Wunsch

/*************************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis          *
 * Package: TMVA                                                                     *
 * Class  : RVariablePlotter                                                         *
 * Web    : http://tmva.sourceforge.net                                              *
 *                                                                                   *
 * Description:                                                                      *
 *      Variable Plotter                                                             *
 *                                                                                   *
 * Authors (alphabetical):                                                           *
 *      Simone Azeglio, University of Turin (Master Student), CERN (Summer Student)  *
 *      Lorenzo Moneta, CERN                                                         *
 *      Stefan Wunsch                                                                *
 *                                                                                   *
 * Copyright (c) 2021:                                                               *
 *                                                                                   *
 * Redistribution and use in source and binary forms, with or without                *
 * modification, are permitted according to the terms listed in LICENSE              *
 * (http://tmva.sourceforge.net/LICENSE)                                             *
 **********************************************************************************/

#ifndef ROOT_TMVA_RVariablePlotter
#define ROOT_TMVA_RVariablePlotter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RVariablePlotter                                                     //
//                                                                      //
// Variable Plotter                                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#include <string>

#include "TLegend.h"
#include "TH1D.h"
#include "THStack.h"

#include "ROOT/RDataFrame.hxx"
#include "ROOT/RDF/RInterface.hxx"

namespace TMVA {
  
    class RVariablePlotter {

    public:
        
        // constructor
        RVariablePlotter( const std::vector<ROOT::RDF::RNode>& nodes, const std::vector<std::string>& labels);
        
        // use tmva plotting style
        void InitializeStyle(bool useTMVAStyle);
        
        // draw variables plot
        void Draw(const std::string& variable,  bool useTMVAStyle);
        
        // draw legend
        void DrawLegend(float minX, float minY, float maxX, float maxY);
        

    private:
        
        std::vector<ROOT::RDF::RNode> fNodes; //! transient   
        std::vector<std::string> fLabels;


     // flag if "boundary vector" is owned by the volume of not
   };

} // namespace TMVA





#endif /* ROOT_TMVA_RVariablePlotter */
