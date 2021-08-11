// @(#)root/tmva $Id$
// Authors: Simone Azeglio, Lorenzo Moneta , Sitong An,  Stefan Wunsch

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
 *      Lorenzo Moneta, CERN
 *      Sitong An, CERN, CMU                                                        *
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

#include "TMVA/RTensor.hxx"


namespace TMVA {
  
    class RVariablePlotter {

    public:
        
        // constructor - RDataFrame input
        RVariablePlotter(const std::vector<ROOT::RDF::RNode>& nodes, const std::vector<std::string>& labels);
        
        // constructor - Tensor input
        RVariablePlotter( const std::vector<TMVA::Experimental::RTensor<Float_t>>& tensors, const std::vector<std::string>& labels);
        
        // use tmva plotting style
        void InitializeStyle(bool useTMVAStyle);
        
        // draw variables plot - RDataFrame input
        void Draw(const std::string& variable, bool useTMVAStyle);
        
        // draw variables plot - RTensor input
        void DrawTensor(const std::string& variable,  const std::vector<std::string>& variables,bool useTMVAStyle);
        
        // draw legend
        void DrawLegend(float minX, float minY, float maxX, float maxY);

        
        // draw ROC Curve - RNode input
        void DrawROCCurve(const std::string& output_variable , bool useTMVAStyle);

        // draw Stacked ROC Curves - RNode input 
        void DrawMultiROCCurve(const std::vector<std::string>& output_variables, bool useTMVAStyle);

        // draw ROC Curve - RTensor input
        void DrawROCCurveTensor(const std::string& output_variable, const std::vector<std::string>& output_variables, bool useTMVAStyle);


        // draw Stacked ROC Curves - RTensor input 
        void DrawMultiROCCurveTensor(const std::vector<std::string>& output_variables, bool useTMVAStyle);

    private:
        
        std::vector<ROOT::RDF::RNode> fNodes; //! transient
        std::vector<TMVA::Experimental::RTensor<Float_t>> fTensors; //! transient
        std::vector<std::string> fLabels;
        
        // convert RTensor to RDataframe
        ROOT::RDF::RNode TensorToNode(const TMVA::Experimental::RTensor<Float_t>& tensor,  const std::vector<std::string>& variables);
        
        // convert vector of RTensors to vector of RDataframes
        std::vector<ROOT::RDF::RNode> TensorsToNodes(const std::vector<std::string>& variables);
        
        


   };

} // namespace TMVA





#endif /* ROOT_TMVA_RVariablePlotter */
