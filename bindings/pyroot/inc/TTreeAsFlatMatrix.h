// Author: Danilo Piparo, Enrico Guiraud, Stefan Wunsch CERN  04/2018

/*************************************************************************
 * Copyright (C) 1995-2016, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeAsFlatMatrix
#define ROOT_TTreeAsFlatMatrix

#include <vector>
#include <string>
#include "TTree.h"
#include "ROOT/RDataFrame.hxx"
#include "Rtypes.h"
#include <utility>

namespace PyROOT {
    template <typename dtype>
    ULong64_t GetVectorAddress(std::vector<dtype> &p){return reinterpret_cast<ULong64_t>(&p);}

    inline ULong64_t GetAddress(std::vector<std::string> &p){return reinterpret_cast<ULong64_t>(&p);}
    inline ULong64_t GetAddress(TTree &p){return reinterpret_cast<ULong64_t>(&p);}

    template <typename BufType, typename... ColTypes, std::size_t... Idx>
    void TTreeAsFlatMatrix(
            std::index_sequence<Idx...>, TTree& tree, std::vector<BufType>& matrix, std::vector<std::string>& columns)
    {
        auto buffer = matrix.data();

        auto fillMatrix = [buffer](ColTypes... cols, ULong64_t entry) {
            int expander[] = { (buffer[entry * sizeof...(Idx) + Idx] = cols, 0)... };
            (void)expander;
         };

        auto columnsWithEntry = columns;
        columnsWithEntry.emplace_back("tdfentry_");

        ROOT::RDataFrame dataframe(tree, columns);
        dataframe.Foreach(fillMatrix, columnsWithEntry);
    }

    template <typename BufType, typename... ColTypes>
    void TTreeAsFlatMatrixHelper(TTree& tree, std::vector<BufType>& matrix, std::vector<std::string>& columns)
    {
        TTreeAsFlatMatrix<BufType, ColTypes...>(std::index_sequence_for<ColTypes...>(), tree, matrix, columns);
    }
}

#endif
