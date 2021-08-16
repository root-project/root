// @(#)root/tmva $Id$
// Author: Jonas Rembser, Sanjiban Sengupta, 2021

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RootStorage::BDT                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *     Class for storing trained BDT models into Root files                       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Jonas Rembser                                                             *
 *      Sanjiban Sengupta                                                         *
 *                                                                                *
 * Copyright (c) 2021:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_SOFIE_ROOTSTORAGEBDT
#define TMVA_SOFIE_ROOTSTORAGEBDT

#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>
#include <unordered_map>

#include "TBuffer.h"


namespace TMVA{
namespace Experimental{
namespace RootStorage{

namespace INTERNAL{
    class XMLAttributes {
        private:
        std::unordered_map<std::string, void*> attributes = {
            {"boostWeight",NULL},
            {"itree",NULL},
            {"pos",NULL},
            {"depth",NULL},
            {"IVar",NULL},
            {"Cut",NULL},
            {"cType",NULL},
            {"res",NULL},
            {"purity",NULL},
            {"nType",NULL}
        };

        public:
        bool setValue(std::string &name, std::string &value);
        bool hasValue(std::string &name);
        void reset();
        auto  boostWeight(){ return *((double*)attributes["boostWeight"]); };
        auto  itree()      { return *((int*)attributes["itree"]); };
        auto  pos()        { return *((char*)attributes["pos"]); };
        auto  depth()      { return *((int*)attributes["depth"]); };
        auto  IVar()       { return *((int*)attributes["IVar"]); };
        auto  Cut()        { return *((double*)attributes["Cut"]); };
        auto  cType()      { return *((int*)attributes["cType"]); };
        auto  res()        { return *((double*)attributes["res"]); };
        auto  purity()     { return *((double*)attributes["purity"]); };
        auto  nType()      { return *((int*)attributes["nType"]); };
    };

    struct BDTWithXMLAttributes {
        std::vector<double> boostWeights;
        std::vector<std::vector<XMLAttributes>> nodes;
        };
}

struct SlowTreeNode {
        bool fIsLeaf = false;
        int fDepth = -1;
        int fIndex = -1;
        int fYes = -1;
        int fNo = -1;
        int fMissing = -1;
        int fCutIndex = -1;
        int fCutType = 0;
        double fCutValue = 0.0;
        double fLeafValue = 0.0;
    };

class BDT: public TObject{
    private:
    std::vector<std::vector<SlowTreeNode>> fSlowForest;
    std::string filename;

    public:
    BDT(){}
    BDT(BDT&& other);
    BDT& operator=(BDT&& other);

    //disallow copy
    BDT(const BDT& other) = delete;
    BDT& operator=(const BDT& other) = delete;
    void Parse(std::string filepath, bool use_purity=true);
    std::vector<SlowTreeNode> GetSlowTreeNodes(std::vector<INTERNAL::XMLAttributes> &nodes, bool usePurity);
    ClassDef(BDT,1);
  };
}//RootStorage
}//Experimental
}//TMVA

#endif //TMVA_ROOTSTORAGE_BDT
