#include "TMVA/RootStorageBDT.h"

namespace TMVA{
namespace Experimental{
namespace RootStorage{

namespace INTERNAL{
     bool XMLAttributes::setValue(std::string& name, std::string& value) {
        if (name == "pos"){
                if(hasValue(name)){
                    std::memcpy(attributes[name],&value[0],sizeof(char));
                    return false;
                }
                else{
                    attributes[name] = malloc(sizeof(char));
                    std::memcpy(attributes[name],&value[0],sizeof(char));
                    return true;
                }
        }
        if ((name == "itree") || (name == "depth") || (name == "IVar") || (name == "nType") || (name == "cType")){
            int castedValue = std::stoi(value);
            if(hasValue(name)){
                    std::memcpy(attributes[name],&castedValue,sizeof(int));
                    return false;
                }
                else{
                    attributes[name] = malloc(sizeof(int));
                    std::memcpy(attributes[name],&castedValue,sizeof(int));
                    return true;
                }
        }

        if ((name == "boostWeight") || (name == "Cut") || (name == "res") ||(name == "purity")){
             double castedValue = std::stod(value);
             if(hasValue(name)){
                    std::memcpy(attributes[name],&castedValue,sizeof(double));
                    return false;
                }
                else{
                    attributes[name] = malloc(sizeof(double));
                    std::memcpy(attributes[name],&castedValue,sizeof(double));
                    return true;
                }
        }

        return false;
    }

     bool XMLAttributes::hasValue(std::string &name){
            if (attributes[name])
                return true;
            return false;
        }

    void XMLAttributes::reset(){
        for (auto itr = attributes.begin(); itr != attributes.end(); itr++){
            itr->second=NULL;
        }
    }
}

BDT::BDT(BDT&& other){
    fSlowForest = other.fSlowForest;
    filename    = other.filename;
}

BDT& BDT::operator=(BDT&& other){
    fSlowForest = other.fSlowForest;
    filename    = other.filename;
    return *this;
}

void BDT::Parse(std::string filepath, bool usePurity){

    char sep = '/';
    #ifdef _WIN32
    sep = '\\';
    #endif

    size_t isep = filepath.rfind(sep, filepath.length());
    if (isep != std::string::npos){
      filename = (filepath.substr(isep+1, filepath.length() - isep));
    }

    //Check on whether the TMVA BDT XML file exists
    if(!std::ifstream(filepath).good()){
        throw std::runtime_error("Model file "+filename+" not found!");
    }

    std::ifstream fPointer(filepath);
    std::string fXmlString;
    fPointer.seekg(0, std::ios::end);
    fXmlString.reserve(fPointer.tellg());
    fPointer.seekg(0, std::ios::beg);
    fXmlString.assign((std::istreambuf_iterator<char>(fPointer)), std::istreambuf_iterator<char>());
    INTERNAL::BDTWithXMLAttributes fBdtXmlAttributes;
    std::vector<INTERNAL::XMLAttributes>* fCurrentTree = nullptr;
    INTERNAL::XMLAttributes* fAttributes = nullptr;

    std::string fName, fValue;
    std::size_t fFirstPosition = 0;

    while((fFirstPosition = fXmlString.find('=',fFirstPosition)) != std::string::npos){
        auto fSecondPosition = fXmlString.rfind(' ',fFirstPosition)+1;
        fName = fXmlString.substr(fSecondPosition,fFirstPosition-fSecondPosition);

        fSecondPosition = fFirstPosition+2;
        fFirstPosition = fXmlString.find('"',fSecondPosition);

        fValue = fXmlString.substr(fSecondPosition,fFirstPosition-fSecondPosition);

        if(fName == "boostWeight"){
            fBdtXmlAttributes.boostWeights.push_back(std::stod(fValue));
        }

        if(fName == "itree"){
            fBdtXmlAttributes.nodes.emplace_back();
            fCurrentTree = &fBdtXmlAttributes.nodes.back();
            fCurrentTree->emplace_back();
            fAttributes = &fCurrentTree->back();
        }

        if(fBdtXmlAttributes.nodes.empty())
        continue;

        if(fAttributes->hasValue(fName)){
            fCurrentTree->emplace_back();
            fAttributes = &fCurrentTree->back();
        }

        fAttributes->setValue(fName,fValue);
    }
    if(fBdtXmlAttributes.nodes.size() != fBdtXmlAttributes.boostWeights.size()){
        throw std::runtime_error("[ERROR] Nodes size and BoostWeights size do not match");
    }

    for(auto &tree: fBdtXmlAttributes.nodes){
        fSlowForest.push_back(GetSlowTreeNodes(tree, usePurity));

    }
}

std::vector<SlowTreeNode> BDT::GetSlowTreeNodes(std::vector<INTERNAL::XMLAttributes> &nodes, bool usePurity){
    std::vector<SlowTreeNode> fTreeNodes(nodes.size());
    long unsigned int index = 0;
    int iNode;
    for(int depth = 0;index!=nodes.size();++depth){
        iNode=0;
        for(auto& node: nodes){
            if(node.depth() == depth){
                fTreeNodes[iNode].fIndex = index;
                ++index;
            }
            ++iNode;
        }
    }
    iNode=0;
    for(auto& node : nodes){
        auto& fNode     = fTreeNodes[iNode];
        fNode.fIsLeaf    = node.nType()!=0;
        fNode.fDepth     = node.depth();
        fNode.fCutIndex  = node.IVar();
        fNode.fCutValue  = node.Cut();
        fNode.fCutType   = node.cType();
        fNode.fLeafValue = usePurity ? node.purity() : node.res();
        if(!fNode.fIsLeaf){
            fNode.fYes   = fTreeNodes[iNode+1].fIndex;
            fNode.fNo    = fNode.fYes+1;
            fNode.fMissing = fNode.fYes;
        }
        ++iNode;
    }
    return fTreeNodes;
}
}
}
}
