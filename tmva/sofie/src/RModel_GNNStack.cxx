#include <limits>
#include <algorithm>
#include <cctype>

#include "TMVA/RModel_GNNStack.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{

    void RModel_GNNStack::Initialize(){
        for(size_t i=0; i<fGraphs.size(); ++i){
            fNeededStdLib.insert(fGraphs[i]->GetFilename());
        }
    }

    void RModel_GNNStack::Generate(int batchSize){
        Initialize();
        std::string hgname;
        GenerateHeaderInfo(hgname);

        // computing inplace on input graph
        fGC += "void infer(GNN::GNN_Data& input_graph){\n";
        for(auto &it:fGraphs){
            fGC+="TMVA_SOFIE"+it->GetFilename()+"::infer(input_graph);\n";
        }

        fGC += ("} //TMVA_SOFIE_" + fName + "\n");
        fGC += "\n#endif  // TMVA_SOFIE_" + hgname + "\n";
    }

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
