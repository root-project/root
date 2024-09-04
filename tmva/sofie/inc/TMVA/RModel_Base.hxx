#ifndef TMVA_SOFIE_RMODEL_BASE
#define TMVA_SOFIE_RMODEL_BASE

#include <type_traits>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <memory>
#include <ctime>
#include <set>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TBuffer.h"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

enum class Options {
   kDefault = 0x0,
   kNoSession = 0x1,
   kNoWeightFile = 0x2,
   kRootBinaryWeightFile = 0x4,
   kGNN = 0x8,
   kGNNComponent = 0x10,
};

enum class WeightFileType { None, RootBinary, Text };

std::underlying_type_t<Options> operator|(Options opA, Options opB);
std::underlying_type_t<Options> operator|(std::underlying_type_t<Options> opA, Options opB);

class RModel_Base {

protected:
   std::string fFileName;  // file name of original model file for identification
   std::string fParseTime; // UTC date and time string at parsing

   WeightFileType fWeightFile = WeightFileType::Text;

   std::unordered_set<std::string> fNeededBlasRoutines;

   const std::unordered_set<std::string> fAllowedStdLib = {"vector", "algorithm", "cmath"};
   std::unordered_set<std::string> fNeededStdLib = {"vector"};
   std::unordered_set<std::string> fCustomOpHeaders;

   std::string fName = "UnnamedModel";
   std::string fGC; // generated code
   bool fUseWeightFile = true;
   bool fUseSession = true;
   bool fIsGNN = false;
   bool fIsGNNComponent = false;

public:
   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel_Base() = default;

   RModel_Base(std::string name, std::string parsedtime);

   // For GNN Functions usage
   RModel_Base(std::string function_name) : fName(function_name) {}

   void AddBlasRoutines(std::vector<std::string> routines)
   {
      for (auto &routine : routines) {
         fNeededBlasRoutines.insert(routine);
      }
   }
   void AddNeededStdLib(std::string libname)
   {
      if (fAllowedStdLib.find(libname) != fAllowedStdLib.end()) {
         fNeededStdLib.insert(libname);
      }
   }
   void AddNeededCustomHeader(std::string filename)
   {
       fCustomOpHeaders.insert(filename);
   }
   void GenerateHeaderInfo(std::string &hgname);
   void PrintGenerated() { std::cout << fGC; }

   std::string ReturnGenerated() { return fGC; }
   void OutputGenerated(std::string filename = "", bool append = false);
   void SetFilename(std::string filename) { fName = filename; }
   std::string GetFilename() { return fName; }
   const std::string & GetName() const { return fName;}
};

enum class GraphType { INVALID = 0, GNN = 1, GraphIndependent = 2 };

enum class FunctionType { UPDATE = 0, AGGREGATE = 1 };
enum class FunctionTarget { INVALID = 0, NODES = 1, EDGES = 2, GLOBALS = 3 };
enum class FunctionReducer { INVALID = 0, SUM = 1, MEAN = 2 };
enum class FunctionRelation { INVALID = 0, NODES_EDGES = 1, NODES_GLOBALS = 2, EDGES_GLOBALS = 3 };

class RModel_GNNBase : public RModel_Base {
public:
   /**
       Default constructor. Needed to allow serialization of ROOT objects. See
       https://root.cern/manual/io_custom_classes/#restrictions-on-types-root-io-can-handle
   */
   RModel_GNNBase() = default;
   virtual void Generate() = 0;
   virtual ~RModel_GNNBase() = default;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODEL_BASE
