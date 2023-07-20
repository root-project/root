#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>
#include <memory>

#include "TMVA/SOFIE_common.hxx"
//#include "RModel.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{


// overload operator * for concatenating SP
inline const std::string operator*(const std::string str, std::size_t n) {
   std::string res(str);

   if (n == 0 || str.empty()) return {};
   if (n == 1) return str;

   const auto period = str.size();

   if (period == 1) return std::string(n, str.front());

   res.reserve(period * n);
   std::size_t m{2};
   for (; m < n; m *= 2) res += res;
   res.append(res.c_str(), (n - (m / 2)) * period);
   return res;
}

class RModel;

class ROperator{


public:
   virtual std::vector<std::string> GetBlasRoutines() { return {}; }
   virtual std::vector<std::string> GetStdLibs() { return {}; }
   virtual std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) = 0;
   virtual std::vector<ETensorType> TypeInference(std::vector<ETensorType>) = 0;
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opname for each operator within the same RModel
   virtual std::string GenerateGPU(std::string OpName) = 0;
   // generate initialization code
   virtual std::string GenerateInitCode() { return "";}
   // generate session data members specific to operator
   virtual std::string GenerateSessionMembersCode(std::string /*opName*/) { return ""; }
   virtual std::string Header() { return "";}


   //virtual void Forward_reference() = 0;
   //irtual void Forward_blas() = 0;
   virtual ~ROperator(){};

protected:

   const std::string SP = "    ";    ///< space used to correctly indent the generated C++ code
   bool fUseSession = false;        ///< flag to identify if using the session class

};



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
