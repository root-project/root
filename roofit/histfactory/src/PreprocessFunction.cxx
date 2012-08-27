
#include "RooStats/HistFactory/PreprocessFunction.h"


RooStats::HistFactory::PreprocessFunction::PreprocessFunction() {;}

RooStats::HistFactory::PreprocessFunction::PreprocessFunction(std::string Name, std::string Expression, std::string Dependents) : 
  fName(Name), fExpression(Expression), fDependents(Dependents) {
  fCommand = GetCommand(Name, Expression, Dependents);
}

std::string RooStats::HistFactory::PreprocessFunction::GetCommand(std::string Name, std::string Expression, std::string Dependents) {
  std::string command = "expr::"+Name+"('"+Expression+"',{"+Dependents+"})";
  return command;
}


void RooStats::HistFactory::PreprocessFunction::Print( std::ostream& stream ) {
  
  stream << "\t \t Name: " << fName
	 << "\t \t Expression: " << fExpression
	 << "\t \t Dependents: " << fDependents
	 << std::endl;  
  
}
