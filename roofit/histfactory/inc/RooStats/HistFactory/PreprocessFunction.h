
#ifndef PREPROCESS_FUNCTION_H
#define PREPROCESS_FUNCTION_H

#include <string>
#include <iostream>

namespace RooStats{
namespace HistFactory {

  class PreprocessFunction {
  public:

    PreprocessFunction();

    
    PreprocessFunction(std::string Name, std::string Expression, std::string Dependents);
    std::string GetCommand(std::string Name, std::string Expression, std::string Dependents);
			   

    void Print(std::ostream& = std::cout);  

    void SetName( const std::string& Name) { fName = Name; }
    std::string GetName() { return fName; }

    void SetExpression( const std::string& Expression) { fExpression = Expression; }
    std::string GetExpression() { return fExpression; }

    void SetDependents( const std::string& Dependents) { fDependents = Dependents; }
    std::string GetDependents() { return fDependents; }
    
    void SetCommand( const std::string& Command) { fCommand = Command; }
    std::string GetCommand() { return fCommand; }

  protected:


    std::string fName;
    std::string fExpression;
    std::string fDependents;

    std::string fCommand;



  };


}
}


#endif
