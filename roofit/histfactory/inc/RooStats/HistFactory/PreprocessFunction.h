#ifndef PREPROCESS_FUNCTION_H
#define PREPROCESS_FUNCTION_H

#include <string>
#include <iostream>

namespace RooStats {
namespace HistFactory {

class PreprocessFunction {
public:
   PreprocessFunction() {}

   PreprocessFunction(std::string const &name, std::string const &expression, std::string const &dependents);

   void Print(std::ostream & = std::cout) const;
   void PrintXML(std::ostream &) const;

   void SetName(const std::string &name) { fName = name; }
   std::string const &GetName() const { return fName; }

   void SetExpression(const std::string &expression) { fExpression = expression; }
   std::string const &GetExpression() const { return fExpression; }

   void SetDependents(const std::string &dependents) { fDependents = dependents; }
   std::string const &GetDependents() const { return fDependents; }

   std::string GetCommand() const;

private:
   std::string fName;
   std::string fExpression;
   std::string fDependents;
};

} // namespace HistFactory
} // namespace RooStats

#endif
