#include "JSONIOUtils.h"

#include <string>

using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

bool startsWith(std::string_view str, std::string_view prefix)
{
   return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
}

bool endsWith(std::string_view str, std::string_view suffix)
{
   return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

std::string removePrefix(std::string_view str, std::string_view prefix)
{
   std::string out;
   out += str;
   out = out.substr(prefix.length());
   return out;
}
std::string removeSuffix(std::string_view str, std::string_view suffix)
{
   std::string out;
   out += str;
   out = out.substr(0, out.length() - suffix.length());
   return out;
}

std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const JSONNode &treeRoot)
{
   std::string varName = treeRoot.find("name")->val();
   double val = 0;
   double maxVal = 0;
   double minVal = 0;
   bool isConstant = false;
   bool isRange = false;

   if (auto n = treeRoot.find("value")) {
      val = n->val_double();
      isConstant = true;
   }

   auto maxNode = treeRoot.find("max");
   auto minNode = treeRoot.find("min");
   if (maxNode && minNode) {
      maxVal = maxNode->val_double();
      minVal = minNode->val_double();
      isRange = true;
   }
   if (!isConstant) {
      val = (maxVal + minVal) / 2;
   }
   // Check if variable is at least a range or constant else throw error
   if (!isConstant && !isRange) {
      throw std::invalid_argument("Invalid Syntax: Please provide either 'value' or 'min' and 'max' or both");
   }

   std::unique_ptr<RooFit::Detail::JSONTree> jsonDict = RooFit::Detail::JSONTree::create();
   JSONNode &n = jsonDict->rootnode().set_map();
   JSONNode &_domains = n["domains"].set_seq().append_child().set_map();
   JSONNode &_parameterPoints = n["parameter_points"].set_seq().append_child().set_map();

   _domains["name"] << "default_domain";
   _domains["type"] << "product_domain";
   JSONNode &_axes = _domains["axes"].set_seq().append_child().set_map();
   _axes["name"] << varName;

   _parameterPoints["name"] << "default_values";
   JSONNode &_parameters = _parameterPoints["parameters"].set_seq().append_child().set_map();
   _parameters["name"] << varName;
   _parameters["value"] << val;

   if (isRange) {
      _axes["max"] << maxVal;
      _axes["min"] << minVal;
   }

   if (isConstant && !isRange) {
      _parameters["const"] << true;
      JSONNode &_misc = n["misc"].set_map();
      JSONNode &rootInternal = _misc["ROOT_internal"].set_map();
      JSONNode &_var = rootInternal[varName].set_map();
      _var["tags"] << "Constant";
   }

   return jsonDict;
}
