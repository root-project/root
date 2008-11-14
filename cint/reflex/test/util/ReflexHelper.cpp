#include "./ReflexHelper.hpp"

//-----------------------------------------------------------------------------------------------------
Reflex::Type TypeProxy::getType(const std::string& typeName)
//-----------------------------------------------------------------------------------------------------
{
    return Reflex::Type::ByName(typeName);
}
