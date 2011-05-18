#if !defined            (_REFLEX_HELPER_HPP__)
#define                  _REFLEX_HELPER_HPP__

#include <Reflex/Type.h>

class TypeProxy;

typedef TypeProxy ReflexHelper;

/**
* @class TypeProxy
*
* for now just mimics normal getType behaviour. In future could be a wrapper to test
* lazy loading, selection file filtering etc
*/
class TypeProxy
{
public:
    static Reflex::Type getType(const std::string& typeName);
};

#endif                 //_REFLEX_HELPER_HPP__
