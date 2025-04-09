#include "IInterface.h"

static const InterfaceID IID_IRndmGauss(154, 1, 0);

class Param {
 protected:
   const InterfaceID m_type;
 public:
   Param( const InterfaceID& type = IID_IRndmGauss ) : m_type(type) {}
};