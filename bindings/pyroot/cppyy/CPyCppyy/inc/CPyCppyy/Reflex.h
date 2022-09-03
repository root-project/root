#ifndef CPYCPPYY_REFLEX_H
#define CPYCPPYY_REFLEX_H

//
// Access to the C++ reflection information
//

namespace Cppyy {

namespace Reflex {

typedef int RequestId_t;

const RequestId_t IS_NAMESPACE    = 1;

const RequestId_t OFFSET          = 2;
const RequestId_t RETURN_TYPE     = 3;
const RequestId_t TYPE            = 4;

typedef int FormatId_t;
const FormatId_t OPTIMAL          = 1;
const FormatId_t AS_TYPE          = 2;
const FormatId_t AS_STRING        = 3;

} // namespace Reflex

} // namespace Cppyy

#endif // !CPYCPPYY_REFLEX_H
