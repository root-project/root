#ifndef CPYRT_REFLEX_H
#define CPYRT_REFLEX_H

//
// Access to the C++ reflection information
//

namespace cppjit::interop {

namespace Reflex {

typedef int RequestId_t;

const RequestId_t IS_NAMESPACE = 1;
const RequestId_t IS_AGGREGATE = 2;

const RequestId_t OFFSET = 16;
const RequestId_t RETURN_TYPE = 17;
const RequestId_t TYPE = 18;

typedef int FormatId_t;
const FormatId_t OPTIMAL = 1;
const FormatId_t AS_TYPE = 2;
const FormatId_t AS_STRING = 3;

} // namespace Reflex

} // namespace cppjit::interop

#endif // !CPYRT_REFLEX_H
