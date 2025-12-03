/*
  File: roottest/python/basic/ReturnValues.C
  Author: Pere Mato
  Created: 02/02/16
  Last: 02/02/16
*/

struct testIntegerResults {
    short shortPlusOne() const { return 1; }
    short shortMinusOne() const { return -1; }
    int intPlusOne() const { return 1; }
    int intMinusOne() const { return -1; }
    long longPlusOne() const { return 1; }
    long longMinusOne() const { return -1; }
    long long longlongPlusOne() const { return 1;}
    long long longlongMinusOne() const { return -1;}
};
