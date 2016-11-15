#include "tExecutor.h"
#include "ROOT/TProcessExecutor.hxx"

int PoolTest() {
   ROOT::TProcessExecutor pool;
   return TExecutorPoolTest(pool);
 }

int main() {
	return PoolTest();
}
