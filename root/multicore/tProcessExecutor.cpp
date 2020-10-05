#include "executorTests.hxx"
#include "ROOT/TProcessExecutor.hxx"

int PoolTest() {
  ROOT::TProcessExecutor pool;
  return ExecutorTest(pool);
}

int main() {
	return PoolTest();
}
