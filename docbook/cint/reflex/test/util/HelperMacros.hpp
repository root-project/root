#ifndef UTIL_HELPERMACROS_HPP
#define UTIL_HELPERMACROS_HPP

#include <cppunit/extensions/HelperMacros.h>
#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <cppunit/plugin/DynamicLibraryManager.h>

namespace ReflexTestUtils
{

CppUnit::TestSuite* getReflexTestSuite()
{
   static CppUnit::TestSuite* reflexTestSuite = new CppUnit::TestSuite("Reflex Tests");
   return reflexTestSuite;
}

class FunctionTestCase : public CppUnit::TestCase
{
public:
   FunctionTestCase(void (*test)()) : m_test(test) {}

   void runTest()
   {
      m_test();
   }

private:
   void (*m_test)();
};

class ReflexTestRegistration
{
public:
   ReflexTestRegistration(void (*test)())
   {
      getReflexTestSuite()->addTest(new FunctionTestCase(test));
   }

private:
   ReflexTestRegistration(const ReflexTestRegistration& that);
   ReflexTestRegistration& operator=(const ReflexTestRegistration& that);
};

#define REFLEX_TEST(name) static void name(); \
                          ReflexTestUtils::ReflexTestRegistration CPPUNIT_MAKE_UNIQUE_NAME(ReflexTest_##name_)((name)); \
                          static void name()
}

int main(int argc, char **argv)
{
   // Load the dictionary specified on the cl
   CppUnit::DynamicLibraryManager dlm(argv[1]);

   // Get the top level suite from the registry
   //CppUnit::Test *suite = CppUnit::TestFactoryRegistry::getRegistry().makeTest();
   CppUnit::Test *suite = ReflexTestUtils::getReflexTestSuite();

   // Adds the test to the list of test to run
   CppUnit::TextUi::TestRunner runner;
   runner.addTest(suite);

   // Change the default outputter to a compiler error format outputter
   runner.setOutputter(new CppUnit::CompilerOutputter(&runner.result(), std::cerr));

   // Run the tests.
   bool wasSucessful = runner.run();

   // Return error code 1 if the one of test failed.
   return wasSucessful ? 0 : 1;
}

#endif // UTIL_HELPERMACROS_HPP
