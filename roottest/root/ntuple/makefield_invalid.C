#include <ROOT/RNTupleModel.hxx>

#include <TSystem.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

// Regression test for https://github.com/root-project/root/issues/13910:
// MakeField() must report a diagnosable error for types that RNTuple cannot
// construct, and it must leave the model and the interpreter usable.

class RNTupleTestAbstract {
public:
   virtual ~RNTupleTestAbstract() = default;
   virtual void Virt() = 0;
};

class RNTupleTestDerived : public RNTupleTestAbstract {
public:
   int fValue = 42;
   void Virt() final {}
};

class RNTupleTestNoDefaultCtor {
public:
   explicit RNTupleTestNoDefaultCtor(int value) : fValue(value) {}
   int fValue = 0;
};

namespace {

int gFailures = 0;

template <typename T>
void ExpectFieldFailure(ROOT::RNTupleModel &model, const char *label, const char *fieldName, const char *expected)
{
   try {
      // Keep the returned value alive until after the failure is reported: it
      // may point to unconstructed memory, so letting it go out of scope would
      // crash instead of reporting a clean test failure.
      auto value = model.MakeField<T>(fieldName);
      (void)value;
      std::cerr << "FAIL: MakeField<" << label << "> did not throw" << std::endl;
      gSystem->Exit(1);
   } catch (const std::exception &ex) {
      const std::string msg = ex.what();
      std::cout << "MakeField<" << label << "> threw: " << msg << std::endl;
      if (msg.find(expected) == std::string::npos) {
         std::cerr << "FAIL: expected the error message to contain \"" << expected << "\"" << std::endl;
         ++gFailures;
      }
   } catch (...) {
      std::cerr << "FAIL: MakeField<" << label << "> threw a non-standard exception" << std::endl;
      ++gFailures;
   }
}

} // namespace

void makefield_invalid()
{
   {
      auto model = ROOT::RNTupleModel::Create();
      ExpectFieldFailure<RNTupleTestAbstract>(*model, "Abstract", "abstract", "abstract");
      if (model->GetDefaultEntry().HasField("abstract")) {
         std::cerr << "FAIL: failed field \"abstract\" was added to the default entry" << std::endl;
         ++gFailures;
      }

      // A failed MakeField() must not leave a partial field in the default
      // entry: further fields on the same model have to keep working.
      try {
         model->MakeField<std::vector<float>>("vec");
         std::cout << "OK: model usable after abstract failure" << std::endl;
      } catch (const std::exception &ex) {
         std::cerr << "FAIL: model unusable after abstract failure: " << ex.what() << std::endl;
         ++gFailures;
      }
   }

   {
      auto model = ROOT::RNTupleModel::Create();
      ExpectFieldFailure<RNTupleTestNoDefaultCtor>(*model, "NoDefaultCtor", "nodefault", "default constructor");
      if (model->GetDefaultEntry().HasField("nodefault")) {
         std::cerr << "FAIL: failed field \"nodefault\" was added to the default entry" << std::endl;
         ++gFailures;
      }

      try {
         model->MakeField<RNTupleTestDerived>("derived");
         std::cout << "OK: model usable after no-default-constructor failure" << std::endl;
      } catch (const std::exception &ex) {
         std::cerr << "FAIL: model unusable after no-default-constructor failure: " << ex.what() << std::endl;
         ++gFailures;
      }
   }

   if (gFailures != 0) {
      std::cerr << gFailures << " check(s) failed" << std::endl;
      gSystem->Exit(1);
   }
   std::cout << "all MakeField failure checks passed" << std::endl;
}
