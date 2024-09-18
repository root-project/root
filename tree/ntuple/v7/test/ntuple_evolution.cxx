#include "ntuple_test.hxx"

#include <TInterpreter.h>

#include <functional>
#include <sstream>
#include <string>
#include <string_view>

#include <unistd.h>
#include <sys/wait.h>

namespace {
// These helpers are split into an *Impl function and a macro to check for and propagate fatal failures to the caller.

void WriteOldInForkImpl(std::function<void(void)> old)
{
   pid_t pid = fork();
   if (pid == -1) {
      FAIL() << "fork() failed";
   }

   if (pid == 0) {
      old();
      if (::testing::Test::HasFatalFailure())
         exit(EXIT_FAILURE);
      exit(EXIT_SUCCESS);
   }

   int wstatus;
   if (waitpid(pid, &wstatus, 0) == -1) {
      FAIL() << "waitpid() failed";
   } else if (!WIFEXITED(wstatus) || WEXITSTATUS(wstatus) != 0) {
      FAIL() << "child did not exit successfully";
   }
}

#define WriteOldInFork(old)                   \
   do {                                       \
      WriteOldInForkImpl(old);                \
      if (::testing::Test::HasFatalFailure()) \
         return;                              \
   } while (0)

void DeclarePointerImpl(std::string_view type, std::string_view variable, void *ptr)
{
   auto ptrString = std::to_string(reinterpret_cast<std::uintptr_t>(ptr));
   std::ostringstream ptrDeclare;
   ptrDeclare << type << " *" << variable << " = ";
   ptrDeclare << "reinterpret_cast<" << type << " *>(" << ptrString << ");";
   ASSERT_TRUE(gInterpreter->Declare(ptrDeclare.str().c_str()));
}

#define DeclarePointer(type, variable, ptr)    \
   do {                                        \
      DeclarePointerImpl(type, variable, ptr); \
      if (::testing::Test::HasFatalFailure())  \
         return;                               \
   } while (0)

void EvaluateIntImpl(const char *expression, int *value)
{
   auto interpreterValue = gInterpreter->MakeInterpreterValue();
   ASSERT_TRUE(gInterpreter->Evaluate(expression, *interpreterValue));
   *value = interpreterValue->GetAsLong();
}

#define EXPECT_EVALUATE_EQ(expression, expected) \
   do {                                          \
      int _value;                                \
      EvaluateIntImpl(expression, &_value);      \
      if (::testing::Test::HasFatalFailure())    \
         return;                                 \
      EXPECT_EQ(expected, _value);               \
   } while (0)

void ProcessLineImpl(const char *line)
{
   TInterpreter::EErrorCode error = TInterpreter::kNoError;
   gInterpreter->ProcessLine(line, &error);
   ASSERT_EQ(error, TInterpreter::kNoError);
}

#define ProcessLine(line)                     \
   do {                                       \
      ProcessLineImpl(line);                  \
      if (::testing::Test::HasFatalFailure()) \
         return;                              \
   } while (0)
} // namespace

TEST(RNTupleEvolution, RemovedMember)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_member.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RemovedMember", "ptrRemovedMember", ptr);
      ProcessLine("ptrRemovedMember->fInt1 = 71;");
      ProcessLine("ptrRemovedMember->fInt2 = 82;");
      ProcessLine("ptrRemovedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedMember {
   int fInt1 = 1;
   // fInt2 member is missing!
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RemovedMember", "ptrRemovedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrRemovedMember->fInt3", 93);
}

TEST(RNTupleEvolution, ReorderedMembers)
{
   FileRaii fileGuard("test_ntuple_evolution_reordered_members.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct ReorderedMembers {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "ReorderedMembers").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("ReorderedMembers", "ptrReorderedMembers", ptr);
      ProcessLine("ptrReorderedMembers->fInt1 = 71;");
      ProcessLine("ptrReorderedMembers->fInt2 = 82;");
      ProcessLine("ptrReorderedMembers->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct ReorderedMembers {
   int fInt3 = 3;
   int fInt2 = 2;
   int fInt1 = 1;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("ReorderedMembers", "ptrReorderedMembers", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt2", 82);
   EXPECT_EVALUATE_EQ("ptrReorderedMembers->fInt3", 93);
}

TEST(RNTupleEvolution, RemovedBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_base_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedBase {
   int fBase = 1;
};
struct RemovedBaseIntermediate : public RemovedBase {
   int fIntermediate = 2;
};
struct RemovedBaseDerived : public RemovedBaseIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RemovedBaseDerived", "ptrRemovedBaseDerived", ptr);
      ProcessLine("ptrRemovedBaseDerived->fBase = 71;");
      ProcessLine("ptrRemovedBaseDerived->fIntermediate = 82;");
      ProcessLine("ptrRemovedBaseDerived->fDerived = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      {
         // TStreamerInfo::Build will report a warning for interpreted classes (but only for base classes).
         // See also https://github.com/root-project/root/issues/9371
         ROOT::TestSupport::CheckDiagsRAII diagRAII;
         diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                               /*matchFullMessage=*/false);
         writer.reset();
      }
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedBaseIntermediate {
   int fIntermediate = 2;
};
struct RemovedBaseDerived : public RemovedBaseIntermediate {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RemovedBaseDerived", "ptrRemovedBaseDerived", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fIntermediate", 2);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fDerived", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fIntermediate", 82);
   EXPECT_EVALUATE_EQ("ptrRemovedBaseDerived->fDerived", 93);
}
