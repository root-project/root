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

TEST(RNTupleEvolution, AddedMember)
{
   FileRaii fileGuard("test_ntuple_evolution_added_member.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMember {
   int fInt1 = 1;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedMember", "ptrAddedMember", ptr);
      ProcessLine("ptrAddedMember->fInt1 = 71;");
      ProcessLine("ptrAddedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMember", "ptrAddedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt2", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMember->fInt3", 93);
}

TEST(RNTupleEvolution, AddedMemberObject)
{
   FileRaii fileGuard("test_ntuple_evolution_added_member_object.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberObjectI {
   int fInt;
};
struct AddedMemberObject {
   AddedMemberObjectI fMember1{1};
   AddedMemberObjectI fMember3{3};
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedMemberObject").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedMemberObject", "ptrAddedMemberObject", ptr);
      ProcessLine("ptrAddedMemberObject->fMember1.fInt = 71;");
      ProcessLine("ptrAddedMemberObject->fMember3.fInt = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      {
         // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
         // See also https://github.com/root-project/root/issues/9371
         ROOT::TestSupport::CheckDiagsRAII diagRAII;
         diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                               /*matchFullMessage=*/false);
         writer.reset();
      }
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedMemberObjectI {
   int fInt;
};
struct AddedMemberObject {
   AddedMemberObjectI fMember1{1};
   AddedMemberObjectI fMember2{2};
   AddedMemberObjectI fMember3{3};
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedMemberObject", "ptrAddedMemberObject", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember1.fInt", 1);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember2.fInt", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember3.fInt", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember1.fInt", 71);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember2.fInt", 2);
   EXPECT_EVALUATE_EQ("ptrAddedMemberObject->fMember3.fInt", 93);
}

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

TEST(RNTupleEvolution, RenamedMember)
{
   // Without explicit detection support for renamings, a renamed member is handled as a removal from the original
   // version (which is simply ignored) and an added member (which is defaulted).
   FileRaii fileGuard("test_ntuple_evolution_renamed_member.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMember {
   int fInt1 = 1;
   int fInt2 = 2;
   int fInt3 = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedMember").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("RenamedMember", "ptrRenamedMember", ptr);
      ProcessLine("ptrRenamedMember->fInt1 = 71;");
      ProcessLine("ptrRenamedMember->fInt2 = 82;");
      ProcessLine("ptrRenamedMember->fInt3 = 93;");
      writer->Fill();

      // Reset / close the writer and flush the file.
      writer.reset();
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMember {
   int fInt1 = 1;
   int fInt2Renamed = 2;
   int fInt3 = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("RenamedMember", "ptrRenamedMember", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt1", 1);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt2Renamed", 2);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt3", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt1", 71);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt2Renamed", 2);
   EXPECT_EVALUATE_EQ("ptrRenamedMember->fInt3", 93);
}

TEST(RNTupleEvolution, RenamedMemberClass)
{
   // RNTuple currently does not support automatic schema evolution when a class is renamed.
   FileRaii fileGuard("test_ntuple_evolution_renamed_member_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMemberClass1 {
   int fInt = 1;
};
struct RenamedMemberClass {
   RenamedMemberClass1 fMember;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedMemberClass").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      // Reset / close the writer and flush the file.
      {
         // TStreamerInfo::Build will report a warning for interpreted classes (but only for members).
         // See also https://github.com/root-project/root/issues/9371
         ROOT::TestSupport::CheckDiagsRAII diagRAII;
         diagRAII.optionalDiag(kWarning, "TStreamerInfo::Build", "has no streamer or dictionary",
                               /*matchFullMessage=*/false);
         writer.reset();
      }
   });

   ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedMemberClass2 {
   int fInt = 1;
};
struct RenamedMemberClass {
   RenamedMemberClass2 fMember;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, AddedBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_added_base_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedBaseIntermediate {
   int fIntermediate = 2;
};
struct AddedBaseDerived : public AddedBaseIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedBaseDerived", "ptrAddedBaseDerived", ptr);
      ProcessLine("ptrAddedBaseDerived->fIntermediate = 82;");
      ProcessLine("ptrAddedBaseDerived->fDerived = 93;");
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
struct AddedBase {
   int fBase = 1;
};
struct AddedBaseIntermediate : public AddedBase {
   int fIntermediate = 2;
};
struct AddedBaseDerived : public AddedBaseIntermediate {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedBaseDerived", "ptrAddedBaseDerived", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fBase", 1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fIntermediate", 2);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fDerived", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fBase", 1);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fIntermediate", 82);
   EXPECT_EVALUATE_EQ("ptrAddedBaseDerived->fDerived", 93);
}

TEST(RNTupleEvolution, AddedSecondBaseClass)
{
   FileRaii fileGuard("test_ntuple_evolution_added_second_base_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct AddedSecondBaseFirst {
   int fFirstBase = 1;
};
struct AddedSecondBaseDerived : public AddedSecondBaseFirst {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "AddedSecondBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
      writer->Fill();

      void *ptr = writer->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
      DeclarePointer("AddedSecondBaseDerived", "ptrAddedSecondBaseDerived", ptr);
      ProcessLine("ptrAddedSecondBaseDerived->fFirstBase = 71;");
      ProcessLine("ptrAddedSecondBaseDerived->fDerived = 93;");
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
struct AddedSecondBaseFirst {
   int fFirstBase = 1;
};
struct AddedSecondBaseSecond {
   int fSecondBase = 2;
};
struct AddedSecondBaseDerived : public AddedSecondBaseFirst, public AddedSecondBaseSecond {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   ASSERT_EQ(2, reader->GetNEntries());

   void *ptr = reader->GetModel().GetDefaultEntry().GetPtr<void>("f").get();
   DeclarePointer("AddedSecondBaseDerived", "ptrAddedSecondBaseDerived", ptr);

   reader->LoadEntry(0);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fFirstBase", 1);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fSecondBase", 2);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fDerived", 3);

   reader->LoadEntry(1);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fFirstBase", 71);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fSecondBase", 2);
   EXPECT_EVALUATE_EQ("ptrAddedSecondBaseDerived->fDerived", 93);
}

TEST(RNTupleEvolution, PrependSecondBaseClass)
{
   // For RNTuple, this case looks like the first base class was renamed. It would be confusing for users to
   // automatically evolve this case, even if the member fields and on-disk columns are compatible.
   FileRaii fileGuard("test_ntuple_evolution_prepend_second_base_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct PrependSecondBaseFirst {
   int fFirstBase = 1;
};
struct PrependSecondBaseDerived : public PrependSecondBaseFirst {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "PrependSecondBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
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
struct PrependSecondBaseFirst {
   int fFirstBase = 1;
};
struct PrependSecondBaseSecond {
   int fSecondBase = 2;
};
struct PrependSecondBaseDerived : public PrependSecondBaseSecond, public PrependSecondBaseFirst {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
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

TEST(RNTupleEvolution, RemovedIntermediateClass)
{
   FileRaii fileGuard("test_ntuple_evolution_removed_intermediate_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RemovedIntermediateBase {
   int fBase = 1;
};
struct RemovedIntermediate : public RemovedIntermediateBase {
   int fIntermediate = 2;
};
struct RemovedIntermediateDerived : public RemovedIntermediate {
   int fDerived = 3;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RemovedIntermediateDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
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
struct RemovedIntermediateBase {
   int fBase = 1;
};
struct RemovedIntermediateDerived : public RemovedIntermediateBase {
   int fDerived = 3;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}

TEST(RNTupleEvolution, RenamedBaseClass)
{
   // RNTuple currently does not support automatic schema evolution when a class is renamed.
   FileRaii fileGuard("test_ntuple_evolution_renamed_base_class.root");

   WriteOldInFork([&] {
      // The child process writes the file and exits, but the file must be preserved to be read by the parent.
      fileGuard.PreserveFile();

      ASSERT_TRUE(gInterpreter->Declare(R"(
struct RenamedBase1 {
   int fBase = 1;
};
struct RenamedBaseDerived : public RenamedBase1 {
   int fDerived = 2;
};
)"));

      auto model = RNTupleModel::Create();
      model->AddField(RFieldBase::Create("f", "RenamedBaseDerived").Unwrap());

      auto writer = RNTupleWriter::Recreate(std::move(model), "ntpl", fileGuard.GetPath());
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
struct RenamedBase2 {
   int fBase = 1;
};
struct RenamedBaseDerived : public RenamedBase2 {
   int fDerived = 2;
};
)"));

   auto reader = RNTupleReader::Open("ntpl", fileGuard.GetPath());
   try {
      reader->GetModel();
      FAIL() << "model reconstruction should fail";
   } catch (const RException &err) {
      EXPECT_THAT(err.what(), testing::HasSubstr("incompatible type name for field"));
   }
}
