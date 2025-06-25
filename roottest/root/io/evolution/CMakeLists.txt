ROOTTEST_ADD_TESTDIRS()

if(ClingWorkAroundMissingDynamicScope)
  set(_myclassold_load -e "(void)gROOT->ProcessLine(\".L MyClassOld.cxx+\")")
endif()

ROOTTEST_COMPILE_MACRO(MyClassOld.cxx
                       FIXTURES_SETUP root-io-evolution-MyClassOld-fixture)

ROOTTEST_ADD_TEST(write
                  MACRO write.C
                  ROOTEXE_OPTS ${_myclassold_load}
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-MyClassOld-fixture
                  FIXTURES_SETUP root-io-evolution-write)

ROOTTEST_ADD_TEST(readold
                  MACRO readold.C
                  ROOTEXE_OPTS ${_myclassold_load}
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-write)


# this test was failing in Makefile too
ROOTTEST_COMPILE_MACRO(MyClass.C
                       FIXTURES_SETUP root-io-evolution-MyClass-fixture)

ROOTTEST_ADD_TEST(readnew
                  MACRO readnew.C
                  WILLFAIL
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-write root-io-evolution-MyClass-fixture)


ROOTTEST_COMPILE_MACRO(ClassConvOld.cxx
                       FIXTURES_SETUP root-io-evolution-ClassConvOld-fixture)

ROOTTEST_ADD_TEST(ClassConvWriteOld
                  MACRO ClassConvWriteOld.cxx
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-ClassConvOld-fixture
                  FIXTURES_SETUP root-io-evolution-ClassConvWriteOld-fixture)

ROOTTEST_ADD_TEST(ClassConvReadOld
                  MACRO ClassConvReadOld.cxx
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-ClassConvWriteOld-fixture)


ROOTTEST_COMPILE_MACRO(ClassConvNew.cxx
                       FIXTURES_SETUP root-io-evolution-ClassConvNew-fixture)

ROOTTEST_ADD_TEST(ClassConvWriteNew
                  MACRO ClassConvWriteNew.cxx
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-ClassConvNew-fixture
                  FIXTURES_SETUP root-io-evolution-ClassConvWriteNew-fixture)

ROOTTEST_ADD_TEST(ClassConvReadNew
                  MACRO ClassConvReadNew.cxx
                  OUTREF ClassConvReadNew.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-ClassConvWriteNew-fixture)


ROOTTEST_COMPILE_MACRO(ClassConvOther.cxx
                       LABELS longtest io
                       FIXTURES_SETUP root-io-evolution-ClassConvOther-fixture)

ROOTTEST_ADD_TEST(ClassConvReadOther
                  MACRO ClassConvReadOther.cxx
                  OUTREF ClassConvReadOther.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-ClassConvOther-fixture root-io-evolution-ClassConvWriteNew-fixture)


ROOTTEST_COMPILE_MACRO(typedefWrite.C
                       FIXTURES_SETUP root-io-evolution-typedefWrite-fixture)

ROOTTEST_ADD_TEST(typedefFile
                  MACRO typedefWrite.C+
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-typedefWrite-fixture
                  FIXTURES_SETUP root-io-evolution-typedefFile-fixture)

ROOTTEST_ADD_TEST(typedef
                  MACRO runtypedef.C
                  OUTREF typedef.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-typedefFile-fixture)


ROOTTEST_COMPILE_MACRO(skipWrite.C
                       FIXTURES_SETUP root-io-evolution-skipWrite-fixture)

ROOTTEST_ADD_TEST(skipFile
                  MACRO skipWrite.C+
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-skipWrite-fixture
                  FIXTURES_SETUP root-io-evolution-skipFile-fixture)


ROOTTEST_COMPILE_MACRO(skipRead.C
                       FIXTURES_SETUP root-io-evolution-skipRead-fixture)

ROOTTEST_ADD_TEST(skip
                  MACRO runSkip.C
                  OUTREF skip.ref
                  OUTCNVCMD sed -e s,offset=[[:space:]]*[0-9]*,offset=na,g
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-skipFile-fixture root-io-evolution-skipRead-fixture)

ROOTTEST_COMPILE_MACRO(boolUpdate.C
                       FIXTURES_SETUP root-io-evolution-boolUpdate-fixture)

ROOTTEST_ADD_TEST(boolUpdate
                  MACRO runboolUpdate.C
                  OUTREF boolUpdate.ref
                  COPY_TO_BUILDDIR boolUpdate.keeproot
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-boolUpdate-fixture)


ROOTTEST_ADD_TEST(Missing
                  MACRO runMissing.C
                  OUTREF Missing.ref
                  COPY_TO_BUILDDIR skim.root BcMC.root
                  LABELS longtest io)

foreach(indx RANGE 1 3)
  ROOTTEST_COMPILE_MACRO(fixarr${indx}.C
                         FIXTURES_SETUP root-io-evolution-fixarr${indx}-fixture)
endforeach()

ROOTTEST_ADD_TEST(fixarr1
                  MACRO fixarr1.C+
                  OUTREF fixarr1.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-fixarr1-fixture
                  FIXTURES_SETUP root-io-evolution-fixarrfile-fixture)

# tests fails on some Mac platforms generating stacktrace
if(NOT APPLE)
  ROOTTEST_ADD_TEST(fixarr2
                    MACRO fixarr2.C+
                    OUTREF fixarr2.ref
                    LABELS longtest io
                    FIXTURES_REQUIRED root-io-evolution-fixarr2-fixture root-io-evolution-fixarrfile-fixture)
endif()

ROOTTEST_ADD_TEST(fixarr3
                  MACRO fixarr3.C+
                  OUTREF fixarr3.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-fixarr3-fixture root-io-evolution-fixarrfile-fixture)


ROOTTEST_ADD_TEST(WriteFixedArrayOld
                  MACRO runWriteFixedArrayOld.C+
                  OUTREF WriteFixedArrayOld.ref
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-WriteFixedArrayOld-fixture)

ROOTTEST_ADD_TEST(WriteFixedArrayNew
                  MACRO runWriteFixedArrayNew.C+
                  OUTREF WriteFixedArrayNew.ref
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-WriteFixedArrayNew-fixture)

ROOTTEST_ADD_TEST(ReadFixedArrayOldWithNew
                  MACRO runReadFixedArrayOldWithNew.C+
                  OUTREF ReadFixedArrayOldWithNew.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-WriteFixedArrayOld-fixture)

ROOTTEST_ADD_TEST(ReadFixedArrayNewWithOld
                  MACRO runReadFixedArrayNewWithOld.C+
                  OUTREF ReadFixedArrayNewWithOld.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-WriteFixedArrayNew-fixture)


ROOTTEST_COMPILE_MACRO(Event_3.cxx
                       FIXTURES_SETUP root-io-evolution-Event_3-fixture)

ROOTTEST_ADD_TEST(ClonesArrayEvo
                  MACRO runClonesArrayEvo.C
                  OUTREF ClonesArrayEvo.ref
                  COPY_TO_BUILDDIR Event_2.root
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-Event_3-fixture)

ROOTTEST_COMPILE_MACRO(cms.cxx
                       FIXTURES_SETUP root-io-evolution-cms-fixture)

ROOTTEST_ADD_TEST(foreign310
                  MACRO runforeign310.C
                  OUTREF foreign310.ref
                  COPY_TO_BUILDDIR CMS.root
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-cms-fixture)


ROOTTEST_ADD_TEST(Skipbaseclass
                  MACRO execSkipbaseclass.cxx+
                  OUTREF execSkipbaseclass.ref
                  COPY_TO_BUILDDIR skipbaseclass.root
                  LABELS longtest io)

ROOTTEST_ADD_TEST(StlInheritOld
                  MACRO execStlInheritOld.cxx+
                  OUTREF execStlInheritOld.ref
                  LABELS longtest io
                  FIXTURES_SETUP root-io-evolution-StlInheritOld-fixture)

ROOTTEST_ADD_TEST(StlInheritNew
                  MACRO execStlInheritNew.cxx+
                  OUTREF execStlInheritNew.ref
                  LABELS longtest io
                  FIXTURES_REQUIRED root-io-evolution-StlInheritOld-fixture)


