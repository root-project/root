if(ROOT_pyroot_FOUND)
  ROOTTEST_ADD_TEST(basic
                    MACRO PyROOT_basictests.py
                    COPY_TO_BUILDDIR ArgumentPassingCompiled.C ReturnValues.C SimpleClass.C ArgumentPassingInterpreted.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ ArgumentPassingCompiled.C+)

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(datatype
                    MACRO PyROOT_datatypetest.py
                    COPY_TO_BUILDDIR DataTypes.C DataTypes.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ DataTypes.C+
                    FIXTURES_SETUP python-basic-datatype-fixture
                    ENVIRONMENT CLING_STANDARD_PCH=none)

  ROOTTEST_ADD_TEST(datatype-numpy
                    MACRO PyROOT_datatypetest_numpy.py
                    FIXTURES_REQUIRED python-basic-datatype-fixture
                    PYTHON_DEPS numpy
                    ENVIRONMENT CLING_STANDARD_PCH=none)
endif()

  ROOTTEST_ADD_TEST(operator
                    MACRO PyROOT_operatortests.py
                    COPY_TO_BUILDDIR Operators.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Operators.C+)

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(overload
                    MACRO PyROOT_overloadtests.py
                    COPY_TO_BUILDDIR Overloads.C Overloads.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Overloads.C+
                    ENVIRONMENT CLING_STANDARD_PCH=none)
endif()
endif()
