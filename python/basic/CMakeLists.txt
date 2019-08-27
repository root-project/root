if(ROOT_python_FOUND)
  ROOTTEST_ADD_TEST(basic
                    MACRO PyROOT_basictests.py
                    COPY_TO_BUILDDIR ArgumentPassingCompiled.C ReturnValues.C SimpleClass.C ArgumentPassingInterpreted.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ ArgumentPassingCompiled.C+
                    ENVIRONMENT EXP_PYROOT=${exp_pyroot})

  ROOTTEST_ADD_TEST(datatype
                    MACRO PyROOT_datatypetest.py
                    COPY_TO_BUILDDIR DataTypes.C DataTypes.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ DataTypes.C+
                    ${PYTESTS_WILLFAIL})

  ROOTTEST_ADD_TEST(operator
                    MACRO PyROOT_operatortests.py
                    COPY_TO_BUILDDIR Operators.C
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Operators.C+)

  ROOTTEST_ADD_TEST(overload
                    MACRO PyROOT_overloadtests.py
                    COPY_TO_BUILDDIR Overloads.C Overloads.h
                    PRECMD ${ROOT_root_CMD} -b -q -l -e .L\ Overloads.C+
                    ENVIRONMENT EXP_PYROOT=${exp_pyroot})
endif()
