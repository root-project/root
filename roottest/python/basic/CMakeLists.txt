if(pyroot)
  ROOTTEST_ADD_TEST(basic
                    MACRO PyROOT_basictests.py
                    COPY_TO_BUILDDIR ArgumentPassingCompiled.C ReturnValues.C SimpleClass.C ArgumentPassingInterpreted.C
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ ArgumentPassingCompiled.C+)

if(NOT MSVC OR win_broken_tests)
  ROOTTEST_ADD_TEST(datatype-numpy
                    MACRO PyROOT_datatypetest_numpy.py
                    COPY_TO_BUILDDIR DataTypes.C DataTypes.h
                    PRECMD ${ROOT_root_CMD} -b -q -e .L\ DataTypes.C+
                    PYTHON_DEPS numpy)
endif()
endif()
