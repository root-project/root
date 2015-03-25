# ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes classes.h SELECTION classes_selection.xml)
ROOTTEST_ADD_TEST(roottest-root-meta-genreflex-ROOT-5952
                  EXEC gendict.sh
                  COPY_TO_BUILDDIR classes_selection.xml classes.h gendict.sh
                  OUTREF selLog.ref)
