include_directories(${CMAKE_CURRENT_SOURCE_DIR}/inc)

ROOTTEST_GENERATE_REFLEX_DICTIONARY(TemplateRedecl
                                    inc/RedeclTrigger.h
                                    SELECTION inc/redecl_selection.xml
                                    FIXTURES_SETUP root-meta-redeclUnload-TemplateRedecl-fixture)

ROOTTEST_ADD_TEST(templateRedeclUnload
                  MACRO testTemplateRedeclUnload.C
                  ROOTEXE_OPTS -e "gInterpreter->AddIncludePath(\"-I${CMAKE_CURRENT_SOURCE_DIR}\")"
                  OUTREF TemplateRedeclUnload.ref
                  FIXTURES_REQUIRED root-meta-redeclUnload-TemplateRedecl-fixture)
