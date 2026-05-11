ROOTTEST_ADD_TEST(zombie
                  COPY_TO_BUILDDIR crashed.root
                  MACRO assertZombie.C
                  ERRREF assertZombie.ref)

ROOTTEST_ADD_TEST(empty
                  MACRO runempty.C
                  OUTREF empty.ref)

# corrupted_recover.root contains an otherwise-valid ROOT file that has been tweaked in the following way:
# - fEND is changed to a larger size than the file, to trigger the recover;
# - a fake key has been appended, that has a className "TH1F" and an object name whose string length is 2 GiB long.
# The test verifies that Recover doesn't try to read out of bounds and refuses to read the string, while still
# properly recovering the first two valid keys.
ROOTTEST_ADD_TEST(corrupted_objname
                  COPY_TO_BUILDDIR corrupted_recover.root
                  COMMAND ${ROOT_root_CMD} -q corrupted_recover.root
                  PASSREGEX "given buffer is too small \\(992 B remaining, need at least 2147483646 more\\)")
