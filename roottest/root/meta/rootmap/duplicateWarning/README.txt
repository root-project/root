17-08-2023

This test has been put in one directory by itself since it deals with rootmap files.
The warning messages that the interpreter can prompt while reading rootmap files is tested. Unfortunately, if any other test is put in this directory, all rootmaps are read and a warning is printed which 
will cause the test to fail.
