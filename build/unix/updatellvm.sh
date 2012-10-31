#! /bin/sh

if [ $# -ne 1 ]; then
   echo "Must specify LLVM/clang revision number"
   exit 1
fi

GOLDEN=$1

cd
rm -rf llvm-export
svn export -r $GOLDEN http://llvm.org/svn/llvm-project/llvm/trunk llvm-export

# don't need the test directory
rm -rf llvm-export/test

rm -rf clang-export
svn export -r $GOLDEN http://llvm.org/svn/llvm-project/cfe/trunk clang-export

# don't need the test directory
rm -rf clang-export/test

# will automatically commit into vendor branch
$ROOTSYS/build/unix/svn_load_dirs.pl https://root.cern.ch/svn/root/vendors llvm llvm-export -m "Update LLVM to r$GOLDEN."

# will automatically commit into vendor branch
$ROOTSYS/build/unix/svn_load_dirs.pl https://root.cern.ch/svn/root/vendors clang clang-export -m "Update Clang to r$GOLDEN."

# Both of the 'svn_load_dirs.pl' steps should have given you a final revision
# number for their checkins. We need to merge those changes into 'trunk/'.
##cd $ROOTSYS/interpreter/llvm/src
##svn merge -c $LLVM_REV https://root.cern.ch/svn/root/vendors/llvm .

##cd $ROOTSYS/interpreter/llvm/src/tools/clang
##svn merge -c $CLANG_REV https://root.cern.ch/svn/root/vendors/clang .

# now apply the patches from 'interpreter/cling/patches'
# and run 'make all-llvm' in '$ROOTSYS' and next 'make test' from
# 'interpreter/llvm/obj'. 
