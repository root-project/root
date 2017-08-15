# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog merged*.root a_*.root b_*.root input*.root freidt_merge.root mfile*.root mzfile*.root \
     mlz4file*.root mzlibfile*.root hsimple*.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += notmergeable

# Search for Rules.mk in roottest/scripts
# Algorithm:  Find the current working directory and remove everything after
#  '*roottest/'.  Append the path for Rules.mk from within roottest, which 
#  should be 'scripts/Rules.mk'.  The roottest path is saved in the
#  ROOTTEST_HOME variable for use by the SUBDIRECTORIES variable and is 
#  exported to eliminate the overhead of findding the path again during 
#  recursive calls of gmake.
# Since all makefiles should be under roottest or one of its
#  subdirectories and all recursions of gmake are called by 
#    'cd [DIR]; gmake ...'
#  this algorithm should not fail in finding /roottest/ in the
#  current working directory.  
# Issues:  This algorithm will fail if a makefile is called from outside the 
#  roottest folder, as in executing 'gmake -f ~/roottest/Makefile' from 
#  the home directory.

ifeq ($(strip $(ROOTTEST_HOME)),)
   export ROOTTEST_HOME := $(shell git rev-parse --show-toplevel)/
   ifeq ($(strip $(ROOTTEST_HOME)),)
      export ROOTTEST_HOME := $(shell expr $(CURDIR) : '\(.*/roottest/\)')
   endif
   ifeq ($(strip $(ROOTTEST_HOME)),)
      $(error The head of roottest was not found.  Set ROOTTEST_HOME)
   endif
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


# The name of this target should be changed according to the test being run.  Any name changes
#  should also be reflected in the TEST_TARGETS variable above.

mytest:
	$(CMDECHO) echo mytest > /dev/null


# Outputs a message if the FAIL variable is null

testWithFailure:
ifeq ($(FAIL),)
	$(WarnFailTest)
endif

# Remember that sometest.log can be made automatically from
#  from runsometest.C
# This target attempts to create a log file (which should be the output of running the test), and
#  compares its output with a prebuilt reference file.  The reference file should be copied from
#  the log files produced during what is considered a successful run of the test.  The testWithDiff
#  files should be renamed according to the test being run.
# By default the TestDiff variable is set to compare the .log and .ref files
# To run a diff and ignore whitespacing use $(TestDiffW)

testWithDiff: testWithDiff.log testWithDiff.ref
	$(TestDiff)

notmergeable.log: freidt_CDMeson_0000.root freidt_CDMeson_0001.root $(ROOT_CORE)
	$(CMDECHO) hadd -f freidt_merge.root  freidt_CDMeson_0000.root freidt_CDMeson_0001.root > notmergeable.log 2>&1

notmergeable: notmergeable.log
	$(TestDiff)

ifneq ($(ClingWorkAroundMissingAutoLoading),)
execFileMerger.log: CALLROOTEXE += -e 'gSystem->Load("libTreePlayer");'
endif

mfile1.root mfile2.root mfile3.root mfile4.root: execMergeMulti.success

# Default compression algorithm (default compression level - 1)
mfile1-4.root: execMergeMulti.success mfile1.root mfile2.root mfile3.root mfile4.root
	$(CMDECHO) ( hadd -f mfile1-2.root mfile1.root mfile2.root && hadd -f mfile3-4.root mfile3.root mfile4.root \
	     && hadd -f mfile1-4.root mfile1-2.root mfile3-4.root ) > mfile1-4.log 2>&1  \
	|| handleError.sh --log=mfile1-4.log --res=$$?

# ZLIB-based compression (compression level 6)
mzlibfile1-4.root: execMergeMulti.success mfile1.root mfile2.root mfile3.root mfile4.root
	$(CMDECHO) ( hadd -f106 mzlibfile1-2.root mfile1.root mfile2.root && hadd -f106 mzlibfile3-4.root mfile3.root mfile4.root \
	     && hadd -f106 mzlibfile1-4.root mzlibfile1-2.root mzlibfile3-4.root ) > mzlibfile1-4.log 2>&1  \
	|| handleError.sh --log=mzlibfile1-4.log --res=$$?

# LZMA-based compression (compression level 6)
mzfile1-4.root: execMergeMulti.success mfile1.root mfile2.root mfile3.root mfile4.root
	$(CMDECHO) ( hadd -f206 mzfile1-2.root mfile1.root mfile2.root && hadd -f206 mzfile3-4.root mfile3.root mfile4.root \
	     && hadd -f206 mzfile1-4.root mzfile1-2.root mzfile3-4.root ) > mzfile1-4.log 2>&1  \
	|| handleError.sh --log=mzfile1-4.log --res=$$?

# LZ4-based compression (compression level 6)
mlz4file1-4.root: execMergeMulti.success mfile1.root mfile2.root mfile3.root mfile4.root
	$(CMDECHO) ( hadd -f406 mlz4file1-2.root mfile1.root mfile2.root && hadd -f406 mlz4file3-4.root mfile3.root mfile4.root \
	     && hadd -f406 mlz4file1-4.root mlz4file1-2.root mlz4file3-4.root ) > mlz4file1-4.log 2>&1  \
	|| handleError.sh --log=mlz4file1-4.log --res=$$?

execTestMultiMerge.log: mfile1-4.root mzfile1-4.root mzlibfile1-4.root mlz4file1-4.root hsimple.root hsimple9.root hsimple9x2.root hsimple209.root hsimple409.root hsimpleK.root hsimpleK202.root hsimpleF.root

hsimple.root: $(ROOT_LOC)/tutorials/hsimple.C
	$(CMDECHO) ! $(CALLROOTEXE) -b -l -q $(ROOT_LOC)/tutorials/hsimple.C > hsimple.log 2>&1

# Default compression algorithm (compression level 9)
hsimple9.root: hsimple.root
	$(CMDECHO) hadd -f109 hsimple9.root hsimple.root > hsimple9.log 2>&1  \
     || handleError.sh --log=hsimple9.log --res=$$?

# Default compression algorithm (compression level 9)
hsimple9x2.root: hsimple.root
	$(CMDECHO) ( hadd -f109 hsimple9x2.root hsimple.root && hadd -f109 -a hsimple9x2.root hsimple.root )  > hsimple9x2.log 2>&1  \
     || handleError.sh --log=hsimple9x2.log --res=$$?

# LZMA-based compression (compression level 9)
hsimple209.root: hsimple.root
	$(CMDECHO) hadd -f209 hsimple209.root hsimple.root > hsimple209.log 2>&1  \
		|| handleError.sh --log=hsimple209.log --res=$$?

# LZ4-based compression (compression level 9)
hsimple409.root: hsimple.root
	$(CMDECHO) hadd -f409 hsimple409.root hsimple.root > hsimple409.log 2>&1  \
		|| handleError.sh --log=hsimple409.log --res=$$?

# Default compression algorithm
hsimpleK.root: hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root
	$(CMDECHO) hadd -fk hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple9x2.root > hsimpleK.log 2>&1  \
		|| handleError.sh --log=hsimpleK.log --res=$$?

# LZMA-based compression (compression level 2) 
hsimpleK202.root: hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root
	$(CMDECHO) hadd -fk202 hsimpleK202.root hsimpleK.root hsimple209.root hsimple409.root hsimple.root hsimple9.root hsimple9x2.root > hsimpleK202.log 2>&1  \
		|| handleError.sh --log=hsimpleK202.log --res=$$?

# LZ4-based compression (compression level 9)
hsimpleK409.root: hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root
	$(CMDECHO) hadd -fk409 hsimpleK409.root hsimpleK202.root hsimpleK.root hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root > hsimpleK409.log 2>&1  \
		|| handleError.sh --log=hsimpleK409.log --res=$$?

# Default compression algorithm
hsimpleF.root: hsimple.root hsimple9.root hsimple209.root hsimple409.root hsimple9x2.root
	$(CMDECHO) hadd -ff hsimpleF.root hsimple9.root hsimple209.root hsimple409.root hsimple.root hsimple9x2.root > hsimpleF.log 2>&1  \
		|| handleError.sh --log=hsimpleF.log --res=$$?


