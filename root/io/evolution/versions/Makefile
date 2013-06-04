# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log *.clog *.rootmap result.root myclass*.root veclong*.root

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += AddVersion VecLong attmarker

ifeq ($(strip $(ROOTTEST_HOME)),)
	export ROOTTEST_HOME:=$(shell expr $(CURDIR) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk

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

lib1.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE)  -b -l -q 'compile.cxx("1")' > lib1.log 2>&1

lib2.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("2")' > lib2.log 2>&1

lib3.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("3")' > lib3.log 2>&1

lib4.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("4")' > lib4.log 2>&1

lib5.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("5")' > lib5.log 2>&1

lib6.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("6")' > lib6.log 2>&1

lib7.$(DllSuf) : compile.cxx myclass.cxx myclass.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("7")' > lib7.log 2>&1

myclass1.root: wcomp.cxx lib1.$(DllSuf) 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("1")' > myclass1.log 2>&1 

myclass2.root: wcomp.cxx lib2.$(DllSuf) 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("2")' > myclass2.log 2>&1 

myclass3.root: wcomp.cxx lib3.$(DllSuf) 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("3")' > myclass3.log 2>&1 

myclass6.root: wcomp.cxx lib6.$(DllSuf) 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("6")' > myclass6.log 2>&1 

myclass7.root: wcomp.cxx lib7.$(DllSuf) 
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("7")' > myclass7.log 2>&1 

FILES = myclass1.root myclass2.root myclass3.root myclass6.root myclass7.root
LIBS = lib1.$(DllSuf) lib2.$(DllSuf) lib3.$(DllSuf) lib4.$(DllSuf) lib5.$(DllSuf) \
        lib6.$(DllSuf) lib7.$(DllSuf)

AddVersion.log: $(FILES) $(LIBS)
	$(CMDECHO) ( $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"1")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"2")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"3")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"4")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"5")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(1,"6")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"1")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"2")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"3")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"4")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"5")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(2,"6")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(3,"6")' \
            && $(CALLROOTEXE) -b -l -q 'runAddVersion.cxx(3,"7")' \
	    ) 2>&1 | sed -e 's@offset=  [0-9]*@offset= n/a@' > AddVersion.log

AddVersion: AddVersion.log AddVersion.ref
	$(TestDiff)

libveclong64_h1.$(DllSuf) : compile.cxx myclass.cxx myclass.h veclong64.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("1","veclong64.h")' > libveclong64_h1.log 2>&1

libveclong64_h2.$(DllSuf) : compile.cxx myclass.cxx myclass.h veclong64.h
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'compile.cxx("2","veclong64.h")' > libveclong64_h2.log 2>&1

veclong-1.root: libveclong64_h1.$(DllSuf) wcomp.cxx
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("1","veclong64.h")' > veclong-1.log 2>&1 

veclong-2.root: libveclong64_h2.$(DllSuf) wcomp.cxx
	$(CMDECHO) $(CALLROOTEXE) -b -l -q 'wcomp.cxx("2","veclong64.h")' > veclong-2.log 2>&1 

VECFILES = veclong-1.root veclong-2.root
VECLIBS = libveclong64_h1.$(DllSuf) libveclong64_h2.$(DllSuf)

VecLong.log: $(VECFILES) $(VECLIBS)
	$(CMDECHO) ( $(CALLROOTEXE) -b -l -q 'runVecLong.cxx(1,"1")' \
            && $(CALLROOTEXE) -b -l -q 'runVecLong.cxx(1,"2")' \
	    ) 2>&1 | sed -e 's@offset=  [0-9]*@offset= n/a@' > VecLong.log

VecLong: VecLong.log VecLong.ref64 VecLong.ref32
	$(TestDiff)

attmarker.log: hsimple14.root hsimple20.root
	$(CMDECHO) hadd -f result.root hsimple14.root hsimple20.root > attmarker.log 2>&1

attmarker: attmarker.log
	$(TestDiff)

