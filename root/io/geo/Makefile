# This is a template for all makefiles.

#Set the list of files to be deleted by clean (Targets can also be specified).:
CLEAN_TARGETS += $(ALL_LIBRARIES) *.log unpack.last \
maketar \
alice_all.C alice_itsd.C alice.root \
atlas_all.C atlas_cryo.C atlas.root \
cms_cmse.C cms.root \
ams.C ams_all.C \
btev.C btev_all.C \
gem.C gem_all.C \
hades_all.C hades.root \
phenix.C phenix_all.C \
aleph_all.C alice_itsv.C atlas_sctt.C brahms.C brahms_all.C \
cdf_all.C cdf_svxc.C cms_calo.C d0.C d0_all.C e907.C e907_all.C \
phobos.C phobos_all.C star_all.C star_svtt.C tesla.C tesla_all.C \
lhcb.C lhcb_all.C lhcbfull_all.C lhcbnobool_all.C lhcbfull.root lhcbnobool.root \
aleph.root  brahms.root  cdf.root  e907.root  star.root \
ams.root    btev.root    d0.root   gem.root   phobos.root  tesla.root \

# Set the list of target to make while testing.  By default, mytest is the
# only target added.  If the name of the target is changed in the rules then
# the name should be changed accordingly in this list.

TEST_TARGETS += readGeo mytest

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
	export ROOTTEST_HOME=$(shell expr $(PWD) : '\(.*/roottest/\)')
endif

include $(ROOTTEST_HOME)/scripts/Rules.mk


# Uncomment the following line to have gmake called recursively on subdirectories which contain a file named "Makefile".
#SUBDIRS = $(shell $(ROOTTEST_HOME)scripts/subdirectories .)

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

%.root: %.C
	$(CMDECHO) root.exe -b -l -q export.C\(\"$*\"\) > $*.log 2>&1

#ams.C, brahms.C, btev.C, d0.C, e907.C, gem.C, phenix.C, phobos.C, tesla.C

rootfiles: ams.root aleph.root  brahms.root  btev.root cdf.root d0.root  \
	e907.root gem.root phobos.root star.root tesla.root

alice_all.C : unpack.last
alice_itsd.C : unpack.last
alice_itsv.C : unpack.last
alice.root : unpack.last
atlas_all.C : unpack.last
atlas_cryo.C : unpack.last
atlas_sctt.C : unpack.last
atlas.root : unpack.last
cms_cmse.C : unpack.last
cms_calo.C : unpack.last
cms.root : unpack.last
ams.C : unpack.last
ams_all.C : unpack.last
btev.C : unpack.last
btev_all.C : unpack.last
gem.C : unpack.last
gem_all.C : unpack.last
hades_all.C : unpack.last
hades.root : unpack.last
phenix.C : unpack.last
phenix_all.C : unpack.last
phobos_all.C : unpack.last
phobos.C : unpack.last
brahms_all.C : unpack.last
brahms.C : unpack.last
d0_all.C : unpack.last
d0.C : unpack.last
tesla_all.C : unpack.last
tesla.C : unpack.last
e907_all.C : unpack.last
e907.C : unpack.last
cdf_all.C : unpack.last
cdf_svxc.C : unpack.last
cdf.root : unpack.last
aleph_all.C : unpack.last
aleph.root : unpack.last
star_all.C : unpack.last
star_svtt.C : unpack.last
star.root : unpack.last
lhcb.C : unpack.last
lhcb_all.C : unpack.last
lhcbfull_all.C : unpack.last
lhcbnobool_all.C : unpack.last
lhcbfull.root : unpack.last
lhcbnobool.root : unpack.last

FILES_IN_TAR = alice_all.C alice_itsd.C alice_itsv.C alice.root atlas_all.C atlas_cryo.C atlas_sctt.C atlas.root cms_cmse.C cms_calo.C cms.root ams.C ams_all.C btev.C btev_all.C gem.C gem_all.C hades_all.C hades.root phenix.C phenix_all.C phobos_all.C phobos.C brahms_all.C brahms.C d0_all.C d0.C tesla_all.C tesla.C e907_all.C e907.C cdf_all.C cdf_svxc.C cdf.root aleph_all.C aleph.root star_all.C star_svtt.C star.root lhcb.C lhcb_all.C lhcbfull_all.C lhcbnobool_all.C lhcbfull.root lhcbnobool.root

unpack.last: geometries.tar.gz
	$(CMDECHO)gunzip -c geometries.tar.gz | tar xf - && touch unpack.last && \
	    touch $(FILES_IN_TAR) && \
            ($(MAKE) rootfiles)

readGeo.log: unpack.last ams.root brahms.root btev.root d0.root e907.root gem.root phobos.root tesla.root

readGeo: readGeo.log
	$(TestDiff)
