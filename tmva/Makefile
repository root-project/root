######################################################################
# Project: TMVA - Toolkit for Multivariate Data Analysis             #
# Code   : Source                                                    #
###################################################################### 

SHELL=bash

MAKEFLAGS = --no-print-directory -r -s --warn-undefined-variables --debug

include Makefile.arch

# Internal configuration
PACKAGE=TMVA
LD_LIBRARY_PATH := $(shell root-config --libdir):$(LD_LIBRARY_PATH)
OBJDIR    = obj
DEPDIR    = $(OBJDIR)/dep
LIBDIR    = lib
VPATH     = $(OBJDIR)
INCLUDES += -I./

DICTHEAD  = src/$(PACKAGE)_Dict.h
DICTFILE  = src/$(PACKAGE)_Dict.C
DICTOBJ   = $(OBJDIR)/$(PACKAGE)_Dict.o
DICTLDEF  = inc/LinkDef.h
SKIPCPPLIST = 
SKIPHLIST =  $(DICTHEAD) $(DICTLDEF)
LIBFILE   = lib/lib$(PACKAGE).a
SHLIBFILE = lib$(PACKAGE).$(DllSuf)
DLLIBFILE = lib/lib$(PACKAGE).dll
ROOTMAP   = lib/lib$(PACKAGE).rootmap
UNAME = $(shell uname)

default: shlib linklib

# List of all source files to build
HLIST   = $(filter-out $(SKIPHLIST),$(wildcard inc/*.h))
CPPLIST = $(filter-out $(SKIPCPPLIST),$(patsubst src/%,%,$(wildcard src/*.$(SrcSuf))))
DICTHLIST = $(HLIST)

# List of all object files to build
OLIST=$(patsubst %.cxx,%.o,$(CPPLIST))
DEPLIST=$(foreach var,$(CPPLIST:.$(SrcSuf)=.d),$(DEPDIR)/$(var))

# Implicit rule to compile all classes
%.o : src/%.cxx 
	if [[ ( ! -e TMVA ) ]]; then \
		ln -sf inc TMVA; \
	fi
	@printf "Compiling $< ... "
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -ggdb -c $< -o $(OBJDIR)/$(notdir $@)
	@echo "Done"

# Rule to make the dictionary
dict: $(DICTOBJ)

$(DICTFILE):  $(DICTHLIST) $(DICTLDEF)
	@echo "Generating dictionary $@" 
	$(shell root-config --exec-prefix)/bin/rootcint -f $@ -c -p $(INCLUDES) $^

$(DICTOBJ): $(DICTFILE)
	@echo "Compiling dictionary $<"
	@mkdir -p $(OBJDIR)
	@$(CXX) $(INCLUDES) $(CXXFLAGS) -g -c  -o $@ $<

# Rule to set up a symbolic links to the created shared library
linklib:
	if [[ ( ( `root-config --platform` == "macosx" ) && \
			( ! -e lib/lib$(PACKAGE).1.dylib ) ) || \
			( ! -e lib/lib$(PACKAGE).1.so ) ]]; then \
		printf "Setting up soft links to the TMVA library ... "; \
		ln -sf $(SHLIBFILE) $(LIBDIR)/lib$(PACKAGE).1.so; \
		if [[ `root-config --platform` == "macosx" ]]; then \
			ln -sf $(SHLIBFILE) $(LIBDIR)/lib$(PACKAGE).1.dylib; \
		fi; \
		echo "Done"; \
	fi

##############################
# The dependencies section   
# - the purpose of the .d files is to keep track of the
#   header file dependence
# - this can be achieved using the makedepend command 
##############################
# .d tries to pre-process .cc
ifneq ($(MAKECMDGOALS),clean)
-include $(DEPLIST)
endif

$(DEPDIR)/%.d: src/%.$(SrcSuf)
	@mkdir -p $(DEPDIR)
	if test -f $< ; then \
		printf "Building $(@F) ... "; \
		$(SHELL) -ec '`root-config --exec-prefix`/bin/rmkdepend -f- -Y -w 3000 -- -I../include -- $< 2> /dev/null 1| sed -e "s-\(.*\).o: .*-$(DEPDIR)\/\1.d &-" > $@'; \
		rm -f $@.bak; \
		echo "Done"; \
	fi

# Rule to combine objects into a library
$(LIBFILE): $(DICTOBJ) $(OLIST)
	@printf "Making static library $(LIBFILE) ... "
	@rm -f $(LIBFILE)
	@ar q $(LIBFILE) $(addprefix $(OBJDIR)/,$(OLIST) $(DICTOBJ)
	@ranlib $(LIBFILE)
	@echo "Done"

# Rule to combine objects into a unix shared library
$(LIBDIR)/$(SHLIBFILE): $(OLIST) $(DICTOBJ)
	@printf "Building shared library $(LIBDIR)/$(SHLIBFILE) ... "
	@mkdir -p $(LIBDIR)
	@rm -f $(LIBDIR)/$(SHLIBFILE)
	@$(LD) -L$(shell root-config --libdir) $(SOFLAGS) $(addprefix $(OBJDIR)/,$(OLIST)) $(DICTOBJ) -o $(LIBDIR)/$(SHLIBFILE) -lMinuit -lXMLIO
#	ln -fs $(SHLIBFILE) lib/lib$(PACKAGE).1.so
	@echo "Done"

# Rule to combine objects into a unix shared library
$(ROOTMAP): $(DICTLDEF)
	@printf "Building $(ROOTMAP) ... "
	@mkdir -p $(LIBDIR)
	rlibmap -f -o $@ -l lib$(PACKAGE).1.$(DllSuf) -d libMinuit.so libMLP.so libMatrix.so libTree.so libGraf.so libTreePlayer.so libXMLIO.so -c $<
	@echo "Done"

# Rule to combine objects into a windows shared library
$(DLLIBFILE): $(OLIST) $(DICTOBJ)
	@printf "Making dll file $(DLLIBFILE) ... "
	@rm -f $(DLLIBFILE)
	$(LD) -Wl,--export-all-symbols -Wl,--export-dynamic -Wl,--enable-auto-import -Wl,-Bdynamic -shared --enable-auto-image-base -Wl,-soname -o $(DLLIBFILE) -Wl,--whole-archive $(addprefix $(OBJDIR)/,$(OLIST) $(patsubst %.$(SrcSuf),%.o,$(DICTFILE))) -Wl,--no-whole-archive -L$(ROOTSYS)/lib -lCore -lCint -lHist -lGraf -lGraf3d -lTree -lRint -lPostscript -lMatrix -lMinuit -lPhysics -lHtml -lXMLIO -lm
	@echo "Done"

# Useful build targets
lib: $(LIBFILE) 

shlib: $(LIBDIR)/$(SHLIBFILE) $(ROOTMAP)

winlib: $(DLLIBFILE)

vars:
	#echo $(patsubst src/%,%,$(wildcard src/*.$(SrcSuf)))
	echo $(LIBDIR)/$(SHLIBFILE)

clean:
	rm -rf obj
	rm -rf lib
	rm -f TMVA 
	rm -f $(DICTFILE) $(DICTHEAD)
	rm -f $(OBJDIR)/*.o
	rm -f $(DEPDIR)/*.d
	rm -f $(LIBFILE)
	rm -f $(LIBDIR)/$(SHLIBFILE)
	rm -f lib/lib$(PACKAGE).1.so
	rm -f lib/lib$(PACKAGE).1.dylib
	rm -f $(ROOTMAP)
	rm -f $(DLLIBFILE)

distclean:
	rm -rf obj 
	rm -f *~
	rm -f $(DICTFILE) $(DICTHEAD)
	rm -f $(LIBFILE)
	rm -f $(LIBDIR)/$(SHLIBFILE	)
	rm -f lib/lib$(PACKAGE).1.so
	rm -f lib/lib$(PACKAGE).1.dylib
	rm -f $(ROOTMAP)
	rm -f $(DLLIBFILE)

.PHONY : winlib shlib lib default clean

