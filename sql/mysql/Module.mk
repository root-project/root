# Module.mk for mysql module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := mysql
MODDIR       := $(ROOT_SRCDIR)/sql/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MYSQLDIR     := $(MODDIR)
MYSQLDIRS    := $(MYSQLDIR)/src
MYSQLDIRI    := $(MYSQLDIR)/inc

##### libRMySQL #####
MYSQLL       := $(MODDIRI)/LinkDef.h
MYSQLDS      := $(call stripsrc,$(MODDIRS)/G__MySQL.cxx)
MYSQLDO      := $(MYSQLDS:.cxx=.o)
MYSQLDH      := $(MYSQLDS:.cxx=.h)

MYSQLH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MYSQLS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MYSQLO       := $(call stripsrc,$(MYSQLS:.cxx=.o))

MYSQLDEP     := $(MYSQLO:.o=.d) $(MYSQLDO:.o=.d)

MYSQLLIB     := $(LPATH)/libRMySQL.$(SOEXT)
MYSQLMAP     := $(MYSQLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MYSQLH))
ALLLIBS     += $(MYSQLLIB)
ALLMAPS     += $(MYSQLMAP)

# include all dependency files
INCLUDEFILES += $(MYSQLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MYSQLDIRI)/%.h
		cp $< $@

$(MYSQLLIB):    $(MYSQLO) $(MYSQLDO) $(ORDER_) $(MAINLIBS) $(MYSQLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRMySQL.$(SOEXT) $@ "$(MYSQLO) $(MYSQLDO)" \
		   "$(MYSQLLIBEXTRA) $(MYSQLLIBDIR) $(MYSQLCLILIB)"

$(call pcmrule,MYSQL)
	$(noop)

$(MYSQLDS):     $(MYSQLH) $(MYSQLL) $(ROOTCLINGEXE) $(call pcmdep,MYSQL)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,MYSQL) -c $(MYSQLINCDIR:%=-I%) $(MYSQLH) $(MYSQLL)

$(MYSQLMAP):    $(MYSQLH) $(MYSQLL) $(ROOTCLINGEXE) $(call pcmdep,MYSQL)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(MYSQLDS) $(call dictModule,MYSQL) -c $(MYSQLINCDIR:%=-I%) $(MYSQLH) $(MYSQLL)

all-$(MODNAME): $(MYSQLLIB)

clean-$(MODNAME):
		@rm -f $(MYSQLO) $(MYSQLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MYSQLDEP) $(MYSQLDS) $(MYSQLDH) $(MYSQLLIB) $(MYSQLMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(MYSQLO) $(MYSQLDO): CXXFLAGS += $(MYSQLINCDIR:%=-I%)
