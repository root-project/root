# Module.mk for mysql module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := mysql
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MYSQLDIR     := $(MODDIR)
MYSQLDIRS    := $(MYSQLDIR)/src
MYSQLDIRI    := $(MYSQLDIR)/inc

##### libMySQL #####
MYSQLL       := $(MODDIRI)/LinkDef.h
MYSQLDS      := $(MODDIRS)/G__MySQL.cxx
MYSQLDO      := $(MYSQLDS:.cxx=.o)
MYSQLDH      := $(MYSQLDS:.cxx=.h)

MYSQLH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MYSQLS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MYSQLO       := $(MYSQLS:.cxx=.o)

MYSQLDEP     := $(MYSQLO:.o=.d)

MYSQLLIB     := $(LPATH)/libMySQL.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MYSQLH))
ALLLIBS     += $(MYSQLLIB)

# include all dependency files
INCLUDEFILES += $(MYSQLDEP)

##### local rules #####
include/%.h:    $(MYSQLDIRI)/%.h
		cp $< $@

$(MYSQLLIB):    $(MYSQLO) $(MYSQLDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMySQL.$(SOEXT) $@ "$(MYSQLO) $(MYSQLDO)" \
		   "$(MYSQLLIBEXTRA) $(MYSQLLIBDIR)/libmysqlclient.a"

$(MYSQLDS):     $(MYSQLH) $(MYSQLL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@$(ROOTCINTTMP) -f $@ -c $(MYSQLH) $(MYSQLL)

$(MYSQLDO):     $(MYSQLDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I$(MYSQLINCDIR) -I. -o $@ -c $<

all-mysql:      $(MYSQLLIB)

clean-mysql:
		@rm -f $(MYSQLO) $(MYSQLDO)

clean::         clean-mysql

distclean-mysql: clean-mysql
		@rm -f $(MYSQLDEP) $(MYSQLDS) $(MYSQLDH) $(MYSQLLIB)

distclean::     distclean-mysql

##### extra rules ######
$(MYSQLO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(MYSQLINCDIR) -o $@ -c $<
