# Module.mk for krb5 authentication module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 18/3/2002

MODDIR       := krb5auth
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

KRB5AUTHDIR  := $(MODDIR)
KRB5AUTHDIRS := $(KRB5AUTHDIR)/src
KRB5AUTHDIRI := $(KRB5AUTHDIR)/inc

##### libKrb5Auth #####
KRB5AUTHH    := $(wildcard $(MODDIRI)/*.h)
KRB5AUTHS    := $(wildcard $(MODDIRS)/*.cxx)
KRB5AUTHO    := $(KRB5AUTHS:.cxx=.o)

KRB5AUTHDEP  := $(KRB5AUTHO:.o=.d)

KRB5AUTHLIB  := $(LPATH)/libKrb5Auth.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(KRB5AUTHH))
ALLLIBS     += $(KRB5AUTHLIB)

# include all dependency files
INCLUDEFILES += $(KRB5AUTHDEP)

##### local rules #####
include/%.h:    $(KRB5AUTHDIRI)/%.h
		cp $< $@

$(KRB5AUTHLIB): $(KRB5AUTHO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libKrb5Auth.$(SOEXT) $@ "$(KRB5AUTHO)" \
		   "$(KRB5AUTHLIBEXTRA) $(KRB5LIBDIR) $(KRB5LIB)"

all-krb5auth:   $(KRB5AUTHLIB)

clean-krb5auth:
		@rm -f $(KRB5AUTHO)

clean::         clean-krb5auth

distclean-krb5auth: clean-krb5auth
		@rm -f $(KRB5AUTHDEP) $(KRB5AUTHLIB)

distclean::     distclean-krb5auth

##### extra rules ######
$(KRB5AUTHO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) -I$(KRB5INCDIR) -o $@ -c $<
