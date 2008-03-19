############################################################
# Automatically created makefile for p2f
############################################################

# Copying $CINTSYSDIR/MAKEINFO #############################

############################################################
# platform/linux2.0RH6.2
#  Platform dependent information for LINUX 2.0 RedHatH6.2 or later
#  Redhat-5.2
############################################################

# Tools
RM	    = rm -f
CP	    = cp
AR	    = ar
AROPT       = clq
KRCC        = gcc -traditional
CC          = gcc -Wall
CPP         = g++ -Wall -fguiding-decls
LD          = g++
CPREP       = gcc -E -C
CPPPREP     = g++ -E -C

# Compiler and linker option
CCDLLOPT    = -fPIC
LDDLLOPT    = -shared
OPTIMIZE    = -O2
LDOPT       = -lm -L/usr/lib/termcap -ltermcap -lbsd -ldl -rdynamic
SYSMACRO    = -DG__REGEXP -DG__SHAREDLIB -DG__OSFDLL -DG__ANSI
OTHMACRO    = -DG__P2FCAST -DG__REDIRECTIO -DG__DETECT_NEWDEL -DG__POSIX -DG__STD_EXCEPTION 
SYSIPATH    =

# libraries
MAINO	    = $(CINTSYSDIR)/main/G__cppmain.o
CINTLIB     = $(CINTSYSDIR)/src/G__ci.a
READLINEA   = /usr/lib/libreadline.a
APIO	    = Api.o Class.o BaseCls.o Type.o DataMbr.o Method.o MethodAr.o CallFunc.o Typedf.o Apiif.o Token.o
RANLIB	    = /usr/bin/ranlib
STDLIBS     = libstrm.o stdstrct.o
PLATFORM    =

# source code postfix
CSRCPOST    = .c
CHDRPOST    = .h
CPPSRCPOST  = .C
CPPHDRPOST  = .h
OBJPOST     = .o
DLLPOST     = .dl




# End of $CINTSYSDIR/MAKEINFO ##############################

# Set variables ############################################
IPATH      = $(SYSIPATH) \
		-I/home/gotom/src/cint \
		-I/home/gotom/src/cint/src 
MACRO      = $(SYSMACRO)
CINTSYSDIR = /home/gotom/src/cint
CINTIPATH  = -I$(CINTSYSDIR)
OBJECT     = p2f
OPTION     =
DLLSPEC    =
LINKSPEC   = -DG__CPPLINK_ON

# Set File names ###########################################
CIFC       =
CIFH       =
CIFO       =
CPPIFC     = G__cpp_p2f.C
CPPIFH     = G__cpp_p2f.h
CPPIFO     = G__cpp_p2f.o

LIBS       = 

CCOPT      = 

CINTOPT      = 

COFILES    = 

RMCOFILES  = 

CHEADER    = 
CHEADERCINT = 

CSTUB      = 
CSTUBCINT  = 

CPPOFILES  = \
		p2f.o 

RMCPPOFILES = \
		p2f.o 

CPPHEADER  = \
		p2f.h 
CPPHEADERCINT  = \
		p2f.h 

CPPSTUB    = 
CPPSTUBCINT = 

# Link Object #############################################
$(OBJECT) : $(MAINO) $(CINTLIB) $(READLINEA) G__setup.o $(COFILES) $(CPPOFILES) $(CIFO) $(CPPIFO) 
	$(LD) $(OPTIMIZE) $(IPATH) $(MACRO) $(CCOPT) -o $(OBJECT) $(MAINO) $(CIFO) $(CPPIFO) $(COFILES) $(CPPOFILES) $(CINTLIB) G__setup.o $(READLINEA) $(LIBS) $(LDOPT)

# Compile User C source files ##############################

# Compile User C++ source files ############################
p2f.o : p2f.C $(CPPHEADER)
	$(CPP) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -o p2f.o -c p2f.C


# Compile dictionary setup routine #######################
G__setup.o : $(CINTSYSDIR)/main/G__setup.c $(CINTSYSDIR)/G__ci.h
	$(CC) $(LINKSPEC) $(CINTIPATH) $(OPTIMIZE) $(OPTION) -o G__setup.o -c $(CINTSYSDIR)/main/G__setup.c

# Compile C++ Interface routine ##########################
$(CPPIFO) : $(CPPIFC)
	$(CPP) $(CINTIPATH) $(IPATH) $(MACRO) $(OPTIMIZE) $(OPTION) $(CCOPT) -c $(CPPIFC)

# Create C++ Interface routine ###########################
$(CPPIFC) : $(CPPHEADER) $(CPPSTUB) $(CINTSYSDIR)/cint
	cint  -w0 -zp2f -n$(CPPIFC) $(DLLSPEC) -D__MAKECINT__ -DG__MAKECINT  -c-1 -A $(IPATH) $(MACRO) $(CINTOPT) $(CPPHEADERCINT)


# Clean up #################################################
clean :
	$(RM) $(OBJECT) core $(CIFO) $(CIFC) $(CIFH) $(CPPIFO) $(CPPIFC) $(CPPIFH) $(RMCOFILES) $(RMCPPOFILES) G__setup.o

# re-makecint ##############################################
makecint :
	makecint -mk Makefile -o p2f -I/home/gotom/src/cint -I/home/gotom/src/cint/src -H p2f.h -C++ p2f.C 

