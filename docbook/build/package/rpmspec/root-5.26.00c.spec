%{expand: %%define pyver %(python -c 'import sys;print(sys.version[0:3])')}

%define base_version 5.26.00
%define patch_release c

Name:         root
Version:      %{base_version}%{patch_release}
Release:      1%{dist}
License:      ROOT Software Terms and Conditions
Packager:     Mark Dalton <dalton@jlab.org>, Sergio Ballestrero <sergio.ballestrero@cern.ch>
Vendor:       ROOT Team
URL:          http://root.cern.ch/
Prefix:       /opt/root
Source0:       ftp://root.cern.ch/root/root_v%{base_version}%{patch_release}.source.tar.gz
#Source1:      root.config.Makefile.linux-OPT.P4
#Patch0:       root_v5.24.00d.patch
Group:        Applications/Science
BuildRoot:    %{_tmppath}/%{name}-%{version}-root
BuildRequires: gcc-c++, libstdc++-devel
BuildRequires: libX11-devel, libXpm-devel
#BuildRequires: xorg-x11-devel, xpm-devel
BuildRequires: libGL-devel, libGLU-devel
BuildRequires: python-devel
BuildRequires: qt-devel
BuildRequires: zlib-devel, libpng-devel, libjpeg-devel, libtiff-devel
BuildRequires: cernlib-devel
BuildRequires: gsl-devel
BuildRequires: pcre-devel
BuildRequires: freetype-devel
BuildRequires: openssl-devel
BuildRequires: fftw-devel
BuildRequires: ncurses-devel
BuildRequires: graphviz-devel
Summary:      Numerical data analysis framework (OO)
Requires(pre,post): /sbin/ldconfig

Provides:root-devel

BuildArchitectures: %ix86 x86_64
ExclusiveArch: %ix86 x86_64

%define cernlib_version 2006
%ifarch %ix86
%define cernlib_path /usr/lib/cernlib
%define cernlib_arch_config linux
%else
%define cernlib_path /usr/lib64/cernlib
%define cernlib_arch_config linuxx8664gcc
%endif

%description
From the web page (http://root.cern.ch/):

The ROOT system provides a set of OO frameworks with all the
functionality needed to handle and analyse large amounts of data in a
very efficient way. Having the data defined as a set of objects,
specialised storage methods are used to get direct access to the
separate attributes of the selected objects, without having to touch
the bulk of the data. Included are histograming methods in 1, 2 and 3
dimensions, curve fitting, function evaluation, minimisation, graphics
and visualisation classes to allow the easy setup of an analysis
system that can query and process the data interactively or in batch
mode.

Thanks to the built in CINT C++ interpreter the command language, the
scripting, or macro, language and the programming language are all
C++. The interpreter allows for fast prototyping of the macros since
it removes the time consuming compile/link cycle. It also provides a
good environment to learn C++. If more performance is needed the
interactively developed macros can be compiled using a C++ compiler.

The system has been designed in such a way that it can query its
databases in parallel on MPP machines or on clusters of workstations
or high-end PC's. ROOT is an open system that can be dynamically
extended by linking external libraries. This makes ROOT a premier
platform on which to build data acquisition, simulation and data
analysis systems.

More information can be obtained via the ROOT web pages:
    http://root.cern.ch

%prep
%setup -q -n root
#kefile.linuxcp %{SOURCE1} config/Makefile.linux
#%patch0 -p1

%build
#export ROOTSYS=%{_prefix}
export QTDIR=/usr/lib64/qt4
./configure %{cernlib_arch_config} \
        	--prefix=%{_prefix} \
        	--libdir=%{_libdir} \
		--docdir=%{_defaultdocdir}/%{name}-%{version} \
		--fontdir=%{_datadir}/fonts/%{name} \
		--with-cern-libdir=%{cernlib_path}/%{cernlib_version}/lib \
		--enable-cern \
		--enable-cintex \
		--enable-editline \
		--enable-qt \
		--enable-qtgsi \
		--enable-opengl \
		--enable-explicitlink \
		--enable-python \
		--enable-mathcore \
		--enable-genvector \
		--enable-mathmore \
		--enable-g4root \
		--enable-gsl-shared \
		--enable-minuit2 \
		--enable-reflex \
		--enable-roofit \
		--enable-shadowpw \
		--enable-shared \
		--enable-soversion \
		--enable-table \
		--enable-xml \
		--enable-xrootd \
		--enable-xft \
		--enable-fftw3 \
		--enable-gdml \
		--enable-tmva \
		--enable-ssl \
		--enable-memstat \
		--enable-gviz \
		--disable-afs \
		--disable-krb5 \
		--disable-ldap \
     		--disable-globus \
		--disable-mysql \
		--disable-pgsql \
		--disable-oracle \
		--disable-pythia6 \
		--disable-rfio \
		--disable-srp \
		--disable-builtin-zlib \
		--disable-builtin-freetype \
		--disable-builtin-pcre

%{__make} %{_smp_mflags}
%{__make} %{_smp_mflags} cintdlls

%install
rm -rf $RPM_BUILD_ROOT
mkdir -p $RPM_BUILD_ROOT/%{_prefix}

make install DESTDIR=${RPM_BUILD_ROOT} USECONFIG=TRUE
find ${RPM_BUILD_ROOT} -name "CVS" | xargs rm -fr 

%clean
rm -rf $RPM_BUILD_ROOT

%post -p /sbin/ldconfig
%postun -p /sbin/ldconfig

%files
%defattr(-,root,root)
%docdir %{_defaultdocdir}
#%%{_libdir}/python%{pyver}/site-packages/*
%{_prefix}
%{_sysconfdir}

%changelog
* Wed Jul 21 2010 Olivier Lahaye <olivier.lahaye1@free.fr> - 5.26.00c-1%{dist}
- New upstream release.
* Fri Apr 23 2010 Olivier Lahaye <olivier.lahaye1@free.fr> - 5.26.00b-2%{dist}
- Added build options: fftw3, gdml, gviz, memstat, g4root, genvector, editline,
  gsl-shared, ssl, tmva
* Wed Apr 07 2010 Olivier Lahaye <olivier.lahaye1@free.fr> - 5.26.00b-1%{dist}
- new upstream version
* Thu Aug 20 2009 Olivier Lahaye <olivier.lahaye1@free.fr> - 5.24.00d-1%{dist}
- bugfix release
* Fri Aug 7 2009 Olivier Lahaye <olivier.lahaye1@free.fr> - 5.24.00-1%{dist}
- new upstream version
* Tue Jul 7 2009 Olivier Lahaye <olivier.lahaye@cea.fr> - 5.16.00-1.el4
- rebuild using qt4
* Tue Jul 7 2009 Olivier Lahaye <olivier.lahaye@cea.fr> - 5.16.00-0.el4
- support for x86_64
* Tue Sep 18 2007 Sergio Ballestrero <sergio.ballestrero@cern.ch> - 5.16.00-0.SL4.WITS
- upstream release 5.16.00
* Fri Mar 23 2007 Sergio Ballestrero <sergio.ballestrero@cern.ch> - 5.14.00d-0.SL4.WITS
- upstream release 5.14.00d
* Tue Jul 25 2006 Sergio Ballestrero <sergio.ballestrero@cern.ch> - 5.12.00-0.SL4.WITS
- Updates for 5.12.00
- copy some config from official root binaries
* Fri May 26 2006 Sergio Ballestrero <sergio.ballestrero@cern.ch> - 5.10.00d-0.SL4.WITS
- Updates for 5.10.00d
* Thu Feb 16 2006 Mark Dalton <dalton@src.wits.ac.za>
- added --enable-minuit2
* Fri Dec 13 2005 Sergio Ballestrero <sergio.ballestrero@cern.ch>
- path, ld.so.conf.d
* Fri Dec 09 2005 Sergio Ballestrero <sergio.ballestrero@cern.ch>
- profile scripts, _wits sub-version, BuildRequires
* Thu Nov 10 2005 Sergio Ballestrero <sergio.ballestrero@cern.ch>
- h2root, OpenGL
* Mon Jun 13 2005 Katarina Pajcel <katarina.pajchel@fys.uio.no>
- Added cintdlls
* Sun May 29 2005 Mattias Ellert <mattias.ellert@tsl.uu.se>
- Updates for 4.04.04
* Mon Feb 23 2004 Jakob Langgaard Nielsen <langgard@nbi.dk>
- Updates for 3.10.02
* Mon Nov 24 2003 Jakob Langgaard Nielsen <langgard@nbi.dk>
- Updates for 3.05.07
* Fri Aug  8 2003 Jakob Langgaard Nielsen <langgard@nbi.dk>
- Updates for 3.05.06
