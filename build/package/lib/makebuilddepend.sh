#!/bin/sh 
#

if test "x$1" = "xrpm" ; then 
    shift
    for i in $* ; do 
	case $i in 
	    *-dev) 						        ;;
	    libroot)							;;
	    root-system-bin)						;;
	    root-cint)							;;
	    root-system-doc)						;;
	    *alien)	echo "BuildRequires: AliEn-Client" 		;;
# Build dependency on AfterStep-devel temporarily commented out 
# until such a time when ROOT can use the normal libAfterImage.
# Input the build dependencies of the libafterimage-dev package
#	    *asimage)	echo -n ", AfterStep-devel"			;;
	    *asimage)							;;
	    *castor)	echo "BuildRequires: castor-devel"		;;
	    *chirp)							;;
	    *clarens)							;;
	    *dcache)							;;
	    *fumili)							;;
	    *fftw3)							;;
	    *gl)							;;
	    *globus)	echo "BuildRequires: globus"			;;
	    *hbook)	echo "BuildRequires: gcc-g77"			;;
	    *krb5)	echo "BuildRequires: krb5-devel"		;;
	    *ldap)	echo "BuildRequires: openldap-devel"		;;
	    *minuit)							;;
	    *minuit2)							;;
	    *mathmore)	echo "BuildRequires: gsl-devel"			;;
	    *mlp)							;;
# This is kinda special 
	    *mysql)	
		cat <<EOF
%if %{?_vendor} 
%if %{_vendor} == "MandrakeSoft"
BuildRequires: MySQL-devel >= 4.1.0
%else
BuildRequires: mysql-devel >= 4.1.0
%endif
%else
BuildRequires: mysql-devel >= 4.1.0
%endif
EOF
		;;
	    *netx)							;;
	    *oracle)    echo "BuildRequires: oracle-instantclient-devel";;
	    *odbc)      echo "BuildRequires: unixODBC-devel >= 2.2.11"  ;;
	    *peac)							;;
	    *pgsql)	echo "BuildRequires: postgresql-devel"		;;
	    *proof)							;;
	    *pythia5)	echo "BuildRequires: pythia5-devel"		;;
	    *pythia6)	echo "BuildRequires: pythia6-devel"		;;
	    *python)	echo "BuildRequires: python-devel >= 2.1"	;;
# this is kinda special 
	    *qt)	
		cat <<EOF
%if %{?_vendor} 
%if %{_vendor} == "MandrakeSoft"
BuildRequires: libqt3-devel
%else
%if %{_vendor} == "suse"
BuildRequires: qt3-devel
%endif
%endif
%else
BuildRequires: qt-devel
%endif
EOF
;;
	    *quadp)							;;
	    *roofit)							;;
	    *ruby)	echo "BuildRequires: ruby-devel >= 1.8"		;;
	    *maxdb)	echo "BuildRequires: libsqlod75-dev"		;;
	    *sql)							;;
	    *srp)							;;
	    *tmva)							;;
	    *venus)							;;
	    *xml)	echo "BuildRequires: libxml2-devel"		;;
	    root-system-proofd)						;;
	    root-system-rootd)						;;
	    root-system-xrootd) echo "BuildRequires: krb5-devel"	;;
	    ttf-root*)							;;
	    root-system-common)						;;
	    *) 
		echo "*** Warning *** Unknown package $i - please update $0" \
		    > /dev/stderr 
		;;
	esac
    done
    exit 0
fi    

### echo %%% Making build dependencies
bd=
have_krb=0
for i in $* ; do 
    case $i in 
	*-dev)							        ;;
	libroot)							;;
	root-system-bin)						;;
	root-cint) 							;;
	root-system-doc)  						;;
	*alien)	   echo -n ", libalien-dev" 				;;
# Build dependency libafterimage-dev temporarily commented out 
# until such a time that ROOT can use the normal libAfterImage.
#	*asimage)  echo -n ", libafterimage-dev"			;;
# Input the build dependencies of the libafterimage-dev package
	*asimage)  
	    echo -n ", libjpeg62-dev, libpng12-dev, libtiff4-dev"
            echo -n ", libungif4-dev, libxinerama-dev";;
	*castor)   echo -n ", libshift-dev"				;;
	*chirp)	   echo -n ", libchirp-dev"				;;
	*clarens)  echo -n ", libxmlrpc-c3-dev | libxmlrpc-c-dev"	;;
	*dcache)   echo -n ", libdcap-dev"				;;
	*fftw3)	   echo -n ", fftw3-dev"				;;
	*fumili)   							;;
	*gl)	   
	    echo -n ", libglu1-mesa-dev | libglu1-xorg-dev "
	    echo -n "| xlibmesa-glu-dev |  libglu-dev"			;;
	*globus)   echo -n ", globus"					;;
	*hbook)	   
	    echo -n ", libpacklib1-dev [!kfreebsd-i386]"
	    echo -n ", g77|fortran-compiler" 				;;
	*krb5)		 
	    if test $have_krb -lt 1 ; then 
		echo -n ", libkrb5-dev|heimdal-dev"
		have_krb=1
	    fi 
	    echo -n ",krb5-user|heimdal-clients"		
	    ;;
	*ldap)	   echo -n ", libldap2-dev | libldap-dev"		;;
	*oracle)   echo -n ", oracle-instantclient-devel"		;;
	*mathmore) echo -n ", libgsl0-dev"				;;
	*minuit)   							;;
	*minuit2)  							;;
	*mlp)	   							;;
	*mysql)	   
	    echo -n ", libmysqlclient15-dev | libmysqlclient14-dev"
	    echo -n "| libmysqlclient12-dev| libmysqlclient-dev" 	;;
	*netx)	   							;;
	*odbc)	   echo -n ", libiodbc2-dev | unixodbc-dev"		;;
	*peac)	   							;;
	*pgsql)	   echo -n ",  libpq-dev | postgresql-dev"		;;
	*proof)	   							;;
	*pythia5)  echo -n ", pythia5-dev"				;;
	*pythia6)  echo -n ", pythia6-dev"				;;
	*python)   echo -n ", python-support (>= 0.3)"			;;
	*qt)	   
	    echo -n ", libqt3-mt-dev, libqt3-headers" 
            echo -n ", qt3-dev-tools, libqt3-compat-headers"		;;
	*quadp)	   							;;
	*roofit)   							;;
	*ruby)	   echo -n ", ruby (>= 1.8), ruby1.8-dev | ruby-dev (>= 1.8)";;
	*maxdb)	   echo -n ", libsqlod75-dev [i386 ia64 amd64]"		;;
	*sql)	   							;;
	*srp)	   echo -n ", libsrputil-dev"				;;
        *tmva)	   							;;
	*venus)	   echo -n ", libvenus-dev"				;;
	*xml)	   echo -n ", libxml2-dev"				;;
	root-system-proofd)						;;
	root-system-rootd)						;;
	root-system-xrootd)	
	    if test $have_krb -lt 1 ; then 
		echo -n ", libkrb5-dev|heimdal-dev"		
		have_krb=1
	    fi
	    ;;
	ttf-root*)							;;
	root-system-common)						;;
	*) 
	    echo "*** Warning *** Unknown package $i - please update $0" \
		> /dev/stderr ;;
    esac
done

#
# EOF
#
