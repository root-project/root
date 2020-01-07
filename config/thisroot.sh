# Source this script to set up the ROOT build that this script is part of.
#
# Conveniently an alias like this can be defined in .bashrc:
#   alias thisroot=". bin/thisroot.sh"
#
# This script if for the bash like shells, see thisroot.csh for csh like shells.
#
# Author: Fons Rademakers, 18/8/2006

drop_from_path()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "drop_from_path: needs 2 arguments"
      return 1
   fi

   local p=$1
   local drop=$2

   newpath=`echo $p | sed -e "s;:${drop}:;:;g" \
                          -e "s;:${drop}\$;;g"   \
                          -e "s;^${drop}:;;g"   \
                          -e "s;^${drop}\$;;g"`
}

clean_environment()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "clean_environment: needs 2 arguments"
      return 1
   fi

   local exp_pyroot=$1
   local pyroot_dir=$2

   # Check if we are using ZSH
   if [ ! -z $ZSH_VERSION ] ; then
      # Check if nonomatch option is set, enable if not and save
      # the initial status
      if ! setopt | grep -q nonomatch; then
         setopt +o nomatch
         unset_nomatch=true
      fi
   fi

   if [ -n "${old_rootsys}" ] ; then
      if [ -n "${PATH}" ]; then
         drop_from_path "$PATH" "${old_rootsys}/bin"
         PATH=$newpath
      fi
      if [ -n "${LD_LIBRARY_PATH}" ]; then
         drop_from_path "$LD_LIBRARY_PATH" "${old_rootsys}/lib"
         LD_LIBRARY_PATH=$newpath
         if [ ! -z "${exp_pyroot}" ] ; then
            drop_from_path "$LD_LIBRARY_PATH" "$pyroot_dir"
            LD_LIBRARY_PATH=$newpath
            for pyroot_libs_dir in ${old_rootsys}/lib/python*
            do
               drop_from_path "$LD_LIBRARY_PATH" "$pyroot_libs_dir"
               LD_LIBRARY_PATH=$newpath
            done
         fi
      fi
      if [ -n "${DYLD_LIBRARY_PATH}" ]; then
         drop_from_path "$DYLD_LIBRARY_PATH" "${old_rootsys}/lib"
         DYLD_LIBRARY_PATH=$newpath
         if [ ! -z "${exp_pyroot}" ] ; then
            drop_from_path "$DYLD_LIBRARY_PATH" "$pyroot_dir"
            DYLD_LIBRARY_PATH=$newpath
            for pyroot_libs_dir in ${old_rootsys}/lib/python*
            do
               drop_from_path "$DYLD_LIBRARY_PATH" "$pyroot_libs_dir"
               DYLD_LIBRARY_PATH=$newpath
            done
         fi
      fi
      if [ -n "${SHLIB_PATH}" ]; then
         drop_from_path "$SHLIB_PATH" "${old_rootsys}/lib"
         SHLIB_PATH=$newpath
         if [ ! -z "${exp_pyroot}" ] ; then
            drop_from_path "$SHLIB_PATH" "$pyroot_dir"
            SHLIB_PATH=$newpath
            for pyroot_libs_dir in ${old_rootsys}/lib/python*
            do
               drop_from_path "$SHLIB_PATH" "$pyroot_libs_dir"
               SHLIB_PATH=$newpath
            done
         fi
      fi
      if [ -n "${LIBPATH}" ]; then
         drop_from_path "$LIBPATH" "${old_rootsys}/lib"
         LIBPATH=$newpath
         if [ ! -z "${exp_pyroot}" ] ; then
            drop_from_path "$LIBPATH" "$pyroot_dir"
            LIBPATH=$newpath
            for pyroot_libs_dir in ${old_rootsys}/lib/python*
            do
               drop_from_path "$LIBPATH" "$pyroot_libs_dir"
               LIBPATH=$newpath
            done
         fi
      fi
      if [ -n "${PYTHONPATH}" ]; then
         drop_from_path "$PYTHONPATH" "${old_rootsys}/lib"
         PYTHONPATH=$newpath
         if [ ! -z "${exp_pyroot}" ] ; then
            drop_from_path "$PYTHONPATH" "$pyroot_dir"
            PYTHONPATH=$newpath
            for pyroot_libs_dir in ${old_rootsys}/lib/python*
            do
               drop_from_path "$PYTHONPATH" "$pyroot_libs_dir"
               PYTHONPATH=$newpath
            done
         fi
      fi
      if [ -n "${MANPATH}" ]; then
         drop_from_path "$MANPATH" "${old_rootsys}/man"
         MANPATH=$newpath
      fi
      if [ -n "${CMAKE_PREFIX_PATH}" ]; then
         drop_from_path "$CMAKE_PREFIX_PATH" "${old_rootsys}"
         CMAKE_PREFIX_PATH=$newpath
      fi
      if [ -n "${JUPYTER_PATH}" ]; then
         drop_from_path "$JUPYTER_PATH" "${old_rootsys}/etc/notebook"
         JUPYTER_PATH=$newpath
      fi
   fi
   if [ -z "${MANPATH}" ]; then
      # Grab the default man path before setting the path to avoid duplicates
      if command -v manpath >/dev/null; then
         default_manpath=`manpath`
      elif command -v man >/dev/null; then
         default_manpath=`man -w 2> /dev/null`
      else
         default_manpath=""
      fi
   fi

   # Check value of $unset_nomatch and unset if needed
   if [ ! -z "${unset_nomatch}" ]; then
      setopt -o nomatch
   fi

}

set_environment()
{
   # Assert that we got enough arguments
   if test $# -ne 2 ; then
      echo "set_environment: needs 2 arguments"
      return 1
   fi

   local exp_pyroot=$1
   local pyroot_dir=$2

   if [ -z "${PATH}" ]; then
      PATH=@bindir@; export PATH
   else
      PATH=@bindir@:$PATH; export PATH
   fi

   if [ -z "${exp_pyroot}" ]; then
      if [ -z "${LD_LIBRARY_PATH}" ]; then
         LD_LIBRARY_PATH=@libdir@
         export LD_LIBRARY_PATH       # Linux, ELF HP-UX
      else
         LD_LIBRARY_PATH=@libdir@:$LD_LIBRARY_PATH
         export LD_LIBRARY_PATH
      fi
      if [ -z "${DYLD_LIBRARY_PATH}" ]; then
         DYLD_LIBRARY_PATH=@libdir@
         export DYLD_LIBRARY_PATH       # Linux, ELF HP-UX
      else
         DYLD_LIBRARY_PATH=@libdir@:$DYLD_LIBRARY_PATH
         export DYLD_LIBRARY_PATH
      fi
      if [ -z "${SHLIB_PATH}" ]; then
         SHLIB_PATH=@libdir@
         export SHLIB_PATH       # Linux, ELF HP-UX
      else
         SHLIB_PATH=@libdir@:$SHLIB_PATH
         export SHLIB_PATH
      fi
      if [ -z "${LIBPATH}" ]; then
         LIBPATH=@libdir@
         export LIBPATH       # Linux, ELF HP-UX
      else
         LIBPATH=@libdir@:$LIBPATH
         export LIBPATH
      fi
      if [ -z "${PYTHONPATH}" ]; then
         PYTHONPATH=@libdir@
         export PYTHONPATH       # Linux, ELF HP-UX
      else
         PYTHONPATH=@libdir@:$PYTHONPATH
         export PYTHONPATH
      fi
   else
      if [ -z "${LD_LIBRARY_PATH}" ]; then
         LD_LIBRARY_PATH=@libdir@:$pyroot_dir
         export LD_LIBRARY_PATH       # Linux, ELF HP-UX
      else
         LD_LIBRARY_PATH=@libdir@:$pyroot_dir:$LD_LIBRARY_PATH
         export LD_LIBRARY_PATH
      fi
      if [ -z "${DYLD_LIBRARY_PATH}" ]; then
         DYLD_LIBRARY_PATH=@libdir@:$pyroot_dir
         export DYLD_LIBRARY_PATH       # Linux, ELF HP-UX
      else
         DYLD_LIBRARY_PATH=@libdir@:$pyroot_dir:$DYLD_LIBRARY_PATH
         export DYLD_LIBRARY_PATH
      fi
      if [ -z "${SHLIB_PATH}" ]; then
         SHLIB_PATH=@libdir@:$pyroot_dir
         export SHLIB_PATH       # Linux, ELF HP-UX
      else
         SHLIB_PATH=@libdir@:$pyroot_dir:$SHLIB_PATH
         export SHLIB_PATH
      fi
      if [ -z "${LIBPATH}" ]; then
         LIBPATH=@libdir@:$pyroot_dir
         export LIBPATH       # Linux, ELF HP-UX
      else
         LIBPATH=@libdir@:$pyroot_dir:$LIBPATH
         export LIBPATH
      fi
      if [ -z "${PYTHONPATH}" ]; then
         PYTHONPATH=$pyroot_dir
         export PYTHONPATH       # Linux, ELF HP-UX
      else
         PYTHONPATH=$pyroot_dir:$PYTHONPATH
      fi
   fi

   if [ -z "${MANPATH}" ]; then
      MANPATH=@mandir@:${default_manpath}; export MANPATH
   else
      MANPATH=@mandir@:$MANPATH; export MANPATH
   fi

   if [ -z "${CMAKE_PREFIX_PATH}" ]; then
      CMAKE_PREFIX_PATH=$ROOTSYS; export CMAKE_PREFIX_PATH       # Linux, ELF HP-UX
   else
      CMAKE_PREFIX_PATH=$ROOTSYS:$CMAKE_PREFIX_PATH; export CMAKE_PREFIX_PATH
   fi

   if [ -z "${JUPYTER_PATH}" ]; then
      JUPYTER_PATH=$ROOTSYS/etc/notebook; export JUPYTER_PATH       # Linux, ELF HP-UX
   else
      JUPYTER_PATH=$ROOTSYS/etc/notebook:$JUPYTER_PATH; export JUPYTER_PATH
   fi
}


### main ###


if [ -n "${ROOTSYS}" ] ; then
   old_rootsys=${ROOTSYS}
fi


SOURCE=${BASH_ARGV[0]}
if [ "x$SOURCE" = "x" ]; then
   SOURCE=${(%):-%N} # for zsh
fi


if [ "x${SOURCE}" = "x" ]; then
   if [ -f bin/thisroot.sh ]; then
      ROOTSYS="$PWD"; export ROOTSYS
   elif [ -f ./thisroot.sh ]; then
      ROOTSYS=$(cd ..  > /dev/null; pwd); export ROOTSYS
   else
      echo ERROR: must "cd where/root/is" before calling ". bin/thisroot.sh" for this version of bash!
      ROOTSYS=; export ROOTSYS
      return 1
   fi
else
   # get param to "."
   thisroot=$(dirname ${SOURCE})
   ROOTSYS=$(cd ${thisroot}/.. > /dev/null;pwd); export ROOTSYS
fi


if [ -z "${ROOT_PYTHON_VERSION}" ] ; then
   py_localruntimedir=@py_localruntimedir@
   py_version=${py_localruntimedir#@localruntimedir@/python}
   if [ -d "$ROOTSYS/lib/python${py_version}" ]; then
      # Experimental PyROOT
      ROOT_PYTHON_VERSION=${py_version}
   fi
else
   # check if version exists and exit if not
   if [ ! -d "@libdir@/python${ROOT_PYTHON_VERSION}" ]; then
      echo ERROR: build with Python version "${ROOT_PYTHON_VERSION}" not found.
      echo Available versions:
      for version_path in @libdir@/python*
      do
         version_number=${version_path#@libdir@/python}
         echo ${version_number}
      done
      return 1
   fi
fi


# Check if the directory created in PyROOT experimental exists
if [ -d "@libdir@/python${ROOT_PYTHON_VERSION}" ]; then
   exp_pyroot=true
fi


# Check if we are in build or installation directory
if [ ! -d "CMakeFiles" ]; then
   pyroot_dir=@CMAKE_INSTALL_FULL_PYROOTDIR@
else
   pyroot_dir=@libdir@/python${ROOT_PYTHON_VERSION}
fi


clean_environment "${exp_pyroot}" "${pyroot_dir}"
set_environment "${exp_pyroot}" "${pyroot_dir}"


# Prevent Cppyy from checking the PCH (and avoid warning)
export CLING_STANDARD_PCH=none

if [ "x`root-config --arch | grep -v win32gcc | grep -i win32`" != "x" ]; then
   ROOTSYS="`cygpath -w $ROOTSYS`"
fi


unset old_rootsys
unset thisroot
unset -f drop_from_path
unset -f clean_environment
unset -f set_environment
unset ROOT_PYTHON_VERSION
unset pyroot_dir
