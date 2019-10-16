#!/bin/bash

MODULEMAP="$ROOTSYS/include/module.modulemap"
PCMFILE=""

function usage()
{
    echo "This script is checks if the same headers are contained in multiple module files(pcms). "
    echo "Duplicates make sense only if they are textual, non-modular headers such as assert.h "
    echo "The script is basic and reports the first two modules containing the duplicate headers. "
    echo ""
    echo "$0"
    echo -e "\t-h --help"
    echo -e "\t--modulemap=$MODULEMAP"
    echo -e "\t--pcmfiles=$PCMFILE"
    echo ""
}

if [[ -z "$@" ]]; then
    echo "Needs arguments."
    usage
    exit
fi

if [[ -z "$ROOTSYS/interpreter/llvm/src/bin/clang" ]]; then
    echo "You need to set ROOTSYS to find clang."
    usage
    exit
fi

CLANG_BINARY=$ROOTSYS/interpreter/llvm/src/bin/clang

for i in "$@"
do
case $i in
    -h=*|--help=*)
    usage    
    exit
    ;;
    --modulemap=*)
    MODULEMAP="${i#*=}"
    shift # past argument=value
    ;;
    --pcmfiles=*)
    PCMFILES="${i#*=}"
    shift # past argument=value
    ;;
    *)
    echo "Unknown parameter $i"
    usage
    exit
    ;;
esac
done


declare -A hashmap
#hashmap["key"]="value"
for PCMFILE in $PCMFILES
do
  # Filter out the ROOT specific pcm files which are actually ROOT files.
  if [[ $PCMFILE == *"_rdict"* ]]; then
    echo "Ignoring ROOT pcm file $PCMFILE..."
    continue
  fi

  HEADERS_IN_PCM=$($CLANG_BINARY -fsyntax-only -Xclang -module-file-info $PCMFILE | grep 'Input file' | sed 's,Input file:,,g' | sed 's, \[.*],,g')
  for header in $HEADERS_IN_PCM
  do
    if [[ ${hashmap[$header]} ]]
    then
      echo "Header $header exist in both $PCMFILE and ${hashmap[$header]}"
      continue
    fi
    hashmap[$header]=$PCMFILE
  done
done
