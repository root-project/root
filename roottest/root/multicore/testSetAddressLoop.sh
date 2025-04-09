#! /usr/bin/env bash

COUNTER=0
while ((COUNTER < 33 )); do
      echo "Run number $COUNTER"
      ./testSetAddress;
      RETVAL=$?
         if ((RETVAL == 1)); then
            echo "ERROR: The executable did not complete successfully!";
            exit 1;
         fi
      let COUNTER=COUNTER+1
done

