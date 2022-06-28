# Tests for PyFilter
This code is to test if PyFilter works with all the data types available for trees.
It generates random numbers that populate the tress, whose branch name is same as the type name.
Some trees had to be generated separately.

## How to run
```bash
python testing.py
```
It will generate the Dataset automatically and test and print if the tests were sucessful.

## Results
The Filter works for 13/16 datatypes of Trees.
Exceptions are: 
1. Strings (C) : Numba does not support
2. Char_t : Not treated the same as int. Thus maybe issue with generating and filterind.
3. UChar_t : Same as above.
