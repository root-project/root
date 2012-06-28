/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
 #include <stdio.h>

 typedef struct	dut_pin_t {
   int		num;
   char  	name[5];
 } dut_pin_t;


dut_pin_t	dut_pin [] = {
  {21,  "A-1"}, {22,  "A0 "},  {23,  "A1 "},  {24,  "A2 "},
  {25,  "A3 "},  {26,  "A4 "},  {27,  "A5 "},  {28,  "A6 "},
  {29,  "A7 "},  {30,  "A8 "},  {32, "ACC"}  , {34,  "A9 "},  
  {35 , "A10"}, {36, "A11"}, {37, "A12"}, {38, "A13"}, 
  {39, "A14"}, {40, "A15"}, {41, "A16"}, {42, "A17"}, 
  {43, "A18"},  {45,  "D0 "},  {46,  "D1 "},  {47,  "D2 "},  
  {48,  "D3 "},   {51,  "D4 "},  {52,  "D5 "},  {53,  "D6 "},  
  {54,  "D7 "} ,   {55,  "D8 "},  {56,  "D9 "},  {57, "D10"},  
  {58, "D11"},  {61, "D12"}, {62, "D13"}, {63, "D14"},  
  {64, "D15"} ,  {66, "RST"}, {67,  "CLK "}, {69, "RDY"},
  {70,  "ADV "},  {74,  "CE "},  {75,  "OE "},  {76,  "WE "},
  {77,  "WP "},   {79,  "IND "},  
  {0, "   "}
};

int		pin_map [] = {
  21,  22, 0, 0	, 22,  35, 0, 0	,  23,  36, 0, 0,  24,  24, 0, 0,
  25,  37, 0, 0,  26,  25, 0, 0,  27,  38, 0, 0,  28,  26, 0, 0,
  29,  39, 0, 0,  30,  27, 0, 0,  32, 23, 0, 0	 , 34,  40, 0, 0,  
  35 , 28, 0, 0, 36, 41, 0, 0, 37, 29, 0, 0, 38, 42, 0, 0, 
  39, 30, 0, 0, 40, 43, 0, 0, 41, 31, 0, 0, 42, 44, 0, 0, 
  43, 18, 0, 0,  45,  0,  0, 0,  46,  1,  0, 0,  47,  2,  0, 0,  
  48,  3,  0, 0,   51,  4,  0, 0,  52,  5,  0, 0,  53,  6,  0, 0,  
  54,  7,  0, 0,   55,  48, 0, 0,  56,  49, 0, 0,  57, 50, 0, 0							 ,  
  58, 51, 0, 0,  61, 52, 0, 0, 62, 53, 0, 0, 63, 54, 0, 0,  
  64, 55, 0, 0,  66, 45, 0, 0, 67, 20, 0, 0, 69, 16, 0, 0,
  70, 34, 0, 0,  74,  19, 0, 0,  75,  21, 0, 0,  76,  46, 0, 0,
  77,  47, 0, 0,   79, 32, 0, 0,  
  0, 0, 0, 0	,
  
  21,  22, 0, 0	, 22,  35, 0, 0	,  23,  36, 0, 0,  24,  24, 0, 0,
  25,  37, 0, 0,  26,  25, 0, 0,  27,  38, 0, 0,  28,  26, 0, 0,
  29,  39, 0, 0,  30,  27, 0, 0,  32, 23, 0, 0	 , 34,  40, 0, 0,  
  35 , 28, 0, 0, 36, 41, 0, 0, 37, 29, 0, 0, 38, 42, 0, 0, 
  39, 30, 0, 0, 40, 43, 0, 0, 41, 31, 0, 0, 42, 44, 0, 0, 
  43, 18, 0, 0,  45,   8, 0, 0,  46,   9, 0, 0,  47,  10, 0, 0,  
  48,  11, 0, 0,   51,  12, 0, 0,  52,  13, 0, 0,  53,  14, 0, 0,  
  54,  15, 0, 0,   55,  56, 0, 0,  56,  57, 0, 0,  57, 58, 0, 0,  
  58, 59, 0, 0,  61, 60, 0, 0, 62, 61, 0, 0, 63, 62, 0, 0,  
  64, 63, 0, 0,  66, 45, 0, 0, 67, 20, 0, 0, 69, 17, 0, 0,
  70, 34, 0, 0,  74,  19, 0, 0,  75,  21, 0, 0,  76,  46, 0, 0,
  77,  47, 0, 0,   79, 33, 0, 0,  
  0, 0, 0, 0	
};


int main() {
  char line_in [200];
  char pattern_name [200];
  int j, pattern_number;
  char pattern_name_array [1024] [40];
  
  FILE *fp;
  
  fp = fopen ("98p02.hpg", "r");
  
  pattern_number = 0;
  
  
  printf("Problem #1: array index increment issue\n\n");
  
  while (fgets(line_in, 120, fp) && pattern_number < 1024) {
    if (line_in [0] != '/' && line_in [1] != '*') {	
      j=0;
      while (line_in [j + 8] != '\t') {
	
	// this form works
	
	pattern_name [j] = line_in [8 + j];
	j++;
	
	// this form doesn't work
	//pattern_name [j] = line_in [8 + j++];
      }
      
      pattern_name [j] = '\0';
      printf("Pattern %d = %s\n", pattern_number, pattern_name);
      
      sprintf (pattern_name_array [pattern_number], "%s", pattern_name);
      pattern_number++;
    }
  }

  fclose (fp);
  
  printf("\n\nProblem #2: variable initialization\n");
  
  for (j=0; j<=5; j++) {
    printf ("dut_pin [%d].num  = %d  ", j, dut_pin [j]. num);
    printf ("dut_pin [%d].name = %s\n", j, dut_pin [j]. name);
  }
  
  printf ("\nExpected values:\n");
  
  sprintf (dut_pin [0]. name, "A-1");
  sprintf (dut_pin [1]. name, "A0 ");
  sprintf (dut_pin [2]. name, "A1 ");
  sprintf (dut_pin [3]. name, "A2 ");
  sprintf (dut_pin [4]. name, "A3 ");
  sprintf (dut_pin [5]. name, "A4 ");
  
  for (j=0; j<=5; j++) {
    printf ("dut_pin [%d].num  = %d  ", j, dut_pin [j]. num);
    printf ("dut_pin [%d].name = %s\n", j, dut_pin [j]. name);
  }

  return 0;
}
						
