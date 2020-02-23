  #ifndef CONFIG_H /*To ensure no errors if the header file is included twice*/
  #define CONFIG_H


  //EXP=10^14

  #define RAND_UPPER 32
  #define EXP 100000000000000   
  #define UNZERO 0.0000001
  #define NUM_SKIPOFFSET_ARGS 2
  #define WRITE_SINGLE_ARG 0
  #define WRITE_MULTIPLE_ARGS 1
  #define M 5
  #define N 5
  #define TOTAL (M*N) 
  #define R_UPPER (M*N)-N
  #define MID ((M*N)/2)
  #define SEED1 30
  #define SEED2 32

  #endif
