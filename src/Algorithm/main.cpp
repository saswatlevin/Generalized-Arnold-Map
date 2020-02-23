  #include <iostream> /*For IO*/
  #include <functional>
  #include <cstdint>  /*For standardized data types*/
  #include <cstdio>   /*For printf*/
  #include <random>   /*For random number generation*/
  #include <vector>   /*For dynamic arrays*/
  #include <numeric>  /*For std::iota*/
  #include <cmath>    /*For fmod to do x%1 , y%1 */
  #include <array>    /*For static arrays*/
  #include <string>   /*For strings*/
  #include <fstream>  /*For file handling*/
  #include <cstdlib>  /*For abs() */
  #include <ctime>    /*To seed srand()*/ 
  #include "CONFIG.hpp"
  #include "functions.hpp"
  
  using namespace std;
  
     
  
  int main()
  {  
    uint8_t alpha1=0,beta1=0,a1=0,b1=0,offset1=0;
    uint8_t alpha2=0,beta2=0,a2=0,b2=0,offset2=0;
    
    uint16_t c1=0;
    uint16_t c2=0;
    
    int32_t r1=0;
    int32_t r2=0;
    
    float x1=0,y1=0,skipped_x1=0,skipped_y1=0; 
    float x2=0,y2=0,skipped_x2=0,skipped_y2=0;
    
    bool isNotDecimal=0;
    
    constants constants_array1;
    constants constants_array2;
    std::array<float,NUM_SKIPOFFSET_ARGS> skipped_x_y1;
    std::array<float,NUM_SKIPOFFSET_ARGS> skipped_x_y2;
    std::array<float,TOTAL> P1;
    std::array<float,TOTAL> P2;
    std::array<uint32_t,M> U;
    std::array<uint32_t,M> V;
    //std::string filename=std::string("");
    
    /*First set of Constants*/
    constants_array1=initializeConstants(SEED1,4);
    alpha1  = constants_array1.alpha;
    beta1   = constants_array1.beta;
    a1      = constants_array1.a;
    b1      = constants_array1.b;
    c1      = constants_array1.c;
    x1      = constants_array1.x;
    y1      = constants_array1.y;
    r1      = constants_array1.r;
    offset1 = constants_array1.offset;
    skipped_x_y1=skipOffset(a1,b1,c1,x1,y1,UNZERO,offset1);
    skipped_x1=skipped_x_y1.at(0);
    skipped_y1=skipped_x_y1.at(1);
    writeToFile("constants1.txt",alpha1,beta1,a1,b1,c1,x1,y1,r1,offset1,WRITE_MULTIPLE_ARGS);
    P1=generateP(a1,b1,c1,skipped_x1,skipped_y1,UNZERO);
    U=generateU(r1,P1);

    printf("\nP1=");
    for(uint8_t i=0;i<TOTAL;++i)
    {
      cout<<P1[i]<<" ";
    }
    
    printf("\nU=");
    for (uint8_t i=0;i<M;++i)
    {
      cout<<U[i]<<" ";
    } 
      printf("\na1=%d\tb1=%d\tc1=%d\tx1=%f\ty1=%f\tr1=%d\toffset1=%d\tskipped_x1=%f\tskipped_y1=%f\t",a1,b1,c1,x1,y1,r1,offset1,skipped_x1,skipped_y1); 

    /*Second set of Constants*/
    constants_array2=initializeConstants(SEED2,6);
    alpha2  = constants_array2.alpha;
    beta2   = constants_array2.beta;
    a2      = constants_array2.a;
    b2      = constants_array2.b;
    c2      = constants_array2.c;
    x2      = constants_array2.x;
    y2      = constants_array2.y;
    r2      = constants_array2.r;
    offset2 = constants_array2.offset;
    skipped_x_y2=skipOffset(a2,b2,c2,x2,y2,UNZERO,offset2);
    skipped_x2=skipped_x_y2.at(0);
    skipped_y2=skipped_x_y2.at(1);
    writeToFile("constants2.txt",alpha2,beta2,a2,b2,c2,x2,y2,r2,offset2,WRITE_MULTIPLE_ARGS);
    P2=generateP(a2,b2,c2,skipped_x2,skipped_y2,UNZERO);
    V=generateU(r2,P2);
    
    printf("\nP2=");
    for(uint8_t i=0;i<TOTAL;++i)
    {
      cout<<P2[i]<<" ";
    }
    
    printf("\nV=");
    for (uint8_t i=0;i<M;++i)
    {
      cout<<V[i]<<" ";
    } 
         
    printf("\na2=%d\tb2=%d\tc2=%d\tx2=%f\ty2=%f\tr2=%d\toffset2=%d\tskipped_x2=%f\tskipped_y2=%f\t",a2,b2,c2,x2,y2,r2,offset2,skipped_x2,skipped_y2);

    //isNotDecimal=checkDecimal(P1);
    //cout<<"\nIsNotDecimal="<<isNotDecimal;
    return 0; 
  }
  

