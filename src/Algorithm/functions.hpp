  
  #ifndef FUNCTIONS_H /*To ensure no errors if the header file is included twice*/
  #define FUNCTIONS_H
  
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
  #include <bitset>   /*For to_string()*/
  #include <opencv2/opencv.hpp> /*For OpenCV*/
  #include <opencv2/highgui/highgui.hpp>
  
  #define RAND_UPPER 32
  #define EXP 100000000000000   
  #define UNZERO 0.0000001
  #define NUM_SKIPOFFSET_ARGS 2
  #define WRITE_SINGLE_ARG 0
  #define WRITE_MULTIPLE_ARGS 1
  #define SEED1 30
  #define SEED2 32
  
  using namespace std;
  using namespace cv;   
  
     struct constants
     {
      uint8_t alpha;
      uint8_t beta;
      uint8_t a;
      uint8_t b;
      uint16_t c;
      float x;
      float y;
      int32_t r;
      uint8_t offset;
     };
    
  /*Function Prototypes*/
  
  constants initializeConstants(uint32_t,uint32_t,uint32_t); 
  void writeToFile(std::string,uint8_t,uint8_t,uint8_t,uint8_t,uint16_t,float,float,int32_t,uint8_t,bool);   
  std::vector<float> skipOffset(uint8_t,uint8_t,uint16_t,float,float,float,uint8_t);
  std::vector<float> generateP(std::vector<float>,uint8_t,uint8_t,uint16_t,float,float,float,uint32_t,uint32_t);
  std::vector<uint32_t> generateU(std::vector<uint32_t>,std::vector<float>,int32_t,uint32_t,uint32_t N);   
  std::vector<uint8_t> flattenGrayScaleImage(Mat image,uint32_t,uint32_t);
  bool checkDecimal(std::vector<float>,uint32_t);
    
  
  bool checkDecimal(std::vector<float> arr,uint32_t TOTAL)
  {
    printf("\nIn checkDecimal\n");
    for(uint32_t i=0;i<TOTAL;++i)
    {
      if(arr[i]>=1.0)
      {
        return 1;
      }      
    }
    return 0;
  }
    
  std::vector<float> skipOffset(uint8_t a,uint8_t b,uint16_t c,float x,float y,float unzero,uint8_t offset)
  {
    printf("\nIn skipOffset\n");
    std::vector<float> result(NUM_SKIPOFFSET_ARGS);
    std::vector<uint8_t> loop_range(offset);
    // Fill loop_range with 0, 1, ..., offset-1
    std::iota (std::begin(loop_range), std::end(loop_range), 0);
   printf("\nBefore Offset loop"); 
   for (uint8_t i : loop_range)
   {    
	x = fmod((x + a*y),1.0) + unzero; 
   
        y = fmod((b*x + c*y),1.0) + unzero;
        
        //printf("\ni=%d\t",i);
   }
   
   //printf("\noffset=%d\t",offset);
   //printf("\nx=%f",x);
   //printf("\ny=%f",y);
   
   result[0]=x;
   result[1]=y;
   return result; 
  } 

  constants initializeConstants(uint32_t seed,uint32_t offset,uint32_t R_UPPER)
  {
    printf("\nIn InitializeCOnstants\n");
    struct constants constants_array;
    
    //uint32_t seed;

    /*Seeding the rand() generator*/
    srand(seed+offset);

    /*Assigning random values to constants*/
    //constants_array.push_back(constants());
    constants_array.alpha =  RAND_UPPER + rand() / (RAND_MAX / ( 1 - RAND_UPPER + 1) + 1); 
    constants_array.beta  =  RAND_UPPER + rand() / (RAND_MAX / (1 - RAND_UPPER + 1) + 1);
    constants_array.a =      RAND_UPPER + rand() / (RAND_MAX / (2 - RAND_UPPER + 1) + 1);
    constants_array.b =      RAND_UPPER + rand() / (RAND_MAX / (2 - RAND_UPPER + 1) + 1);
    constants_array.c =      1 + (constants_array.a * constants_array.b);
    constants_array.x =      0 + rand() / ( RAND_MAX / (0.0001 - 1.0 + 1.0) + 1.0);
    constants_array.y =      0 + rand() / ( RAND_MAX / (0.0001 - 1.0 + 1.0) + 1.0);
    constants_array.r =      1+(rand()%RAND_UPPER);
    constants_array.offset = RAND_UPPER + rand() / (RAND_MAX / ( 1 - RAND_UPPER + 1) + 1);

    
    return constants_array;
  }
  
  void writeToFile(std::string filename,uint8_t alpha,uint8_t beta,uint8_t a,uint8_t b,uint16_t c,float x,float y,int32_t r,uint8_t offset,bool mode)
  { 
    printf("\nIn WriteToFile\n");
    std::string stringToWrite=std::string("");
    std::string constant=std::string("");
    
    if (mode==1)
    {
     constant=std::to_string(alpha);
     stringToWrite+=constant;
     stringToWrite+="\n";
     return;
    }
    
    
    constant=std::to_string(alpha);
    stringToWrite+=constant;
    stringToWrite+="\n";
    
    constant=std::to_string(beta);
    stringToWrite+=constant;
    stringToWrite+="\n";
    
    constant=std::to_string(a);
    stringToWrite+=constant;
    stringToWrite+="\n";

    constant=std::to_string(b);
    stringToWrite+=constant;
    stringToWrite+="\n";
    
    constant=std::to_string(c);
    stringToWrite+=constant;
    stringToWrite+="\n";
    
    constant=std::to_string(x);
    stringToWrite+=constant;
    stringToWrite+="\n";
  
    constant=std::to_string(y);
    stringToWrite+=constant;
    stringToWrite+="\n";

    constant=std::to_string(offset);
    stringToWrite+=constant;
    stringToWrite+="\n";
  
    //cout<<"\nstringToWrite= "<<stringToWrite;
    /*Output stream to write to file*/
    std::ofstream file;
    file.open(filename);
    file << stringToWrite;
    file.close();
}  
  
  std::vector<float> generateP(std::vector<float> P,uint8_t a,uint8_t b,uint16_t c,float x,float y,float unzero,uint32_t MID,uint32_t TOTAL)
  {  
    
    //std::vector<float> P(TOTAL);
    printf("\nIn GenerateP\n");
    //float intermediate_x=0,intermediate_y=0;
    printf("\nBefore GenP loop\n");
    for(uint32_t i=0;i<MID+1;++i)
    { 
      
      x=fmod((x+a*y),1.0)+unzero;
      y=fmod((b*x+c*y),1.0)+unzero;
      //printf("\ni=%d\t",i);
      P[2*i]=x;
      P[2*i+1]=y;
    }    
    
    return P;
  } 
    

  std::vector<uint32_t> generateU(std::vector<uint32_t> U,std::vector<float> P,int32_t r,uint32_t M,uint32_t N)
  {
    printf("\nIn genU\n");
    //float dN=(float)N;
    float remainder;
    //std::vector<uint32_t> U(M);
    //cout<<"\nBefore entering the genU loop\n";
    printf("\nBeforeGenU Loop\n");
    for(uint16_t i=0;i<M;++i)
    {
      
      //printf("\ni=%d\t",i); 
      remainder=fmod((P[r+i]*EXP),N); 
      //printf("\n%f",remainder);
      U[i]=remainder;
    }
   
   return U; 
 }   
     
 std::vector<uint8_t> flattenGrayScaleImage(cv::Mat image,uint32_t M,uint32_t N)
 {
    std::vector<uint8_t> img_vec(M*N);
    image=cv::Mat(img_vec).reshape(-1);
    
    return img_vec;
 }

#endif
