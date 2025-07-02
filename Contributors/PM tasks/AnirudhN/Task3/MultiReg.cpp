#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include<fstream>
#include<regex>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include<math.h>
using namespace std;
using namespace Eigen;
float RSE(VectorXf Y,VectorXf YCap, long int prm)
{float rr=0;
   for(long long int i=0;i<Y.size();i++)
     rr+=(Y[i]-YCap[i])*(Y[i]-YCap[i]);
rr=(float)sqrt(rr/(Y.size()-prm-1));
return rr;
}
long double Fstat(VectorXf Y,VectorXf YCap, long int prm)
{
long double mn=0;
 for(long long int i=0;i<YCap.size();i++)
  mn+=Y(i);
  mn=(long double)mn/YCap.size();
  long double rr=0;
   for(long long int i=0;i<Y.size();i++)
     rr+=(Y[i]-YCap[i])*(Y[i]-YCap[i]);
     long double R0=0;
     for(long long int i=0;i<Y.size();i++)
     R0+=(Y[i]-mn)*(Y[i]-mn);
     return (long double)(R0-rr)*(Y.size()-prm-1)/(rr*prm);

}

long double Rsq(VectorXf Y,VectorXf YCap, long int prm)
{
    long double mn=0;
 for(long long int i=0;i<Y.size();i++)
  mn+=Y(i);
  mn=(long double)mn/Y.size();
  long double rr=0;
   for(long long int i=0;i<Y.size();i++)
     rr+=(Y[i]-YCap[i])*(Y[i]-YCap[i]);
     long double R0=0;
     for(long long int i=0;i<Y.size();i++)
     R0+=(Y[i]-mn)*(Y[i]-mn);
     return (long double) (1-rr/R0);
}
vector<string> splt(const string& ln)  //split fields
{ stringstream ss(ln);
vector<string> fields;
string fld;
while(getline(ss,fld,','))
{
   fld.erase(0,fld.find_first_not_of("\t"));
fld.erase(fld.find_last_not_of("\t")+1);
fields.push_back(fld);


}



string input = fields[3];
    std::regex pattern(R"((\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2}) UTC)");
    std::smatch matches;

    if (std::regex_search(input, matches, pattern)) {
        for(int i=1;i<7;i++)
       fields.push_back(matches[i]);//to get dd tt mm yy
       

    }



return fields;
}



int main()
{   MatrixXf mat(199999,12);
    Eigen::VectorXf Y(199999),Ycap(199999); // target variable
  vector<long int>skips;
   string flname="uber.csv";
   ifstream file(flname);
   if(!file.is_open())
   {cout<<"File not open";
      return 1;}
   
 long long int ct=0 ;//for counting lines
 string ln;
 getline(file,ln);//To bypass head
 while(getline(file,ln))
{
    vector<string> fields;
    fields=splt(ln);
  if(fields.size()<15 ) {skips.push_back(0);continue;}
    for(int i=0;i<fields.size();i++)
    {if(fields[i].empty()) 
        {   skips.push_back(i);
          
            goto skp;}
    }
    try
     {Y(ct)=stof(fields[2]);
    for(int i=4;i<fields.size();i++)//ERRORHEREERRORHEREERROEHEREERROE
    {mat(ct,i-4)=stof(fields[i]);
    
    } mat(ct,11)=1;
     } catch (const std::exception& e) {
        cerr << "Conversion error at line " << ct << ": " << e.what() << " for value: " << fields[1] << endl;}
          ct++;
          skp: ;//no empty rows
    
}
//accounting for Y removals


VectorXf B_cap=((mat.transpose()*mat).inverse())*(mat.transpose()*Y);
for(long long int i=0;i<200000-skips.size();i++)
{Ycap(i)=0;
    for(int j=0;j<12;j++)
  {Ycap(i)+=B_cap(j)*mat(i,j);
  }

}Ycap.resize(200000-skips.size());
cout<<"coefficients:"<<endl;
for (float x : B_cap) {
        std::cout << x << std::endl;
    
    }
cout<<"RSE"<<RSE(Y,Ycap,11)<<endl;
cout<<"Rsquare:"<<Rsq(Y,Ycap,11)<<endl;
cout<<"Fstastistic:"<<Fstat(Y,Ycap,11)<<endl;

    return 0;
}

//g++ MultipleReg.cpp -o MultipleReg -I C:/toolbox/eigen-3.4.0
