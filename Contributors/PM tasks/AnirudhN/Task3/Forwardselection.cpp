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

//residual standard error
float RSE(VectorXf Y,VectorXf YCap, long int prm)
{float rr=0;
   for(long long int i=0;i<Y.size();i++)
     rr+=(Y[i]-YCap[i])*(Y[i]-YCap[i]);
rr=(float)sqrt(rr/(Y.size()-prm-1));
return rr;
}



//split fields


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
float mean(VectorXf Y)
{float ss=0;//sum
    for(long int i=0;i<Y.size();i++)
      ss+=Y(i);
return (float)ss/Y.size();
}

//Forward bias
int PIN(int prm,VectorXf Y,vector<int>flds,MatrixXf mat,float PreRSE,VectorXf Bo)
{ MatrixXf mi(199999,prm+1);
   VectorXf Ycap(199999),Bcap(prm+1); 
   int ct1=-1;
    for(int i=0;i<flds.size();i++)
    {
        mi.col(i)=mat.col(flds[i]);//fill existing values
    }
    int k=0;
 for(int i=0;i<11;i++)
 {int sk=0;
    for(int j=0;j<flds.size();j++)
    {if (i==flds[j])
       sk=1;
       break;
        
    }
      if(sk==0)
{mi.col(prm)=mat.col(i);//prm +1 in total
VectorXf Bcap=((mi.transpose()*mi).inverse())*(mi.transpose()*Y);
   for(long long int i=0;i<Ycap.size();i++)
{Ycap(i)=0;
    for(int j=0;j<prm;j++)
   { Ycap(i)+=mi(i,j)*Bcap(j);}}
 float RS2=RSE(Y,Ycap,prm);
   if(RS2<PreRSE)
   {
    PreRSE=RS2;
    ct1=i;
   } }

    
 }

 if(ct1==-1)
 {for(int i=0;i<Bo.size();i++)
   cout<<Bo(i)<<" "<<flds[i]<<endl;

   return 0;
}
 else {flds.emplace_back(ct1);
    if(prm<15)
        PIN(prm+1,Y,flds,mat,PreRSE,Bcap);
    else for(int i=0;i<Bcap.size();i++)
          cout<<Bcap(i)<<" "<<flds[i]<<endl;
return 0;
}
}



//Main


int main()
{   MatrixXf mat(199999,12),m1(199999,1),m2(199999,2);

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
          
            goto skp;
        }
    }
    try
     {Y(ct)=stof(fields[2]);
    for(int i=4;i<fields.size();i++) 
       {mat(ct,i-4)=stof(fields[i]);
    
    } mat(ct,11)=1;}


    catch (const std::exception& e) {
        cerr << "Conversion error at line " << ct << ": " << e.what() << " for value: " << fields[1] << endl;}
          ct++; 
          skp: ;
     }
  
m1.col(0)=mat.col(11);
//0 parms
for(long long int i=0;i<Ycap.size();i++)
{Ycap(i)=mean(Y);}
float RS1=RSE(Y,Ycap,0);
float min=RS1;
int ct1=-1;
//1 params
VectorXf Bcap;
m2.col(0)=mat.col(11);
for(int i=0;i<11;i++)
{m2.col(2)=mat.col(i);
    Bcap=((m2.transpose()*m2).inverse())*(m2.transpose()*Y);
    for(long long int i=0;i<Ycap.size();i++)
{Ycap(i)=0;
    {for(int j=0;j<2;j++)
    Ycap(i)+=m2(i,j)*Bcap(j);}}
    float RS2=RSE(Y,Ycap,1);
   if(RS2<min)
   {
    min=RS2;
    ct1=i;
   } 
  


}
vector<int> fld;
fld.emplace_back(11);
 if(ct1==-1)
   {
    cout<<"Only the constant is used "<<mean(Y);
    return 0;
   }
 else{fld.emplace_back(ct1);

//Parm 2

PIN(2, Y,fld, mat,min, Bcap );
}
 return 0;
   }
