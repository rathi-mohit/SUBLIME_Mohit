#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <fstream>
#include <string>
#include <ctime> 
#include <cmath>
using namespace std;
using Eigen::MatrixXd;

//the observation on line 87948 had an error so I removed it

//Variables used for regression: Year, Month, Time, Pickup Longitude, Pickup Latitude
//Dropoff Longitude, Dropoff Latitude, Number of Passengers, Distance

double dist(long double lat1, long double lat2, long double lon1, long double lon2)
{
    long double difflat=(lat2-lat1)*M_PI/180.0;
    long double difflong=(lon2-lon1)*M_PI/180.0;
    int r=6371;
    double d=2*r*asin(sqrt((1-cos(difflat)+cos(lat1*M_PI/180)*cos(lat2*M_PI/180)*(1-cos(difflong)))/2));
    return d;
}

void calc(MatrixXd X,const Eigen::VectorXd y, Eigen::VectorXd& b, double& mse, Eigen::VectorXd& t, double& f, double& rse, double& Rsq, int N, int p)
{
    b=(X.transpose()*X).inverse()*X.transpose()*y;
    double rss=(y-X*b).squaredNorm();  
    MatrixXd Var=(rss/(N-p-1))*(X.transpose()*X).inverse(); //Covariance matrix
    mse=rss/N;
    for(int i=0; i<p+1; i++)
    {
        t(i)=b(i)/sqrt(Var(i,i)); //t-statistics
    }

    double tss=(y.mean()-y.array()).matrix().squaredNorm();

    f=(tss-rss)/(p)*(N-p-1)/rss; //f-statistic
  
    rse=sqrt(rss/(N-p-1)); //rse
    
    Rsq=1-rss/tss; //R^2
}
int main()
{
    int p=9;
    ifstream file("uber.csv");
    string s="";
    MatrixXd a(200000, p+1);
    Eigen::VectorXd v(200000);
    getline(file, s); //header line
    int q=0;
    vector<double long> g(4);
    for(int u=0; u<199999; u++)
    {
        getline(file, s, ',');
        getline(file, s, ',');

        getline(file, s, ',');
        v(q)=stod(s); 
        
        getline(file, s, ',');
        a(q,0)=1; //column corresponding to b0 (constant term)
        a(q,1)=stoi(s.substr(0,4)); //year
        a(q,2)=stoi(s.substr(5,2)); //month
        a(q,3)=stold(s.substr(11,2))+stod(s.substr(14,2))/60+stod(s.substr(17,2))/3600; //time in hrs, with t=0 being 00:00 AM
        
        for(int j=4; j<8; j++)
        {
            getline(file, s, ',');
            a(q,j)=stold(s); //latitudes and longitudes
        }
        getline(file, s, '\n');
        a(q,8)=stod(s); //number of passengers

        a(q,9)=dist(a(q,5), a(q,7), a(q,4), a(q,6)); //distance
        if(0.1<=a(q,9)&&500>=a(q,9)) //removes odd outliers, limits values to the range 100m to 500km.
        q++;
    
      
    }
    a.conservativeResize(q,p+1);
    v.conservativeResize(q);
    cout<<"No of valid observations: "<<v.size()<<endl;
    Eigen::VectorXd b(p+1);
    Eigen::VectorXd t(p+1);
    double mse;
    double f;
    double rse;
    double Rsq;
    calc(a, v, b, mse, t, f, rse, Rsq, q, p);
    cout<<"Value of estimators \t t-statistic"<<endl;
    for(int i=0; i<p+1; i++)
    {
        cout<<b(i) <<"\t\t"<<t(i)<<endl;
    }
    cout<<"\nF-statistic: "<<f<<endl;
    cout<<"RSE: "<<rse<<endl;
    cout<<"MSE: "<<mse<<endl;
    cout<<"Rsquared: "<<Rsq<<endl;
}

//F-statistic: 32666.6
//RSE: 5.98823
//MSE: 35.8571
//Rsquared: 0.604.
//The model is not too good but there does seem to exist a relation between the predictors and the response,
//as implied by the high F-statistic. 
//It seems that distance, year, latitude and longitude have the most effect. 
