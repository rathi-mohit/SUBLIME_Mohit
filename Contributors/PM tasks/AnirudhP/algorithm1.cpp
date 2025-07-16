#include <iostream>
#include <vector>
#include <fstream>
#include <Eigen/Dense>
#include <ctime> 
#include <cmath>
using namespace std;
using namespace Eigen;

int q=0;

bool compSmall(VectorXd a, VectorXd b)
{
    return (a(q+1)<b(q+1));
}

bool compLarge(VectorXd a, VectorXd b)
{
    return (a(q+1)>b(q+1));
}

void rsort(vector<VectorXd> a, MatrixXd& X, VectorXd& y, int r)
{
    int n=a.size();
    int m=a[0].rows();
    //a[i](0) corresponds to the response for all observations i.
    for(q=0; q<m-1; q++)
    {
        //smallest r observations for each feature
        partial_sort(a.begin()+2*q*r, a.begin()+(2*q+1)*r, a.end(), compSmall);
        //largest r observations for each feature
        partial_sort(a.begin()+(2*q+1)*r, a.begin()+(2*q+2)*r, a.end(), compLarge);

        //storing the observations
        for(int j=2*q*r; j<(2*q+2)*r; j++)
        {
           y(j)=a[j](0); //response
           X(j,0)=1;  //column of 1s corresponding to constant term
           X.row(j).tail(m-1)=a[j].tail(m-1).eval(); //
        }  
    }
}

//finding b
VectorXd regression(const MatrixXd& X, const VectorXd& y)
{
    return((X.transpose()*X).ldlt().solve(X.transpose()*y));
}

//calculating covariance matrix
MatrixXd covariance(const MatrixXd& X, const VectorXd& y, const double& RSS)
{
    return((RSS)/(X.rows()-X.cols())*(X.transpose()*X).ldlt().solve(MatrixXd::Identity(X.cols(), X.cols())));
}

//calculates t-statistic of each parameter
VectorXd tstatistic(const MatrixXd& X, const VectorXd& y, const VectorXd& b, const double& RSS)
{
    MatrixXd cov=covariance(X,y,RSS);
    VectorXd t(b.rows());
    for(int i=0; i<b.rows(); i++)
    {
        t(i)=b(i)/sqrt(cov(i,i));
    }
    
    return t;
}

double fstatistic(const double& RSS, const double& TSS, const int& n, const int& p)
{
    return(((TSS-RSS)/(p))/(RSS/(n-p-1)));
}

//calculating the distance
double dist(long double lat1, long double lat2, long double lon1, long double lon2)
{
    long double difflat=(lat2-lat1)*M_PI/180.0;
    long double difflong=(lon2-lon1)*M_PI/180.0;
    int r=6371;
    double d=2*r*asin(sqrt((1-cos(difflat)+cos(lat1*M_PI/180)*cos(lat2*M_PI/180)*(1-cos(difflong)))/2));
    return d;
}

int main()
{
    ifstream file("uber.csv");
    vector<VectorXd> a;
    string s;
    VectorXd d(10);
    int f=0;
    getline(file, s);//header
    for(int i=0; i<199999; i++)
    {
        getline(file, s, ',');
        getline(file, s, ',');

        getline(file, s, ',');
        d(0)=stod(s);

        getline(file, s, ',');
        d(1)=stoi(s.substr(0,4)); //year
        d(2)=stoi(s.substr(5,2)); //month
        d(3)=stold(s.substr(11,2))+stod(s.substr(14,2))/60+stod(s.substr(17,2))/3600; //time in hrs, with t=0 being 00:00 AM
        
        for(int j=4; j<8; j++)
        {
            getline(file, s, ',');
            d(j)=stold(s); //latitudes and longitudes
        }
        getline(file, s, '\n');
        d(8)=stod(s); //number of passengers

        d(9)=dist(d(5), d(7), d(4), d(6)); //distance
        if(0.1<=d(9)&&500>=d(9)) //removes odd outliers, limits values to the range 100m to 500km.
        {
           a.push_back(d);   
        }
    }

    int k=9000; //size of subdata
    MatrixXd X(k,10);
    VectorXd y(k);

    rsort(a, X, y, (int)(k/20));
    //calculating Rsq
    VectorXd b=regression(X,y);
    double RSS=(y-X*b).squaredNorm();
    double TSS=(y.array()-y.mean()).matrix().squaredNorm();
    double Rsq=1-RSS/TSS;

    //t-statistic and f-statistic
    VectorXd t=tstatistic(X, y, b, RSS);
    double fstat=fstatistic(RSS, TSS, X.rows(), X.cols()-1);


    cout<<"Value \t\t"<<"t-statistic"<<endl;
    for(int i=0; i<X.cols(); i++)
    {
        cout<<b(i)<<"\t\t"<<t(i)<<endl;
    }

    cout<<"f-statistic: "<<fstat<<endl;
    cout<<"R-squared: "<<Rsq<<endl;
}

