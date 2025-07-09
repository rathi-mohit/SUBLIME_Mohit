#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <ctime> 
#include <cmath>
#include <fstream>
using namespace std;
using namespace Eigen;

/* NOTE: The code takes a very long time (t = (around) 1.5 hrs) to run, mainly due to the lasso cross validation code,
ridge+other code takes around 3 to 4 minutes. */


double ridge(const MatrixXd&X, const VectorXd&y, double lambda, VectorXd& b)
{
    MatrixXd A=MatrixXd::Identity(X.cols(), X.cols());
    A(0,0)=0; //this is to prevent regularizing the constant term b0
    b=(X.transpose()*X+lambda*A).ldlt().solve((X.transpose())*y);
    double MSE=(y-X*b).squaredNorm()/X.rows();
    return MSE;
}

void coord_descent(const MatrixXd&X, double lambda, VectorXd& b, int k,const VectorXd& r, double ss)
{
    double m=X.col(k).dot(r)+ss*b(k);

    if(m>lambda/2)
    {
       b(k)=(m-lambda/2)/ss;
    }
    else if(m<-lambda/2)
    {
       b(k)=(m+lambda/2)/ss;
    }
    else
    {
        b(k)=0;
    }
}

double lasso(const MatrixXd&X, const VectorXd&y, double lambda, VectorXd& b)
{
    b=VectorXd::Zero(X.cols());
    VectorXd b_old=VectorXd::Ones(X.cols());
    b(0)=y.mean();
    b_old(0)=y.mean();
    int f=0;

    VectorXd r=y-X*b;

    //storing all column norms (squared) in a vector so that we don't calculate it each time.
    VectorXd ss=X.colwise().squaredNorm(); 
    while((b-b_old).lpNorm<Infinity>()>0.0000001 && f<100)
    {
        b_old=b;
        for(int i=1; i<X.cols(); i++)
        {
            coord_descent(X, lambda, b, i, r, ss(i)); 
            //r=y-Xb, b has changed, so we update r to reflect the change
            r=r-(b(i)-b_old(i))*X.col(i);
        }
        f=f+1;
    }
    double MSE=(y-X*b).squaredNorm()/X.rows();
    return MSE;
}

//standardizing the data
void standardize(MatrixXd&X, VectorXd&stddev)
{   
    for(int i=1; i<X.cols(); i++)
    {
        double x_mean=X.col(i).mean();
        X.col(i)=(X.col(i).array()-x_mean).matrix();
        double x_stddev=sqrt(X.col(i).dot(X.col(i))/(X.rows()));
        X.col(i)=X.col(i)/x_stddev;
        stddev(i)=x_stddev; //we store the stddev values for rescaling
    }
}

//rescaling the values of b
//note that the matrix here X is the unstandardized matrix
void rescale(VectorXd&b, MatrixXd&X, VectorXd&stddev)
{
    for(int i=1; i<X.cols(); i++)
    {
        b(i)=b(i)/stddev(i);
        b(0)=b(0)-X.col(i).mean()*b(i);
    }
}

//cross validation for lasso. 
//It would have been more efficient if I did cross validation for both lasso and ridge simulatneously,
//but it would have looked really messy + I wanted separate code for ridge and lasso

void cv_lasso(const MatrixXd&X, const VectorXd&y, double& lambda,const vector<MatrixXd>& X_tra,const vector<VectorXd>& Y_tra, const vector<MatrixXd>& X_test,const vector<VectorXd>& Y_test)
{
    int s=X.rows()/10; //some rows (<10) are left out. you could probably repeat some rows to prevent this
    double MSE;
    double MSE_min=10000; 
    VectorXd b(X.cols());
    float ind;
    double l;
    
    for(float i=-6; i<=4; i=i+0.1) 
    {
       l=pow(10, i); //log scale for lambda
       MSE=0;

       for(int j=0; j<10; j++)
       {
          lasso(X_tra[j], Y_tra[j], l, b);  //training the model
          MSE=MSE+(Y_test[j]-X_test[j]*b).squaredNorm()/s; //testing the model
       }
       MSE=MSE/10; //cross validation error

       //finding model with minimum cross validation error
       if(MSE<MSE_min)
       {
        MSE_min=MSE;
        ind=i;
        lambda=l;
       }

       //if the minimum does not change for a few consecutive cases,
       //we assume that our current minimum is the global minimum
       if(i-ind>=0.6)
       {
         break;
       }
    }

}

void cv_ridge(const MatrixXd& X, const VectorXd& y, double& lambda, const vector<MatrixXd>& X_tra,const vector<VectorXd>& Y_tra, const vector<MatrixXd>& X_test,const vector<VectorXd>& Y_test)
{
    int s=(X.rows()/10); 
    double MSE;
    double MSE_min=100000;
    VectorXd b(X.cols());
    float ind;
    double l;

    for(float i=-6; i<=4; i=i+0.1) //log scale for lambda
    {
       MSE=0;
       l=pow(10, i);
       for(int j=0; j<10; j++)
       {
          ridge(X_tra[j], Y_tra[j], l, b); //training the model
          MSE=MSE+(Y_test[j]-X_test[j]*b).squaredNorm()/s; //testing the model
       }
       MSE=MSE/10; //cross-validation error.

       //minimum cross validation error
       if(MSE<MSE_min)
       {
        MSE_min=MSE;
        ind=i;
        lambda=l;
       }

       //if the minimum does not change for a few consecutive cases,
       //we assume that our current minimum is the global minimum
       if(i-ind>=0.6)
       {
         break;
       }
    }
}


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
    MatrixXd a(199999, 10);
    VectorXd v(199999);
    string s;
    getline(file, s);
    int q=0;
    for(int i=0;i<199999;i++)
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
        if(0.1<=a(q,9)&&300>=a(q,9)) //removes odd outliers, limits values to the range 100m to 500km.
        q++;
            
    }
    a.conservativeResize(q,10);
    v.conservativeResize(q);

    MatrixXd a_old=a;
    VectorXd stddev(a.cols());
    standardize(a,stddev);
    VectorXd b_ridge(a.cols());
    VectorXd b_lasso(a.cols());
    double l_ridge;
    double l_lasso;

    // dividing the training data into 10 parts for 10-fold cross validation:
    vector<MatrixXd> X_tra(10), X_test(10);
    vector<VectorXd> Y_tra(10), Y_test(10);
    int si=a.rows()/10;

    for(int i=0; i<10; i++)
    {
        X_tra[i].resize(9*si, a.cols());
        X_test[i].resize(si, a.cols());

        Y_tra[i].resize(9*si);
        Y_test[i].resize(si);
        

        X_tra[i]<<a.middleRows(0,i*si), a.middleRows((i+1)*si,(9-i)*si);
        Y_tra[i]<<v.middleRows(0,i*si), v.middleRows((i+1)*si,(9-i)*si);

        X_test[i]=a.middleRows(si*i, si);
        Y_test[i]=v.middleRows(si*i, si);
    }

    

    cout<<"Ridge:"<<endl;
    cv_ridge(a, v, l_ridge, X_tra, Y_tra, X_test, Y_test);
    double MSE_ridge=ridge(a, v, l_ridge, b_ridge);
    double TSS=(v.array()-v.mean()).matrix().squaredNorm();
    double Rsq_ridge=1-(a.rows()*MSE_ridge)/TSS;
    rescale(b_ridge, a_old, stddev);
    cout<<"Coefficients: \n"<<b_ridge<<endl;
    cout<<"Lambda: "<<l_ridge<<endl;
    cout<<"R^2: "<<Rsq_ridge<<endl;

    cout<<"Lasso:"<<endl;
    cv_lasso(a, v, l_lasso, X_tra, Y_tra, X_test, Y_test);
    double MSE_lasso=lasso(a, v, l_lasso, b_lasso);
    double Rsq_lasso=1-(a.rows()*MSE_lasso)/TSS;
    rescale(b_lasso, a_old, stddev);
    cout<<"Coefficients: \n"<<b_lasso<<endl;
    cout<<"Lambda: "<<l_lasso<<endl;
    cout<<"R^2: "<<Rsq_lasso<<endl;
}

/*  The code takes a very long time (t>1 hour) to run, mainly due to the lasso code,
ridge+other code takes around 3 to 4 minutes.

Ridge:
Coefficients: 
   -1153.56
   0.581983
   0.089631
-0.00114742
    6.30939
  -0.166793
   -5.94574
    0.51528
  0.0658539
    1.79902
Lambda: 50.1183
R^2: 0.634062

Lasso:
Coefficients: 
 -1154.34
 0.577677
0.0870845
        0
0.0702294
-0.102575
        0
 0.150293
0.0596077
  1.79846
Lambda: 3162.24
R^2: 0.63007

Lasso does zero out some of the coefficients

I should have probably used a different dataset with n closer to p, since this dataset has 
a large n compared to p, it is probably not ideal for ridge and lasso.
*/
