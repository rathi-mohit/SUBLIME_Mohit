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

//I know that introducing a variable such as dist would introduce collinearity, 
//I just wanted to see how it would affect regression and forward selection.

double dist(long double lat1, long double lat2, long double lon1, long double lon2)
{
    long double difflat=(lat2-lat1)*M_PI/180.0;
    long double difflong=(lon2-lon1)*M_PI/180.0;
    int r=6371;
    double d=2*r*asin(sqrt((1-cos(difflat)+cos(lat1*M_PI/180)*cos(lat2*M_PI/180)*(1-cos(difflong)))/2));
    return d;
}

void fss(const MatrixXd& X,const Eigen::VectorXd& y, int p)
{
    vector<int> v(p);
    vector<double> AIC(p);
    vector<double> BIC(p);
    vector<double> Cp(p);
    vector<double> AdjRsq(p);
    vector<Eigen::VectorXd> M(p); //stores the value of the estimators of each model, from size to 1 to p.
    vector<int> predictors;  //temporarily stores the index of each estimator used in a model
    vector<vector<int>> M_list(p); //stores the index of each estimator used in each model
    Eigen::VectorXd b;
    Eigen::VectorXd bmax;
    double rss;
    double rss_max;
    double tss=(y.mean()-y.array()).matrix().squaredNorm();
    double Rsq;
    double Rsq_max;
    //finding variance
    Eigen::VectorXd r1=(X.transpose()*X).inverse()*X.transpose()*y;
    double r2=(y-X*r1).squaredNorm();
    double var=r2/(X.rows()-(p+1));;
    int k;
    //pool of estimators which can be added:
    for(int i=0; i<p; i++)
    {
        v[i]=i+1;
    }

    for(int i=0; i<p; i++)
    {
        Rsq_max=0;
        for(int j=0; j<p-i; j++)
        {
            MatrixXd W(X.rows(),i+2); //matrix with observations corresponding to i+1 predictors.
            W.col(0)=X.col(0); //Column of ones (intercept term)
            for(int g=0; g<i; g++)
            {
                W.col(g+1)=X.col(M_list[i-1][g]); //previously chosen predictors
            }
            W.col(i+1)=X.col(v[j]); //adding a new estimator to see if it improves the model
            b=(W.transpose()*W).inverse()*W.transpose()*y; //least squares solution
            rss=(y-W*b).squaredNorm();
            Rsq=1-rss/tss;
            //finding model with maximum R_sq
            if(Rsq>Rsq_max)
            {
                Rsq_max=Rsq;
                rss_max=rss;
                bmax=b;
                k=v[j];
            }
        }
        AIC[i]=(rss_max+2*(i+1)*var)/(var*X.rows()); //find AIC, BIC, Cp and adjusted R^2 for the models
        BIC[i]=(rss_max+2*log(X.rows())*(i+1)*var)/X.rows();
        Cp[i]=(rss_max+2*(i+1)*var)/X.rows(); 
        AdjRsq[i]=1-(rss_max*(X.rows()-1))/(tss*(X.rows()-i-2));
        predictors.push_back(k); //adding k to the list of predictors, of the model with i+1 estimators
        M_list[i]=predictors; //adding the list of predictors to the M_list array
        v.erase(remove(v.begin(), v.end(), k), v.end()); //removing the estimator from the pool of estimators which can be added
        M[i]=bmax;
    }

    double AIC_min=AIC[0];
    double Cp_min=Cp[0];
    double BIC_min=BIC[0];
    double AdjR_max=AdjRsq[0];
    int AIC_minpos=0;
    int BIC_minpos=0;
    int Cp_minpos=0;
    int AdjR_maxpos=0;
    //finding minimum Cp, AIC, BIC, maximum adjusted R^2
    for(int i=1; i<p; i++)
    {
        if(AIC[i]<AIC_min)
        { 
           AIC_min=AIC[i];
           AIC_minpos=i;
        }
        if(Cp[i]<Cp_min)
        {
            Cp_min=Cp[i];
            Cp_minpos=i;
        }
        if(BIC[i]<BIC_min)
        {
            BIC_min=BIC[i];
            BIC_minpos=i;
        }
         if(AdjRsq[i]>AdjR_max)
        {
            AdjR_max=AdjRsq[i];
            AdjR_maxpos=i;
        }
    }

    cout<<"Min Cp: "<<Cp_min<<" No of parameters: "<<(Cp_minpos+1)<<endl;
    cout<<"Estimators: \n"<<M[Cp_minpos]<<endl;

    //displaying the parameter which was chosen:
    //1-year, 2-month, 3-time, 4-pickup longitude, 5-pickup latitude
    //6-dropoff longitude, 7-dropoff latitude, 8-no of passengers, 9-distance

    cout<<"Parameters chosen: "<<endl;
    for(int i=0; i<M_list[Cp_minpos].size()-1; i++)
    {
        cout<<M_list[Cp_minpos][i]<<", ";
    }
    cout<<M_list[Cp_minpos][M_list[Cp_minpos].size()-1]<<endl;

    cout<<"Min AIC: "<<AIC_min<<" No of parameters: "<<(AIC_minpos+1)<<endl;
    cout<<"Estimators: \n"<<M[AIC_minpos]<<endl;

    cout<<"Parameters chosen: "<<endl;
    for(int i=0; i<M_list[AIC_minpos].size()-1; i++)
    {
        cout<<M_list[AIC_minpos][i]<<", ";
    }
    cout<<M_list[AIC_minpos][M_list[AIC_minpos].size()-1]<<endl;
    
    cout<<"Min BIC: "<<BIC_min<<" No of parameters: "<<(BIC_minpos+1)<<endl;
    cout<<"Estimators: \n"<<M[BIC_minpos]<<endl;

    cout<<"Parameters chosen: "<<endl;
    for(int i=0; i<M_list[BIC_minpos].size()-1; i++)
    {
        cout<<M_list[BIC_minpos][i]<<", ";
    }
    cout<<M_list[BIC_minpos][M_list[BIC_minpos].size()-1]<<endl;

    cout<<"Max adjusted R^2: "<<AdjR_max<<" No of parameters: "<<(AdjR_maxpos+1)<<endl;
    cout<<"Estimators: \n"<<M[AdjR_maxpos]<<endl;

    cout<<"Parameters chosen: "<<endl;
    for(int i=0; i<M_list[AdjR_maxpos].size()-1; i++)
    {
        cout<<M_list[AdjR_maxpos][i]<<", ";
    }
    cout<<M_list[AdjR_maxpos][M_list[AdjR_maxpos].size()-1]<<endl;

    //AIC, Cp and adjR^2 end up choosing a model with 9 parameters. BIC chooses the model with 8 parameters.
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
            a(q,j)=stold(s);
        }
        getline(file, s, '\n');
        a(q,8)=stod(s);

        a(q,9)=dist(a(q,5), a(q,7), a(q,4), a(q,6)); //distance
        if(0.1<=a(q,9)&&500>=a(q,9)) //removes unreasonable outliers, limits values to the range 100m to 500km.
        q++;
      
    }
    a.conservativeResize(q,p+1);
    v.conservativeResize(q);
    //cout<<"No of valid observations: "<<v.size()<<endl;
    fss(a, v, p);
}