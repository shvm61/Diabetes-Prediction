#include<bits/stdc++.h>
using namespace std;
#define tr_size 537 //training dataset size
#define ts_size 231 //testing dataset size
double train_input[tr_size][8],train_output[tr_size],test_input[ts_size][8],test_output[ts_size];
double maxx[8]={0},minx[8]={0};
double mean0[8],mean1[8],sdsq0[8],sdsq1[8];
int count0=0,count1=0;
//reading csv file and store data to separe training and testing part
void read_csvfile(){
	fstream fin;
	fin.open("diabetes.csv");
    int a=0;
    string word;
    int i=0,flag=0;
    while (fin.good()) { 
  		string line;
        getline(fin, line);
        if(a==0)
        {
        	a=1;
        	continue;
		}
        stringstream s(line);
        int count=0;
        
        while (getline(s, word, ',')) { 
        	stringstream geek(word);
  			double x=0.0;
  			geek >> x;
  			if(count==8)
            {
            	if(flag==0)
            	{
            		train_output[i]=x;
            		continue;
				}
				else
				{
					test_output[i]=x;
					continue;
				}
			}
            
  			if(flag==0){
  					train_input[i][count]=x;
			  	}
			else
			{
				test_input[i][count]=x;
			}
            count++;
            
        }
        i++;
        if(i==tr_size)
        {
        	i=0;
			flag=1;	
		}
    }
}
//calculating mean for training data part
void calmean()
{
	double sum0[8]={0},sum1[8]={0};
	
		for(int i=0;i<tr_size;i++)
		{
			if(train_output[i]==0)
			{
				for(int j=0;j<8;j++)
					sum0[j]+=train_input[i][j];
			}
			else
			{
				for(int j=0;j<8;j++)
					sum1[j]+=train_input[i][j];
			}
		}
		for(int i=0;i<8;i++)
		{
			mean0[i]=sum0[i]/(double)count0;
			mean1[i]=sum1[i]/(double)count1;
		}
}
//calculating square of standard deviation from training dataset
void cal_std_dev_sq()
{
	double sqsum0[8]={0},sqsum1[8]={0};
	for(int i=0;i<tr_size;i++)
	{
		if(train_output[i]==0)
		{
			for(int j=0;j<8;j++)
			{
				sqsum0[j]+=((train_input[i][j]-mean0[j])*(train_input[i][j]-mean0[j]));
			}
		}
		else
		{
			for(int j=0;j<8;j++)
			{
				sqsum1[j]+=((train_input[i][j]-mean1[j])*(train_input[i][j]-mean1[j]));
			}
		}
	}
	for(int i=0;i<8;i++)
	{
		sdsq0[i]=sqsum0[i]/(double)(count0-1);
		sdsq1[i]=sqsum1[i]/(double)(count1-1);
	}
}
//printing data
void print_data()
{
	for(int i=0;i<tr_size;i++)
    {
    	for(int j=0;j<8;j++)
    	{
    		cout<<train_input[i][j]<<"   ";
		}
		cout<<train_output[i];
		cout<<endl;
	}
}
//calculating max for normalization
void find_max()
{
	
	for(int i=0;i<8;i++)
	{
		maxx[i]=INT_MIN;
		for(int j=0;j<tr_size;j++)
		{
			if(maxx[i]<train_input[j][i])
			{
				maxx[i]=train_input[j][i];
			}
		}
	}
	
}
//calculating min for normalization
void find_min()
{
	for(int i=0;i<8;i++)
	{
		minx[i]=INT_MAX;
		for(int j=0;j<tr_size;j++)
		{
			if(minx[i]>train_input[j][i])
			{
				minx[i]=train_input[j][i];
			}
		}
	}
	
}
//normalizing training dataset
void train_normalize()
{
	find_max();
	find_min();
	for(int i=0;i<tr_size;i++)
	{
		for(int j=0;j<8;j++)
		{
			train_input[i][j]=(train_input[i][j]-minx[j])/(maxx[j]-minx[j]);
		}
	}
}
//calculating class 0 probability for feature of training dataset
double train_calculate_0prob(int i,double p0)
{
	double prod=1.0;
	for(int j=0;j<8;j++)
	{
		if(sdsq0[j]==0)
			continue;
		double temp1=(double)2*(3.141)*sdsq0[j];
		temp1=sqrt(temp1);
		temp1=(double)1/temp1;
		double temp2=((train_input[i][j]-mean0[j])*(train_input[i][j]-mean0[j]));
		temp2=temp2/((double)2*sdsq0[j]);
		temp2=-temp2;
		temp2=exp(temp2);
		prod=prod*(temp2*temp1);
	}
	prod=prod*p0;
	return prod;
	
}
//calculating class 1 probabilities for feacture of training dataset
double train_calculate_1prob(int i,double p1)
{
	double prod=1.0;
	for(int j=0;j<8;j++)
	{
		if(sdsq1[j]==0)
			continue;
		double temp1=(double)2*(3.141)*sdsq1[j];
		temp1=sqrt(temp1);
		temp1=(double)1/temp1;
		double temp2=((train_input[i][j]-mean1[j])*(train_input[i][j]-mean1[j]));
		temp2=temp2/((double)2*sdsq1[j]);
		temp2=-temp2;
		temp2=exp(temp2);
		prod=prod*(temp2*temp1);
	}
	prod=prod*p1;
	return prod;
	
}
//calculating class 0 probabilities for feacture of testing dataset
double test_calculate_0prob(int i,double p0)
{
	double prod=1.0;
	for(int j=0;j<8;j++)
	{
		if(sdsq0[j]==0)
			continue;
		double temp1=(double)2*(3.141)*sdsq0[j];
		temp1=sqrt(temp1);
		temp1=(double)1/temp1;
		double temp2=((test_input[i][j]-mean0[j])*(test_input[i][j]-mean0[j]));
		temp2=temp2/((double)2*sdsq0[j]);
		temp2=-temp2;
		temp2=exp(temp2);
		prod=prod*(temp2*temp1);
	}
	prod=prod*p0;
	return prod;
	
}
//calculating class 1 probabilities for feacture of testing dataset
double test_calculate_1prob(int i,double p1)
{
	double prod=1.0;
	for(int j=0;j<8;j++)
	{
		if(sdsq1[j]==0)
			continue;
		double temp1=(double)2*(3.141)*sdsq1[j];
		temp1=sqrt(temp1);
		temp1=(double)1/temp1;
		double temp2=((test_input[i][j]-mean1[j])*(test_input[i][j]-mean1[j]));
		temp2=temp2/((double)2*sdsq1[j]);
		temp2=-temp2;
		temp2=exp(temp2);
		prod=prod*(temp2*temp1);
	}
	prod=prod*p1;
	return prod;
	
}
int main()
{
	read_csvfile();
	//train_normalize();
	print_data();
	//counting total class 1 and class 0 
	for(int i=0;i<tr_size;i++)
	{
		if((int)train_output[i]==1)
			count1++;
		else
			count0++;
	}
	//calculating class probability
	double p0=(double)count0/(double)tr_size;
	double p1=(double)count1/(double)tr_size;
	//calculate mean
	calmean();
	//calculate standard deviation
	cal_std_dev_sq();
	int count=0;
	//cout<<"train_probability:"<<endl;
	//making prediction for training dataset
	for(int i=0;i<tr_size;i++)
	{
		int out;
		double a0=train_calculate_0prob(i,p0);
		double a1=train_calculate_1prob(i,p1);
		if(a0>=a1)
			out=0;
		else
			out=1;
		if(out==(int)train_output[i])
			count++;
		//cout<<"p(0)= "<<a0/(a0+a1)<<"     "<<"p(1)= "<<a1/(a0+a1)<<"   "<<"pred_out= "<<out<<"   "<<"actual output= "<<train_output[i]<<endl;
	}
	//calculating training accuracy
	double train_accuracy=((double)(count)/(double)tr_size)*100;
	//test_normalize();
	ofstream myf; // to store output testing data to files
	myf.open("D:/out2.txt");
	//making prediction for testing dataset
	count=0;
	cout<<"test probability:"<<endl;
	for(int i=0;i<ts_size;i++)
	{
		for(int j=0;j<8;j++)
		{
			myf<<test_input[i][j]<<",";
		}
		int out;
		double a0=test_calculate_0prob(i,p0);
		double a1=test_calculate_1prob(i,p1);
		myf<<(a0/(a0+a1))*100<<","<<(a1/(a0+a1))*100<<",";
		if(a0>=a1)
		{
			out=0;
			myf<<"0"<<","<<test_output[i]<<endl;
		}
		else
		{
			out=1;
			myf<<"1"<<","<<test_output[i]<<endl;
		}
		if(out==(int)test_output[i])
			count++;
		cout<<"p(0)= "<<(a0/(a0+a1))*100<<"     "<<"p(1)= "<<(a1/(a0+a1))*100<<"   "<<"pred_out= "<<out<<"    "<<"actual output= "<<test_output[i]<<endl;
	}
	double test_accuracy=((double)(count)/(double)ts_size)*100;
	cout<<"train_accuracy="<<train_accuracy<<endl;
	cout<<"test_accuracy="<<test_accuracy<<endl;
}
