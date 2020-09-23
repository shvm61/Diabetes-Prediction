#include<bits/stdc++.h>
using namespace std;
int dataset=768; //number of rows of data in dataset
int parameters=10; //number of parameters in a rows
int train_set=615;// 80% of data from dataset for training 
int test_set=152;//20% of data from dataset for testing
int p2=8;
double input[1000][10];
double arr[155][8];
double coefficient[10];
double l_rate=0.1;
int in_no;
//generate random number
double getRand(void)
{
 	return ((double)rand())/(double)RAND_MAX;
}
void random_coefficient()               //initialisation of coefficient with random number range  from 0 to 1
{
	for(int i=0;i<parameters;i++)
	coefficient[i]=(double)rand() / (double)((unsigned)RAND_MAX+1);
}
float findmax(int pos)           //find maximum from every columns of dataset
{
	float a=INT_MIN;
	for(int i=0;i<dataset;i++)
	{
		if(a<input[i][pos])
			a=input[i][pos];
	}
	return a;
}
float findmin(int pos)    //find minimum from every columns of dataset
{
	float a=INT_MAX;
	for(int i=0;i<dataset;i++)
	{
		if(a>input[i][pos])
			a=input[i][pos];
	}
	return a;
}
void normalize(int pos)   //normalisation of dataset to scale between 0 to 1
{
	float max=findmax(pos);
	float min=findmin(pos);
	float val=max-min;
	for(int i=0;i<dataset;i++)
	{
		input[i][pos]=(input[i][pos]-min)/val;
	}
}
float sigmoid(float x) 
{
	return (1/(1+exp(-1*x)));
}
double yhat,expected;
void predict(int cols)  //calculate the likelihood
{
	yhat=0,expected=0;
	yhat=coefficient[0];
	for(int i=0;i<p2;i++)
		yhat=yhat+coefficient[i+1]*input[cols][i];
	expected=sigmoid(yhat);
}

double error;
int divide=5;
void train()
{
	for(int div=1;div<=6;div++)
	{
		int num=0;
		 num=num+((train_set-1)/6)*div;        //k- fold cross validation of dataset
		for(int ephoch=0;ephoch<1000;ephoch++) //iterate every k-fold dataset 1000 times
		{
			
			
			for(int i=0;i<num;i++)
			{
				
				error=0.0;	
				predict(i);
				error=expected-input[i][9];
				coefficient[0]=coefficient[0]-l_rate*error*expected*(1.0-expected);   //update bias
				for(int k=0;k<p2;k++)
				{
					coefficient[k+1]=coefficient[k+1]-l_rate*expected*error*(1.0-expected)*input[i][k]; 
					//update other coefficients
				}			
			}
		
		}
	}

}
void test()
{
	int count=1;
	ofstream myf;
	myf.open("D:/out1.txt");
	for(int i=train_set;i<dataset;i++)
	{
		predict(i);
		if(expected<0.5)
		{
			cout<<"value= "<<expected<<" "<<"predicted output= "<<"0"<<" actual output= "<<input[i][9]<<endl;
		}
		else
		{
			cout<<"value= "<<expected<<" "<<"predicted output= "<<"1"<<" actual output= "<<input[i][9]<<endl;
		}
		if((expected<0.5 && input[i][9]==0) || (expected>=0.5 && input[i][9]==1) )
		{
			count++;
		}
		
			for(int j=0;j<p2;j++)
			{
				myf<<arr[i-train_set][j]<<",";
			}
			myf<<expected<<",";
			if(expected<0.5)
			{
				myf<<"0"<<","<<input[i][9]<<",";
				if(input[i][9]==0)
				{
					myf<<"correct"<<endl;
				}
				else
				{
					myf<<"incorrect"<<endl;
				}
			}
			else
			{
				myf<<"1"<<","<<input[i][9]<<",";
				if(input[i][9]==1)
				{
					myf<<"correct"<<endl;
				}
				else
				{
					myf<<"incorrect"<<endl;
				}
			}
	}
	cout<<"test_accuracy= "<<((double)count/(double)test_set)*100<<"%"<<endl;
	int count1=0;
	for(int i=0;i<train_set;i++)
	{
		predict(i);
		if((expected<0.5 && input[i][9]==0) || (expected>=0.5 && input[i][9]==1) )
		{
			count1++;
		}
	}
	cout<<"train_accuracy= "<<((double)count1/(double)train_set)*100<<"%"<<endl;
}
int main()
{
	srand(time(0));
	std::fstream myfile1("D:\\strp.txt", std::ios_base::in);
    double a;
    int count=0;

    int bount=0;
	int flag=0;

    while (flag<dataset)
    {
            bount=0;
            int l=0;
            while(myfile1 >> a)
             {
               
                 input[flag][l]=a;
                 l++;
                 bount++;
                 
                 if(bount==parameters)
                   {
						break;
					}
             }
             flag++;
	}
	int k=0;
	for(int i=train_set;i<dataset;i++)
	{
		for(int j=0;j<p2;j++)
		{
			arr[k][j]=input[i][j];
		}
		k++;
	}
	for(int i=0;i<p2;i++)
	{
		normalize(i);
	}
	random_coefficient();
	test();
	cout<<endl;
	// print_coeff();
	train();
	
	test();
	cout<<endl;
		
}
