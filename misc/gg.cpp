#include<bits/stdc++.h>
using namespace std;
double l_rate=0.1;
double input[768][10];//dataset 
double output[768]={0};
int in_no;
float findmax(int pos)
{
	float a=INT_MIN;
	for(int i=0;i<768;i++)
	{
		if(a<input[i][pos])
			a=input[i][pos];
	}
	return a;
}
float findmin(int pos)
{
	float a=INT_MAX;
	for(int i=0;i<768;i++)
	{
		if(a>input[i][pos])
			a=input[i][pos];
	}
	return a;
}
void normalize(int pos)   //normalisation of dataset to sacle between 0 to 1
{
	float max=findmax(pos);
	float min=findmin(pos);
	float val=max-min;
	for(int i=0;i<768;i++)
	{
		input[i][pos]=(input[i][pos]-min)/val;
	}
}
float sigmoid(float x)
{
	return 1/(1+exp(-x));
}
float div_sigmoid(float x)
{
	return sigmoid(x)*(1-sigmoid(x));
}
double error[768];
// hidden layer weight
double hiddenNode[8]={0};
double hiddenZNode[8]={0};
double input_weight[8][8];
double output_weight[8];
double bias;
double b[8];

void initalize()    //intialisation of weights between 0 to 1
{
	bias=1;
	for(int i=0;i<8;i++)
	{
		output_weight[i]=(double)rand() / (double)((unsigned)RAND_MAX+1);
		b[i]=(double)rand() / (double)((unsigned)RAND_MAX+1);
		for(int j=0;j<8;j++)
		{
			input_weight[i][j]=(double)rand() / (double)((unsigned)RAND_MAX+1);
		}
	}
}
void feedforward(int cols)   //feedforward calculation
{
	for(int i=0;i<8;i++)
	{
		hiddenNode[i]=0;
		for(int j=0;j<8;j++)
		{
			hiddenNode[i]=hiddenNode[i]+(input[cols][i]*input_weight[j][i]);
		}
		hiddenNode[i]=hiddenNode[i]+b[i];
		hiddenZNode[i]=sigmoid(hiddenNode[i]);
	}
	for(int i=0;i<8;i++)
	{
		output[cols]=output[cols]+(hiddenZNode[i]*output_weight[i]);
	}
		output[cols]=output[cols]+bias;
		output[cols]=sigmoid(output[cols]);
}
double es=0;
void update(int cols) //updates of input_weight and output_weights
{
	double de=0,e=0,output_weight_div[8],delin[8],del[8];
	de=(input[cols][9]-output[cols])*div_sigmoid(output[cols]);
	e=(input[cols][9]-output[cols]);
	es=es+0.5*e*e;
	for(int i=0;i<8;i++)
	{
		output_weight_div[i]=l_rate*de*hiddenZNode[i];
		delin[i]=de*output_weight[i];
		del[i]=delin[i]*div_sigmoid(hiddenZNode[i]);			
	}
	for(int i=0;i<8;i++)
	{
		for(int j=0;j<8;j++)
		{
			input_weight[i][j]=input_weight[i][j]+(l_rate*del[j]*input[cols][i]);
		}
	}
	for(int i=0;i<8;i++)
	{
		b[i]=b[i]+(l_rate*del[i]);
		output_weight[i]=output_weight[i]+output_weight_div[i];
		
	}

}

int main()
{
	srand(time(0));
	 std::fstream myfile1("D:\\strp.txt", std::ios_base::in);
	 ofstream myf;
	    myf.open("D:/out.txt");
    double a;
    int count=0;

    int bount=0;
	int flag=0;

    while (flag<768)     //fetch the data from txt file into input
    {
            bount=0;
            int l=0;
            while(myfile1 >> a)
             {
               
                 input[flag][l]=a;
                 l++;
                 bount++;
                 if(bount==10)
                   {
						break;
					}
             }
             flag++;
	}
	myfile1.close();
	double arr[155][8];
	int k=0;
	for(int i=615;i<768;i++)
	{
		for(int j=0;j<8;j++)
			{
				arr[k][j]=input[i][j];
			}
			k++;
	}
	for(int i=0;i<8;i++)
	{
		normalize(i);
	}
	initalize();
	// print_weight();
	for(int div=1;div<=6;div++)
	{
		int num=0;
		 num=num+(614/6)*div;    //k-fold cross validation of dataset
	for(int ephoch=0;ephoch<1000;ephoch++)
	{
		
	for(int i=0;i<num;i++)
	{
		feedforward(i);
		update(i);
		
	}
		
	}
}
	// print_weight();
	for(int i=615;i<768;i++)
	{
		feedforward(i);
		if(output[i]<0.5)
		{
			cout<<"output_value="<<output[i]<<"  predicted output= "<<"0"<<"  actual output= "<<input[i][9]<<endl;
		}
		else
		{
			cout<<"output_value="<<output[i]<<"  predicted output= "<<"1"<<"  actual output= "<<input[i][9]<<endl;
		}
		
		
	if((output[i]<0.5 && input[i][9]==0)||(output[i]>=0.5 && input[i][9]==1))
	{
		count++;
	}

	}
	int count1=0;
	for(int i=0;i<615;i++)
	{
		feedforward(i);
		if((output[i]<0.5 && input[i][9]==0)||(output[i]>=0.5 && input[i][9]==1))
		{
			count1++;
		}
		
	}
	cout<<"train_accouracy= "<<((double)count1/615.0)*100<<endl;
	for(int i=615;i<768;i++)
	{
		
		
		for(int j=0;j<8;j++)
		{
			myf<<arr[i-615][j]<<",";
		}
		myf<<output[i]<<",";
		if(output[i]<0.5)
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
		cout<<"test_accuracy= "<<(count/154.0)*100<<"%"<<endl;
	
//	printall();	
		
}
