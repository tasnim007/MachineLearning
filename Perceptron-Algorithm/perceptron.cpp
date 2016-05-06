#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LEARNING_RATE 0.1

float x[208],y[208];
int catg[208];
float w[3]={0.1};



int calculateOutput(float w[],float x,float y)
{
    float sum=x*w[0]+y*w[1]+w[2];
   	return (sum>=0)?1:0;
}



void test(){

	float lerr,gerr;
    int err_count;
    int patternCount,i,j,iteration,output,tmp;


	FILE *ftest;
    if ((ftest=fopen("Test.txt","r"))==NULL)
    {
        printf("Cannot open test file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(ftest,"%f %f %d",&x[i],&y[i],&catg[i])!=EOF)
    {
        i++;
    }
    patternCount=i;

    err_count=0;

    for (j=0;j<patternCount;j++)
    {
        output=calculateOutput(w,x[j],y[j]);
        lerr=catg[j]-output;
        gerr=gerr+abs(lerr);
        tmp=int(gerr);
        if(tmp!=0)
            err_count++;
    }
    printf("Testing data count=%d\n",patternCount);
    printf("Testing error count=%d\n",err_count);
    err_count=patternCount-err_count;
    printf("Percentage of Accuracy=%.2f%% \n",(err_count)*100.0/patternCount);
}

int main()
{

    float lerr,gerr;
    int err_count;
    int patternCount,i,j,iteration,output,tmp;

    FILE *ftrain;
    if ((ftrain=fopen("Train.txt","r"))==NULL)
    {
        printf("Cannot open train file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(ftrain,"%f %f %d",&x[i],&y[i],&catg[i])!=EOF)
    {
        i++;
    }
    patternCount=i;

    printf("Training data count=%d\n",patternCount);
   
    iteration=0;
    do
    {
        iteration++;
        gerr=0;
        for (j=0;j<patternCount;j++)
        {
            output=calculateOutput(w,x[j],y[j]);
            lerr=catg[j]-output;

            w[0]=w[0]+LEARNING_RATE*lerr*x[j];
            w[1]=w[1]+LEARNING_RATE*lerr*y[j];
            w[2]=w[2]+LEARNING_RATE*lerr;

            gerr=gerr+abs(lerr);
        }
        //printf("Iteration %d : Global Err=%f\n",iteration,gerr);

    }while(gerr!=0);

    printf("\nEquation: %.2f*x+(%.2f*y)+(%.2f)=0\n",w[0],w[1],w[2]);



   

    test();


    return 0;
}
