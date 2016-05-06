#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define LEARNING_RATE 0.1
#define MAX_ITR 20

float x[208],y[208];
int catg[208];
float w[3]={0.1};
float ws[3];

float init()
{
    return (rand()%20/45.0);
}

int calculateOutput(float w[],float x,float y)
{
    float sum=x*w[0]+y*w[1]+w[2];
    return (sum>=0)?1:-1;
}

int main()
{

    float lerr,gerr,hold=100.0;
    int err_count;
    int patternCount,i,j,iteration,output,tmp;

    srand((unsigned)time(NULL));

    FILE *ftrain;
    if ((ftrain=fopen("train1.txt","r"))==NULL)
    {
        printf("Cannot open train file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(ftrain,"%f %f %d",&x[i],&y[i],&catg[i])!=EOF)
    {
        if(catg[i]==0)
        {
            catg[i]=-1;
        }
        i++;
    }
    patternCount=i;

    printf("Training data count=%d\n",patternCount);
    /*---------------training phase--------------------*/
    ws[0]=init();
    ws[1]=init();
    ws[2]=init();

    printf("Initial ws: %f %f %f\n",ws[0],ws[1],ws[2]);

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

        if(hold>gerr)
        {
            hold=gerr;
            ws[0]=w[0];
            ws[1]=w[1];
            ws[2]=w[2];
        }
        printf("Iteration %d : Global Err=%f, RMSE = %.4f\n",iteration,gerr,sqrt(gerr/patternCount));
        //printf("\nEquation: %.2f*x+(%.2f*y)+(%.2f)=0\n",ws[0],ws[1],ws[2]);
    }while(iteration<MAX_ITR);

    printf("\nEquation: %.2f*x+(%.2f*y)+(%.2f)=0\n",ws[0],ws[1],ws[2]);



    /*----------------testing phase----------------*/

    FILE *ftest;
    if ((ftest=fopen("test1.txt","r"))==NULL)
    {
        printf("Cannot open test file.\n");
        exit(1);
    }

    i = 0;
    while (fscanf(ftest,"%f %f %d",&x[i],&y[i],&catg[i])!=EOF)
    {
        if(catg[i]==0)
        {
            catg[i]=-1;
        }
        i++;
    }
    patternCount=i;

    err_count=0;

    for (j=0;j<patternCount;j++)
    {
        output=calculateOutput(ws,x[j],y[j]);
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


    return 0;
}
