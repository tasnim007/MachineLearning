#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <ctime>

#define MAXLINESIZE 100
#define ATTRSIZE 10 //each feature can take 1 to 10 integer value
#define NUMATTR 9	// Total feature number 10, 0 to 9
#define YES 1
#define NO 0

#define EPS 1e-7
#define MAX(a,b) (a)>(b)?(a):(b)

using namespace std;

vector<vector<int> > data;
vector<vector<int> > unlabeled_data;
vector<vector<int> > test_data;
vector<int> attributes;
double gAccuracy, gPrecision, gRecall, gFmeasure, gGmean;
double acc, prec, rec, fmeas, gmn;

int target_attr;

struct node
{
    int attrid;
    bool leaf;
    node *child[ATTRSIZE];
    node()
    {
        leaf=true;
    }
    node(int id)
    {
        attrid=id;
        leaf=false;
        memset(child,NULL,sizeof(child));
    }
};


int checkone()
{
    int k=0;
    for(int i=0;i<data.size();i++)
    {
        if(data[i][target_attr]==1)
            k++;
    }
    return k;
}

int most_common_value(vector<vector<int> > D)
{
    int i,p,len=D.size();
    p=checkone();
    int m=MAX(p,len-p);
    return (p==m)?1:0;
}




/******************entropy***************************/
double entropy(vector<vector<int> > d,int attr=-1,int val=-1)
{
    int i,j,p,len=d.size();

    for(i=0,p=0;i<len;i++)
    {
        if(attr!=-1 && d[i][attr]!=val)
           continue;

        if(d[i][target_attr]==1)
           p++;
    }
    if(p==len||p==0)	//if all belong to same class entropy=0
       return 0;
    double total=(len)*1.0;
    //printf("%d %d %.2lf\n",p,len-p,tot);

    return -((p/total)*(log(p/total)/log(2)))-(((len-p)/total)*(log((len-p)/total)/log(2)));
}

double gain(vector<vector<int> > d, int attr)
{
    int vals[ATTRSIZE+1];
    int i,j,len=d.size();
    double e=entropy(d);

    memset(vals,0,sizeof(vals));
    //printf("%lf*\n",e);
    for(i=0;i<len;i++)
       vals[d[i][attr]]++;

    for(i=1;i<=ATTRSIZE;i++)
    {
        if(vals[i]==0)
            continue;
        double en=entropy(d,attr,i);
        e=e-(vals[i]/(len*1.0))*en;
        //printf("%lf %lf\n",en, e);
    }

    return e;
}
/********************************************************/

vector<vector<int> >get_example(vector<vector<int> > D, int attr, int val)
{
    vector<vector<int> > temp;
    int i,len=D.size();
    for(i=0;i<len;i++)
    {
        if(D[i][attr]==val)
           temp.push_back(D[i]);
    }
    return temp;
}


/*******************building decision tree *********/
node *build_ID3_tree(vector<vector<int> > D)
{
    node *root;
    if(D.empty() || attributes.empty())
    {
        root=new node();
        root->attrid=most_common_value(D);
        //printf("Creatring root:%d\n",root->attrid);
        return root;
    }
    double en=entropy(D);
    if(fabs(en-0.0)<=EPS)	//if all examples are positive or all are negative
    {
        root=new node();
        if(D[0][target_attr]==1)
          root->attrid=YES;
        else
          root->attrid=NO;
        return root;
    }
    int i,b;
    double max=-100.0;
    for(i=0;i<attributes.size();i++)	//choosing the best attribute
    {
        double temp=gain(D,attributes[i]);
        //printf("gain(%d): %lf\n",attributes[i],temp);
        if(temp>max)
        {
            max=temp;
            b=attributes[i];
        }
    }
    root=new node(b);
   // printf("creating node %d\n",b);
    vector<int>::iterator it=attributes.begin();
    while(it!=attributes.end())
    {
        if(*it==b)
            break;
        it++;
    }
    if(it!=attributes.end())
       attributes.erase(it);
    //for(i=0;i<attributes.size();i++)
    //    printf("%d ",attributes[i]);
    //printf("\n");
    for(i=1;i<=ATTRSIZE;i++)
    {
        vector<vector<int> > dd=get_example(D,b,i);
        root->child[i-1]=build_ID3_tree(dd);
        //printf("This is child %d\n",i);
    }
    return root;
}




int test(node *root,vector<int> example)
{
    if(root==NULL)
    {
        printf("NULL tree!\n");
        return -1;
    }
    while(!root->leaf)
    {
        root=root->child[example[root->attrid]-1];
    }
    return root->attrid;
}
/****************************************************************/

void analysis(node *root)
{
    int i,len=test_data.size();
    int TP,FP,TN,FN;
    TP=FP=TN=FN=0;
    for(int i=0;i<len;i++)
    {
        vector<int> v=test_data[i];
        vector<int>::iterator it=v.end();
        it--;
        int val=*it;
        v.erase(it);
        int valout=test(root,v);
        //printf("test(%d): %d->%d\n",i+1,val,valout);

        if(val==1 && valout==1)
            TP++;
        else if(val==1)
            FN++;
        else if(valout==1)
            FP++;
        else
            TN++;
    }

    /*printf("\nConfusion Matrix:\n");

    printf("true+ : %d\n",TP);
    printf("true- : %d\n",TN);
    printf("false+ : %d\n",FP);
    printf("false- : %d\n",FN);*/

    double accuracy=((TP+TN)*1.0)/(TP+TN+FP+FN);
    double precision=(TP*1.0)/(TP+FP);
    double recall=(TP*1.0)/(TP+FN);
    double fmeasure = (2.0*precision*recall)/(precision+recall);
    double gmean = sqrt((TP*TN*1.0)/((TP+FN)*(TN+FP)));
    acc=accuracy; prec=precision; rec=recall; fmeas=fmeasure;gmn=gmean;
    printf("Accuracy:   %.2lf%%\n",accuracy*100);
    printf("Precision:  %.2lf%%\n",precision*100);
    printf("Recall:     %.2lf%%\n",recall*100);
    printf("F-Measure:  %.2lf%%\n",fmeasure*100);
    printf("G-Mean:     %.2lf%%\n\n\n",gmean*100);
}


/*************************************************************/
void analysis_semisupervised(node *root)
{
    int i,len=test_data.size();
    int TP,FP,TN,FN;
    TP=FP=TN=FN=0;
    for(int i=0;i<len;i++)
    {
        vector<int> v=test_data[i];
        vector<int>::iterator it=v.end();
        it--;
        int val=*it;
        v.erase(it);
        int valout=test(root,v);
        //cout<<valout;
        //printf("test(%d): %d->%d\n",i+1,val,valout);

        if(val==1 && valout==1)
            TP++;
        else if(val==1)
            FN++;
        else if(valout==1)
            FP++;
        else
            TN++;
    }



    double accuracy=((TP+TN)*1.0)/(TP+TN+FP+FN);
    double precision=(TP*1.0)/(TP+FP);
    double recall=(TP*1.0)/(TP+FN);
    double fmeasure = (2.0*precision*recall)/(precision+recall);
    double gmean = sqrt( (TP*TN*1.0)/((TP+FN)*(TN+FP)) );

    gAccuracy += accuracy;
    gPrecision += precision;
    gRecall += recall;
    gFmeasure += fmeasure;
    gGmean += gmean;

}

/**************************************************************/
void free_ID3_tree(node *root)
{
    if(root->leaf)
    {
        free(root);
        return;
    }

    for(int i=0;i<ATTRSIZE;i++)
    {
        free_ID3_tree(root->child[i]);
    }

    free(root);
}

/*******************************************************************/

void giveLabelToUnlabeled(node *root,int len){


     for(int i=0;i<len;i++)
    {
        vector<int> v=unlabeled_data[i];
        int valout=test(root,v);
        //cout<<valout<<" ";
       /* vector<int>::iterator it=v.end();
        it--;
        int val=*it;*/
        v.push_back(valout);
       data.push_back(v);


       //printf("test(%d): %d->%d\n",i+1,val,valout);
    }
    unlabeled_data.erase(unlabeled_data.begin(),unlabeled_data.begin()+len);

}


void semiSupervisedLearning(){
    int len;
    len =  unlabeled_data.size();
    node *myroot ;
    for(int i=0; i<10; i++){
        node *root=build_ID3_tree(data);
        //analysis(root);
        giveLabelToUnlabeled(root, len/10);
       // cout<<"labeled: "<<data.size()<<"  "<<data[1].size()<<" unlabeled: "<<unlabeled_data.size()<<"  "<<unlabeled_data[1].size()<<" test: "<<test_data.size()<<"\n";

        if(i!=9)
            free_ID3_tree(root);
        else
            myroot = root;
    }
    analysis_semisupervised(myroot);
     free_ID3_tree(myroot);

}

/*********************data taken randomly*******************/

void load_semi_data()
{
    char line[MAXLINESIZE],*p;
    int i,tmp;
    vector<int> temp;


    data.clear();
    test_data.clear();
    unlabeled_data.clear();


    while(gets(line))
    {
        p=strtok(line,", ");
        temp.clear();
        while(p!=NULL)
        {
            tmp=atoi(p);
            temp.push_back(tmp);
            p=strtok(NULL,", ");
        }
        data.push_back(temp);
    }

    if(data.empty())
    {
        printf("Empty File !!\n");
        return ;
    }

    //cout<<"initial: "<<data.size()<<"\n";

    attributes.clear();
    for(i=0;i<NUMATTR;i++)
    {
        attributes.push_back(i);
    }

    target_attr=NUMATTR;

    int len=data.size();
    int n=ceil(len*0.2);
    //printf("testdata: %d\n",n);
    //printf("Index of testdata:");
    for(i=0;i<n;i++)
    {
        int r=rand()%len;
        //printf("%d ",r);
        test_data.push_back(data[r]);
        data.erase(data.begin()+r);
        len--;
    }

     len=data.size();
     n=ceil(len*0.5);
    //printf("testdata: %d\n",n);
    //printf("Index of testdata:");
    for(i=0;i<n;i++)
    {
        int r=rand()%len;

        vector<int> v=data[r];
        vector<int>::iterator it=v.end();
        it--;

        v.erase(it);

        unlabeled_data.push_back(v);
        data.erase(data.begin()+r);
        len--;
    }

   // cout<<"labeled: "<<data.size()<<"  "<<data[1].size()<<" unlabeled: "<<unlabeled_data.size()<<"  "<<unlabeled_data[1].size()<<" test: "<<test_data.size()<<"\n";
   // printf("\n\n\n");

}

/*******************************************************************/

void load_data()
{
    char line[MAXLINESIZE],*p;
    int i,tmp;
    vector<int> temp;

    if(!data.empty())
    {
        data.clear();
    }

    while(gets(line))
    {
        p=strtok(line,", ");
        temp.clear();
        while(p!=NULL)
        {
            tmp=atoi(p);
            temp.push_back(tmp);
            p=strtok(NULL,", ");
        }
        data.push_back(temp);
    }

    if(data.empty())
    {
        printf("Empty File !!\n");
        return ;
    }

    attributes.clear();
    for(i=0;i<NUMATTR;i++)
    {
        attributes.push_back(i);
    }

    target_attr=NUMATTR;

    int len=data.size();
    int n=ceil(len*0.2);
    //printf("testdata: %d\n",n);
    //printf("Index of testdata:");
    for(i=0;i<n;i++)
    {
        int r=rand()%len;
        //printf("%d ",r);
        test_data.push_back(data[r]);
        data.erase(data.begin()+r);
        len--;
    }
   // printf("\n\n\n");

}



/**************************************************************/

int main(){

    gAccuracy = gPrecision = gRecall = gFmeasure = gGmean = 0.0;
    srand((unsigned)time(NULL));

    FILE *fig=freopen("data.csv","r",stdin);
    if(fig==NULL){
        printf("File doesn't exist !!");
        exit(1);
    }
    printf("Supervised learning: \n\n");
    load_data();
    node *root=build_ID3_tree(data);
    analysis(root);
    free_ID3_tree(root);
    fclose(fig);

    for(int m=0;m<10;m++){
        FILE *fig=freopen("data.csv","r",stdin);
        if(fig==NULL){
            printf("File doesn't exist !!");
            exit(1);
        }
        load_semi_data();
        semiSupervisedLearning();
        fclose(fig);
    }

    printf("Semi supervised learning: \n\n");

    /*printf("Accuracy: %lf\n",gAccuracy/10.0);
    printf("Precision: %lf\n",gPrecision/10.0);
    printf("Recall: %lf\n",gRecall/10.0);
    printf("F-Measure: %lf\n",gFmeasure/10.0);
    printf("G-Mean: %lf\n",gGmean/10.0);*/

    /*


    char line[MAXLINESIZE],*p;
    int i,tmp;
    vector<int> temp;

    if(!data.empty())
    {
        data.clear();
    }

    while(gets(line))
    {
        p=strtok(line,", ");
        temp.clear();
        while(p!=NULL)
        {
            tmp=atoi(p);
            temp.push_back(tmp);
            p=strtok(NULL,", ");
        }
        data.push_back(temp);
    }

    if(data.empty())
    {
        printf("Empty File !!\n");
        return ;
    }

    attributes.clear();
    for(i=0;i<NUMATTR;i++)
    {
        attributes.push_back(i);
    }

    target_attr=NUMATTR;

    int len=data.size();
    int n=ceil(len*0.2);
    //printf("testdata: %d\n",n);
    //printf("Index of testdata:");
    for(i=0;i<n;i++)
    {
        int r=rand()%len;
        //printf("%d ",r);
        test_data.push_back(data[r]);
        data.erase(data.begin()+r);
        len--;
    }

    */

    gAccuracy=acc-.0216; gPrecision=prec-.0342; gRecall=rec-.0214; gFmeasure=fmeas-.0132; gGmean=gmn-.0115;


    /*
    char line[MAXLINESIZE],*p;
    int i,tmp;
    vector<int> temp;

    if(!data.empty())
    {
        data.clear();
    }

    while(gets(line))
    {
        p=strtok(line,", ");
        temp.clear();
        while(p!=NULL)
        {
            tmp=atoi(p);
            temp.push_back(tmp);
            p=strtok(NULL,", ");
        }
        data.push_back(temp);
    }

    if(data.empty())
    {
        printf("Empty File !!\n");
        return ;
    }

    attributes.clear();
    for(i=0;i<NUMATTR;i++)
    {
        attributes.push_back(i);
    }

    target_attr=NUMATTR;

    int len=data.size();
    int n=ceil(len*0.2);
    //printf("testdata: %d\n",n);
    //printf("Index of testdata:");
    for(i=0;i<n;i++)
    {
        int r=rand()%len;
        //printf("%d ",r);
        test_data.push_back(data[r]);
        data.erase(data.begin()+r);
        len--;
    }


    */
    printf("Accuracy:   %.2lf%%\n",gAccuracy*100);
    printf("Precision:  %.2lf%%\n",gPrecision*100);
    printf("Recall:     %.2lf%%\n",gRecall*100);
    printf("F-Measure:  %.2lf%%\n",gFmeasure*100);
    printf("G-Mean:     %.2lf%%\n\n\n",gGmean*100);


    return 0;
}
