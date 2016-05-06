#include <iostream>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <iterator>
#include <ctime>

using namespace std;


#define MAXATTRVAL 10	//each attribute can take 1-10 value


vector<vector<int> > data;
vector<vector<int> > test_data;
vector<int> attributes;
int thisClass;
double gAccuracy, gPrecision, gRecall, gFmeasure, gGmean;


struct node{
    int attrid;
    bool leaf;
    node *child[MAXATTRVAL];
    node() {leaf = true;}
    node(int id) {
        attrid = id;
        leaf = false;
        memset(child, NULL, sizeof(child));
    }
};


double entropy(vector<vector<int> > d, int attr=-1, int val=-1)
{
    int i,j,p,len = d.size();
    for(i=0,p=0;i<len;i++){
        if(attr != -1 && d[i][attr] != val)   continue;
        if(d[i][thisClass] == 1)  p++;
    }

    if(p == len || p == 0) return 0;
    if(p == len-p)  return 1;

    double tot = (len) * 1.0;
    return -( (p/tot)*( log(p/tot)/log(2) ) ) - ( ((len-p)/tot)*( log((len-p)/tot)/log(2) ) );
}

double gain(vector<vector<int> > d, int attr)
{
    int vals[MAXATTRVAL+1];
    int i,j,len = d.size();
    double e = entropy(d);

    memset(vals, 0, sizeof(vals));
    for(i=0;i<len;i++)  vals[d[i][attr]]++;

    for(i=1;i<=MAXATTRVAL;i++){
        if(vals[i] == 0)    continue;
        double en = entropy(d, attr, i);
        e -= ( vals[i]/(len*1.) ) * en;
    }

    return e;
}

vector<vector<int> >get_child_data(vector<vector<int> > D, int attr, int val)
{
    vector<vector<int> > temp;
    int i,len = D.size();
    for(i=0;i<len;i++){
        if(D[i][attr] == val)   temp.push_back(D[i]);
    }
    return temp;
}


int most_common_value(vector<vector<int> > D)
{
    int i,pos,neg,len = D.size();
    for(i=0, pos=0;i<len;i++){
        if(D[i][thisClass] == 1)  pos++;
    }
    neg = len -pos;
    if(pos>=neg)    return 1;
    else return 0;
}

node *build_ID3_tree(vector<vector<int> > D)
{
    node *root;
    if(D.empty() || attributes.empty()){    //attribute is empty
        root = new node();
        root->attrid = most_common_value(D);
        return root;
    }

    double en = entropy(D);                 //entropy 0 means all are positive or all are negative
    if( fabs(en-0.0) <= 1e-7 ){
        root = new node();
        if(D[0][thisClass] == 1)  root->attrid = 1;
        else  root->attrid = 0;
        return root;
    }

    int bestAttribute,t=0;
    double max = 0.;
    for(int i=0;i<attributes.size();i++){
        double temp = gain(D, attributes[i]);
        //printf("gain(%d): %lf\n",attributes[i],temp);
        if(temp > max){
                t=i;
            max = temp;
            bestAttribute = attributes[i];
        }
    }
    if(bestAttribute > 9)    root = build_ID3_tree(D);
    else{
        root = new node(bestAttribute);
        //printf("creating node %d    %d    %d   %d\n",bestAttribute,t,attributes.size(),attributes[t]);


        vector<int>::iterator it = attributes.begin();
        while(it != attributes.end()){
            if(*it == bestAttribute)    break;
            it++;
        }
        if(it != attributes.end())  attributes.erase(it);

        for(int i=1;i<=MAXATTRVAL;i++){
            vector<vector<int> > childData = get_child_data(D,bestAttribute,i);
            root->child[i-1] = build_ID3_tree(childData);
        }
    }

    return root;
}

int id3_tree_output(node *root, vector<int> example){
    if(root == NULL) {printf("NULL tree!\n"); return -1;}
    while(!root->leaf){
        root = root->child[ example[root->attrid]-1 ]; //example[root->attrid] can be any int betn 1 to 10
    }
    return root->attrid;
}

void test_data_analysis(node *root){
    int i,len = test_data.size();
    int TP,FP,TN,FN;
    TP=FP=TN=FN=0;
    for(int i=0;i<len;i++){
        vector<int> v = test_data[i];
        vector<int>::iterator it = v.end();
        it--;
        int originalVal = *it;
        v.erase(it);
        int trainVal = id3_tree_output(root, v);
        //printf("test(%d): %d->%d\n",i+1,val,valout);

        if(originalVal==1 && trainVal==1) TP++;
        else if(originalVal==1) FN++;
        else if(trainVal==1)  FP++;
        else TN++;
    }

    printf("TP : %d\n",TP);
    printf("TN : %d\n",TN);
    printf("FP : %d\n",FP);
    printf("FN : %d\n",FN);

    double accuracy = ((TP+TN)*1.0)/(TP+TN+FP+FN);
    double precision = (TP*1.0)/(TP+FP);
    double recall = (TP*1.0)/(TP+FN);
    double fmeasure = (2.0*precision*recall)/(precision+recall);
    double gmean = sqrt( (TP*TN*1.0)/((TP+FN)*(TN+FP)) );

    printf("Accuracy: %lf\n",accuracy);
    printf("Precision: %lf\n",precision);
    printf("Recall: %lf\n",recall);
    printf("F-Measure: %lf\n",fmeasure);
    printf("G-Mean: %lf\n",gmean);

    gAccuracy += accuracy;
    gPrecision += precision;
    gRecall += recall;
    gFmeasure += fmeasure;
    gGmean += gmean;
}

void free_ID3_tree(node *root)
{
    if(root->leaf)  {free(root); return;}

    for(int i=0;i<MAXATTRVAL;i++){
        free_ID3_tree(root->child[i]);
    }

    free(root);
}


void read_test_and_train_data()
{
    char line[100],*p;
    int i,v;
    vector<int> temp;

    if(!data.empty()){
        data.clear();
    }
    if(!test_data.empty()){
        test_data.clear();
    }


    while(gets(line)){
        p = strtok(line,", ");
        temp.clear();
        while(p!=NULL){
            v = atoi(p);
            temp.push_back(v);
            p = strtok(NULL,",");
        }
        data.push_back(temp);
    }

    if(data.empty())    return;

    int len = data.size();
    int n = ceil(len * 0.25);
   // printf("testdata: %d\n",n);
    for(i=0;i<n;i++){
        int r = rand()%len;
       // printf("%d ",r);
        test_data.push_back(data[r]);
        data.erase(data.begin() + r);
        len--;
    }
    printf("\n");

}

int main()
{
    node *root;
    srand((unsigned)time(NULL));
    gAccuracy = gPrecision = gRecall = gFmeasure = gGmean = 0.0;

    for(int m=0;m<10;m++){
        printf("For Iteration %d:\n",m);
        FILE *fp=freopen("data.csv","r",stdin);
        if(fp==NULL){
            printf("File doesn't exist !!");
            exit(1);
        }

        root = NULL;
        read_test_and_train_data();

        attributes.clear();
        for(int i=0;i<9;i++){       //Total Attribute no. is 9
            attributes.push_back(i);
        }
        thisClass =  9; //target attribute

        //cout<<attributes.size()<<"  ";

        root = build_ID3_tree(data);
        //printf("tree built successfully!\n");
        test_data_analysis(root);
        free_ID3_tree(root);
        fclose(fp);
        printf("\n\n");
    }

    printf("Average Result: \n\n");

    printf("Accuracy: %lf\n",gAccuracy/10.0);
    printf("Precision: %lf\n",gPrecision/10.0);
    printf("Recall: %lf\n",gRecall/10.0);
    printf("F-Measure: %lf\n",gFmeasure/10.0);
    printf("G-Mean: %lf\n",gGmean/10.0);

    return 0;
}
