#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <vector>
#include <iterator>
#include <fstream>
#include <algorithm>
#include <ctime>

using namespace std;

#define ROHT 0.5



vector<vector<float> > data;
vector<vector<float> > keslerVector;
vector<vector<float> > testData;
vector<vector<float> > testKeslerVector;
vector<int> className;
vector<int> testClassName;
vector<float> w;
vector<float> zeroes;


int featureNumber, classNumber, exampleNumber;


vector<vector<float> > construct_KeslerVector(vector<vector<float> > d){
    vector<float> original;
    vector<float> neg;
    vector<float> keslerTemp;
    vector<vector<float> > dd;
    for(int i=0; i<d.size(); i++){
        original.clear();
        neg.clear();
        original = d[i];

        for(int z=0; z<featureNumber+1; z++){
            neg.push_back(-1*original[z]);
        }

        for(int j=1; j<=classNumber; j++){  //to construct (classNumber(M)-1) number vectors from each example
            if(j==className[i]){
                continue;
            }
            else{
                keslerTemp.clear();

                for(int k=1; k<=classNumber; k++){  //kesler vector size is k times of original vector size
                    if(k==className[i]){
                        keslerTemp.insert(keslerTemp.end(),original.begin(),original.end());
                    }
                    else if(k==j){
                         keslerTemp.insert(keslerTemp.end(),neg.begin(),neg.end());
                    }
                    else{
                         keslerTemp.insert(keslerTemp.end(),zeroes.begin(),zeroes.end());
                    }
                }
            }

          dd.push_back(keslerTemp);
        }

    }

    cout<<"\n\nKesler vector: Count = "<<dd.size()<<"   "<<"\n\n";

    /*for(int i=0; i< dd.size(); i++){
       for(int j=0; j<dd[i].size(); j++){
            cout<<dd[i][j]<<"  ";
        }
       // cout<< className[i];
        cout<<"\n";
    }*/

    return dd;
}


void kesler_algorithm(){
    int errorCount, iteration;
    float sum;
    iteration = 0;
    do{
        iteration++;
        errorCount = 0;
        for(int i=0; i< keslerVector.size(); i++){
            sum = 0;
            for(int j=0; j<keslerVector[i].size(); j++){
                sum += keslerVector[i][j]*w[j];
            }
            if(sum <= 0){
                 errorCount++;
                 for(int j=0; j<keslerVector[i].size(); j++){
                    w[j] = w[j] + ROHT*keslerVector[i][j];
                 }
            }
        }
    }while(errorCount!=0);  //until all kesler vectors lie on positive side of decision hyperplane
    cout<<"\n\nTotal iteration: "<<iteration<<"\n\n";
}


void test(){
    int errorCount = 0;
    float sum;
    for(int i=0; i< testKeslerVector.size(); i++){
            sum = 0;
            for(int j=0; j<testKeslerVector[i].size(); j++){
                sum += testKeslerVector[i][j]*w[j];
            }
            if(sum < 0){
                 errorCount++;
            }
        }
      cout<<"\nKeslar Vector Mismatch: "<<errorCount;
}


void readTestData(){
    vector<float> temp;
    float fVal;
    int iVal;
    ifstream read("Test.txt");
    if (read.is_open()){
        //for(int j=0; j<exampleNumber; j++){
        while(!read.eof()){
            temp.clear();
            for(int i=0; i<featureNumber; i++){
                read>>fVal;
                temp.push_back(fVal);
            }
            temp.push_back(1);

            read>>iVal;
            testClassName.push_back(iVal);

            testData.push_back(temp);
        }
    }

   if(testData.empty()){
        printf("Empty File !!");
        return ;
    }

   cout<<"\nOriginal Test Data feature vectors (extended 1 dimension): Count = "<< testData.size() <<"\n\n";
  /* for(int i=0; i< exampleNumber; i++){
        for(int j=0; j<featureNumber+1; j++){
            cout<<testData[i][j]<<"   ";
        }
        //cout<< testClassName[i];
        cout<<"\n";
    }*/
}


void readTrainData(){
    vector<float> temp;
    float fVal;
    int iVal;
    ifstream read("Train.txt");
    if (read.is_open()){
        read>>featureNumber;
        read>>classNumber;
        read>>exampleNumber;
        for(int j=0; j<exampleNumber; j++){
            temp.clear();
            for(int i=0; i<featureNumber; i++){
                read>>fVal;
                temp.push_back(fVal);
            }
            temp.push_back(1); //extending 1 dimension

            read>>iVal;
            className.push_back(iVal);

            data.push_back(temp);
        }
    }

   if(data.empty()){
        printf("Empty File !!");
        return ;
    }


    cout<<"Original Training Data feature vectors (extended 1 dimension): count = "<<data.size()<<"\n\n";
   /* for(int i=0; i< exampleNumber; i++){
        for(int j=0; j<featureNumber+1; j++){
            cout<<data[i][j]<<"   ";
        }
        //cout<< className[i];
        cout<<"\n";
    }*/

    for(int i=0; i<featureNumber+1; i++){
        zeroes.push_back(0);
    }


    float r;
    for(int i=0;i<(featureNumber+1)*classNumber; i++){
        r = rand()% 1000;
        r/= (float)1000;
        w.push_back(r);
    }

  /*  cout<<"weight vector:  ";
    for(int k=0;k<w.size();k++)
        cout<<w[k]<<" ";*/

}


int main()
{
    srand((unsigned)time(NULL));
    int i,j;

    cout<<"Training Phase: \n\n";
    readTrainData();
    keslerVector =  construct_KeslerVector(data);

    kesler_algorithm();

    int m = 0;
    cout<<"\nFinal weight vector:  \n";
    for(int i =0; i< classNumber; i++){
        cout<< "w"<<(i+1)<<" = [ ";
        for(int j = 0; j<featureNumber+1; j++){
           cout<< w[m]<<"  ";
           m++;
        }
        cout<<"]\n";
    }
    cout<<"\n\n";


    cout<<"Testing Phase: \n\n";
    readTestData();
    testKeslerVector =  construct_KeslerVector(testData);
    test();

    cout<<"\n\n";

    return 0;
}
