#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <ctime>
#include <vector>
#include <set>
#include <map>
#include <algorithm>
#include <iterator>

using namespace std;

#define DELIM ",./ 0123456789\n\"\'"


class document{
	public:
        int total_word;			         //total words
        double tf_idf_magnitude;         //TF-IDF vector norm(magnitude)
        string topic;			         //document topic
        vector<int> words;		         //list of word ids in this document
        map<int,double> tf_idf_weight;	 //TF-IDF weight
        map<int,int> freqs;		         //word frequency list

        document(){
            total_word=0;
            tf_idf_magnitude=0;
        }
        document(const char *str){
            topic=string(str);
            total_word=0;
            tf_idf_magnitude=0;
        }
};


map<string,int > global_dictionary;				//global dictionary
vector<document > train_docs,test_docs;			//training and test documents

int gfreq[30000];								//global frequency list of words (#ofDocuments containing word of id i --> required for IDF)
int trainingDocsNumber;							//number of training documents
int testDocsNumber;								//number of test documents
int nwords;										//number of total words




//comparison function for sorting pairs<docid, dist>
bool comp(pair<int,double> a,pair<int,double> b){
	return a.second < b.second;
}




document load_document(bool train, char *line){
	string top(line);			//topic
	getchar();					//blank line

	gets(line);					//title
	char c = getchar();			//blank line
	if(c != '\n')
        gets(line);	            //if the line wasn't blank gobble the line [wrong input]

	gets(line);					//location, date
	getchar();					//blank line

	char s[10000]={0};	        //store the story text here ( MAXSTORYSIZE = 100000)

	while(gets(line) != NULL){	//read story text
		if(strlen(line)==0)
			break;	            //end of story text
		strcat(s,line);
		strcat(s,"\n");
	}
	if(strlen(s) == 0){
        return document("");    //return empty document on empty story text
    }

	document d;

	d.topic = top;
	d.freqs.clear();

	char *p = strtok(s,DELIM);

	while(p != NULL){					//parse the story text for words
		string t(p);					//a word
		transform(t.begin(), t.end(),t.begin(), ::tolower);	//convert to lower case

		int dt = global_dictionary[t];

		if(dt == 0){					//new word in dictionary
			dt = nwords;
			global_dictionary[t] = dt;				//add the word to the dictionary
			d.words.push_back(nwords);	//add the word to the documents word-list
			d.freqs[nwords] = 1;		//first encounter of the word
			gfreq[nwords]++;			//new word
			nwords++;
		}
		else{							//existing word in global dictionary
			if(d.freqs[ dt ] == 0){		//new word in this document
				d.words.push_back(dt);	//add the word to word-list
				d.freqs[dt] = 1;
				if(train)
					gfreq[dt]++;	//????? condition lagbe keno?? sob khetre howar kotha
			}
			else{						//old word
				d.freqs[dt]++; 			//count frequency
			}
		}
		d.total_word++;							//count total words for the document
		p = strtok(NULL,DELIM);
	}
	return d;
}


void load_data(bool train){

	char line[200];
	int i,j;

	while(gets(line) != NULL){
		if(strlen(line) == 0)
			continue;	                            //skip blank line
		document d = load_document(train, line);	//load a document [train/test]
		if(d.topic.size() == 0)
			continue;	                            //skip the document if empty
		if(train){							        //add document to specified list
			train_docs.push_back(d);
			trainingDocsNumber++;
		}
		else{
			test_docs.push_back(d);
			testDocsNumber++;
		}
	}

	if(train){
		for(i=0;i<trainingDocsNumber;i++)
			sort(train_docs[i].words.begin(),train_docs[i].words.end());		//sort the word id in the documents
	}
	else{
		for(i=0;i<testDocsNumber;i++)
			sort(test_docs[i].words.begin(),test_docs[i].words.end());
	}
	if(train){		                                                            //calculate the TF-IDF weights for the words in the documents
		for(i=0;i<trainingDocsNumber;i++){
			for(j=0;j<(signed)train_docs[i].words.size();j++){
				double TF = train_docs[i].freqs[ train_docs[i].words[j] ] / (train_docs[i].total_word * 1.0);
				double IDF = log((1.0*trainingDocsNumber)/gfreq[ train_docs[i].words[j] ]);
				train_docs[i].tf_idf_weight[ train_docs[i].words[j] ] = TF*IDF;
				train_docs[i].tf_idf_magnitude += TF*IDF*TF*IDF;
			}
			train_docs[i].tf_idf_magnitude = sqrt(train_docs[i].tf_idf_magnitude);
		}
	}
    else{
        for(i=0;i<testDocsNumber;i++) {
            for(j=0;j<(signed)test_docs[i].words.size();j++){
                double TF = test_docs[i].freqs[ test_docs[i].words[j] ] / (test_docs[i].total_word * 1.0);
                double IDF = log((1.0*trainingDocsNumber)/gfreq[ test_docs[i].words[j] ]);
                test_docs[i].tf_idf_weight[ test_docs[i].words[j] ] = TF*IDF;
                test_docs[i].tf_idf_magnitude += TF*IDF*TF*IDF;
            }
            test_docs[i].tf_idf_magnitude = sqrt(test_docs[i].tf_idf_magnitude);
        }
    }
}



void hammingDistance(){
	int i,j,k,cnt[10]={0};
	register int l;
	double dist,d,a,b;
	map<int, int>::iterator it;
	vector<pair<int, double> > dm;

	for(i=0;i<testDocsNumber;i++){		                //for all testdata
		dm.clear();				                        //clear the distance measure vector
		document *testdoc = &test_docs[i];	            //current test data

		for(j=0;j<trainingDocsNumber;j++){	            //for all training data
			dist = 0.0;
			document *traindoc = &train_docs[j];	    //current training data

            vector<int> v(testdoc->words.size()+traindoc->words.size()+2);
            set_symmetric_difference(testdoc->words.begin(),testdoc->words.end(),traindoc->words.begin(),traindoc->words.end(),v.begin());  //store the words in v that appears only one of the two document aka uncommon words
            for(l=0;v[l];l++);                          //difference count
            dist = l;
            dm.push_back(make_pair(j,dist));	          //training documents with corresponding distance from current testdata
		}

		sort(dm.begin(), dm.end(), comp);		          //sort thetraining documents in ascending order of distance

        map<string, int> fr;                               //record topic frequencies of nearest neighbours
		for(k=1;k<=5;k+=2){                                //choosing values for k
		    int max = 0, p = 0;
			for(l=0;l<k;l++){                              //for k nearest neighbours in dm
                string top = train_docs[dm[l].first].topic;   //topic of the l'th nearest neighbour
			    fr[ top ]++ ;                                   //count topic frequency
			    if(fr[ top ] > max){                            //record the max
                    max = fr[ top ];
                    p = dm[l].first;
			    }
			}
			if(train_docs[p].topic == testdoc->topic){		    //matches topic!
                cnt[k]++;
            }
		}
	}


    printf("\n\nHamming Distance:\n");
    printf("------------------\n");
    for(k=1;k<=5;k+=2){
       printf("Accuracy (k=%d)    : %.2lf%%\n",k,(cnt[k]*100.0)/testDocsNumber);
    }

}





int main(int argc, char **argv)
{
    memset(gfreq,0,sizeof(gfreq));
	global_dictionary.clear();
    trainingDocsNumber = 0;
    testDocsNumber = 0;
    nwords = 1;

	time_t start,end;
	time(&start);
	//freopen("out.txt","w",stdout);

	freopen("training.data","r",stdin);
    load_data(true);

	printf("Training docs     : %d\n",trainingDocsNumber);
	printf("Total words       : %d\n",global_dictionary.size());

	freopen("test.data","r",stdin);
	load_data(false);
	//printf("test data loaded!\n");
	printf("Test docs         : %d\n",testDocsNumber);
	printf("Total words       : %d\n",global_dictionary.size());
	fflush(stdout);


    //hammingDistance();

	time(&end);
	int dif = difftime(end,start);
	printf("\nExecution time    : ");
	if(dif>=60)
        printf("%dm %ds\n",dif/60,dif%60);
    else
        printf("%ds\n",dif);

	return 0;
}








