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


#define DELIM " 0123456789+-/&.,;:!\"\'\t\n"
#define SMOOTHFACTOR .0025

string stop_words[] = {"a","able","about","across","after","all","almost","also","am","among","an","and","any","are","as","at","be","because","been","but","by","can","cannot","could","dear","did","do","does","either","else","ever","every","for","from","get","got","had","has","have","he","her","hers","him","his","how","however","i","if","in","into","is","it","its","just","least","let","like","likely","may","me","might","most","must","my","neither","no","nor","not","of","off","often","on","only","or","other","our","own","rather","said","say","says","she","should","since","so","some","than","that","the","their","them","then","there","these","they","this","tis","to","too","twas","us","wants","was","we","were","what","when","where","which","while","who","whom","why","will","with","would","yet","you","your","ain't","aren't","can't","could've","couldn't","didn't","doesn't","don't","hasn't","he'd","he'll","he's","how'd","how'll","how's","i'd","i'll","i'm","i've","isn't","it's","might've","mightn't","must've","mustn't","shan't","she'd","she'll","she's","should've","shouldn't","that'll","that's","there's","they'd","they'll","they're","they've","wasn't","we'd","we'll","we're","weren't","what'd","what's","when'd","when'll","when's","where'd","where'll","where's","who'd","who'll","who's","why'd","why'll","why's","won't","would've","wouldn't","you'd","you'll","you're","you've"};


class document{
	public:
        int total_word;
        double tf_idf_magnitude;
        string topic;
        vector<int> words;
        map<int,double> tf_idf_weight;
        map<int,int> freqs;

        document(){
            total_word=0;
            tf_idf_magnitude=0;
        }
        document(const char *str){
            topic=string(str);
            total_word=0;
            tf_idf_magnitude=0;
        }

        int binarySearch(int key){
            int l=0, r=words.size()-1,m;
            while(l<=r){
                m = (l+r)/2;
                if(words[m] > key){
                    r = m-1;
                }else if(words[m] < key){
                    l = m+1;
                }else{
                    return m;
                }
            }
            return -1;
        }
};


map<string,int > global_dictionary;
vector<document > train_docs,test_docs;
vector<int> randtest;

map<string, int> topic_list; 		//topic list
vector<int> topic_document[200]; 	//each vector consist of document id of same topic
int vocubulary_plus_n[200];			//Vocabulary(#of distinct words in the file)+count(sum of #words contained in each of doc )
int words_in_topic[30000][200];		//number of occurrences of word i in topic j



int wordContainingDocsfreq[30000];					//#of training Documents containing word of id i --> (required for IDF)
int trainingDocsNumber;
int testDocsNumber;
int nwords;											//word count
int stopwordcount;
int ntopics;
int train_vocabulary;
//double SMOOTHFACTOR;

//comparison function for sorting pairs<docid, dist>
bool comp(pair<int,double> a,pair<int,double> b){
	return a.second < b.second;
}

bool isStopWord(string s){
    int stop_size = sizeof(stop_words)/sizeof(stop_words[0]);
    bool flag = false;
    for(int i = 0; i< stop_size; i++){
        if(stop_words[i] == s){
            flag = true;
            break;
        }
    }
    return flag;
}


document readDocument(bool train, char *line){
	string top(line);			//topic
	getchar();					//blank line

	gets(line);					//title
	char c = getchar();			//blank line
	if(c != '\n')
        gets(line);	            //if the line wasn't blank gobble the line [wrong input]

	gets(line);					//location, date
	getchar();					//blank line

	char s[10000]={0};

	while(gets(line) != NULL){
		if(strlen(line)==0)
			break;
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

	while(p != NULL){
		string t(p);
		transform(t.begin(), t.end(),t.begin(), ::tolower);
        if(isStopWord(t)){
            p = strtok(NULL,DELIM);
            stopwordcount++;
            continue;
        }

		int dt = global_dictionary[t];

		if(dt == 0){					//new word in dictionary
			dt = nwords;
			global_dictionary[t] = dt;
			d.words.push_back(nwords);
			d.freqs[nwords] = 1;
			wordContainingDocsfreq[nwords]++;
			nwords++;
		}
		else{							//existing word in global dictionary
			if(d.freqs[ dt ] == 0){		//new word in this document
				d.words.push_back(dt);
				d.freqs[dt] = 1;
				if(train)
					wordContainingDocsfreq[dt]++;
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


void readData(bool train){

	char line[200];
	int i,j;

	while(gets(line) != NULL){
		if(strlen(line) == 0)
			continue;	                            //skip blank line
		document d = readDocument(train, line);
		if(d.topic.size() == 0)
			continue;	                            //skip the document if empty

		if(train){
            if(topic_list.find(d.topic) != topic_list.end()){	//existing topic
                topic_document[topic_list[d.topic]].push_back(train_docs.size());	//documents(id) of same topic are pushed into same vector
            }else{
                topic_list[d.topic] = ntopics;
                topic_document[ntopics].push_back(train_docs.size());
                ntopics++;
            }
        }

		if(train){
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
				double IDF = log((1.0*trainingDocsNumber)/wordContainingDocsfreq[ train_docs[i].words[j] ]);
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
                double IDF = log((1.0*trainingDocsNumber)/wordContainingDocsfreq[ test_docs[i].words[j] ]);
                test_docs[i].tf_idf_weight[ test_docs[i].words[j] ] = TF*IDF;
                test_docs[i].tf_idf_magnitude += TF*IDF*TF*IDF;
            }
            test_docs[i].tf_idf_magnitude = sqrt(test_docs[i].tf_idf_magnitude);
        }
    }
}

vector<pair<int, double> > dm;

void hammingDistance(){
	int i,j,k,cnt[10]={0};
	register int l;
	double dist;

	for(i=0;i<testDocsNumber;i++){
		dm.clear();
		document *testdoc = &test_docs[i];

		for(j=0;j<trainingDocsNumber;j++){
			dist = 0.0;
			document *traindoc = &train_docs[j];

            vector<int> v(testdoc->words.size()+traindoc->words.size()+2);
            set_symmetric_difference(testdoc->words.begin(),testdoc->words.end(),traindoc->words.begin(),traindoc->words.end(),v.begin());  //store the words in v that appears only one of the two document aka uncommon words
            for(l=0;v[l];l++);                          //difference count
            dist = l;
            dm.push_back(make_pair(j,dist));
		}

		sort(dm.begin(), dm.end(), comp);

        map<string, int> fr;                               //record topic frequencies of nearest neighbours
		for(k=1;k<=5;k+=2){
		    int max = 0, p = 0;
			for(l=0;l<k;l++){                              	//for k nearest neighbours in dm
                string top = train_docs[dm[l].first].topic;
			    fr[ top ]++ ;
			    if(fr[ top ] > max){
                    max = fr[ top ];
                    p = dm[l].first;
			    }
			}
			if(train_docs[p].topic == testdoc->topic){
                cnt[k]++;
            }
		}
	}


    printf("\n\nHamming Distance:\n\n");
    for(k=1;k<=5;k+=2){
       printf("When k=%d   Accuracy : %.2lf%%\n",k,(cnt[k]*100.0)/testDocsNumber);
    }
}



void euclideanDistance(){
	int i,j,k,cnt[10]={0};
	register int l;
	double dist,d,a,b;
    map<int, int>::iterator it;

	for(i=0;i<testDocsNumber;i++){
		dm.clear();
		document *testdoc = &test_docs[i];

		for(j=0;j<trainingDocsNumber;j++){
			dist = 0.0;
			document *traindoc = &train_docs[j];

            vector<int> v(traindoc->words.size()+testdoc->words.size()+2);
            set_union(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin()); //store the words in v that appears any of the two document
            for(l=0;v[l];l++){
                a=b=0;
                it = testdoc->freqs.find( v[l] );
                if(it!=testdoc->freqs.end())
                    a = it->second;

                it = traindoc->freqs.find( v[l] );
                if(it!=traindoc->freqs.end())
                    b = it->second;

                d = a-b;
                dist += d*d;
            }
            dist = sqrt(dist);

            dm.push_back(make_pair(j,dist));	          //training documents with corresponding distance from current testdata
		}

		sort(dm.begin(), dm.end(), comp);		          //sort thetraining documents in ascending order of distance

        map<string, int> fr;
		for(k=1;k<=5;k+=2){
		    int max = 0, p = 0;
			for(l=0;l<k;l++){
                string top = train_docs[dm[l].first].topic;   //topic of the l'th nearest neighbour
			    fr[ top ]++ ;
			    if(fr[ top ] > max){
                    max = fr[ top ];
                    p = dm[l].first;
			    }
			}
			if(train_docs[p].topic == testdoc->topic){
                cnt[k]++;
            }
		}
	}


    printf("\n\nEuclidean Distance:\n\n");
    for(k=1;k<=5;k+=2){
       printf("When k=%d   Accuracy : %.2lf%%\n",k,(cnt[k]*100.0)/testDocsNumber);
    }
}


double cosineSimilarity(){
	int i,j,k,cnt=0;
	register int l;
	double dist;

	for(i=0;i<randtest.size();i++){
		dm.clear();				                        //clear the distance measure vector
		document *testdoc = &test_docs[randtest[i]];

		for(j=0;j<trainingDocsNumber;j++){
			dist = 0.0;
			document *traindoc = &train_docs[j];

            int s;
            if(traindoc->words.size() > testdoc->words.size())
                s = testdoc->words.size();
            else
                s = traindoc->words.size();

            vector<int> v(s);
            set_intersection(traindoc->words.begin(),traindoc->words.end(),testdoc->words.begin(),testdoc->words.end(),v.begin());
            for(l=0;v[l];l++)
                dist += traindoc->tf_idf_weight[ v[l] ] * testdoc->tf_idf_weight[ v[l] ];
            dist /= traindoc->tf_idf_magnitude;
            dist /= testdoc->tf_idf_magnitude;      //now dist = cos(theta)
            dist = acos(dist);

            dm.push_back(make_pair(j,dist));	          //training documents with corresponding distance from current testdata
		}

		sort(dm.begin(), dm.end(), comp);		          //sort thetraining documents in ascending order of distance

        map<string, int> fr;
		//for(k=5;k<=5;k+=2){
		    int max = 0, p = 0;
			for(l=0;l<5;l++){
                string top = train_docs[dm[l].first].topic;   //topic of the l'th nearest neighbour
			    fr[ top ]++ ;                                   //count topic frequency
			    if(fr[ top ] > max){                            //record the max
                    max = fr[ top ];
                    p = dm[l].first;
			    }
			}
			if(train_docs[p].topic == testdoc->topic){
                cnt++;
            }
		//}
	}


    return (cnt*100.0)/randtest.size();
}



void priorCalculation(){
    int i,j,k;

    for(i=1;i<ntopics;i++){					//Vocabulary(#of distinct words in the file)+count(sum of #words contained in each of doc )
        int count = 0;
        for(j=0;j<topic_document[i].size();j++){	//for all the documents that are of topic i
            document *d = &train_docs[ topic_document[i][j] ];
            for(k=0;k < d->words.size();k++){
                count += d->freqs[ d->words[k] ];
            }
        }
        vocubulary_plus_n[i] = count + train_vocabulary*SMOOTHFACTOR;
    }

    int sum,sz;									//number of occurrences of word i in topic j
    for(i=1;i<global_dictionary.size();i++){	//for all words in the dictionary
        for(j=0;j<ntopics;j++){					//j->topic id
            sum = 0;
            for(k=0;k<topic_document[j].size();k++){	//protita class e jabo jader moddhe "i" word id er word ta ase
                document *d = &train_docs[ topic_document[j][k] ];
                int ind = d->binarySearch( i );
                if(ind != -1) sum += d->freqs[ d->words[ind] ];
            }
            words_in_topic[i][j] = sum;
        }
    }
}


double naiveBayes(){
    int i,tp,w,cnt,naive_topic,matched=0;
    double prior_probability,log_likelihood,posterior_probability,Pmax;

    for(i=0;i<randtest.size();i++){
        document td = test_docs[ randtest[i] ];
        Pmax = -9999999.;
        for(tp=1;tp<ntopics;tp++){		//for each class (topic) calculate posterior probability of the document
            log_likelihood=0;
            for(w=0;w<td.words.size();w++){
               cnt = words_in_topic[ td.words[w] ][tp];	//#of 'w' word in document whose topic is tp
               log_likelihood += log((cnt+SMOOTHFACTOR)/vocubulary_plus_n[tp]);
            }
            prior_probability = topic_document[tp].size();
            prior_probability /= trainingDocsNumber;	//prior probability of tp class(topic)

            posterior_probability = log(prior_probability) + log_likelihood;

            if(posterior_probability > Pmax){
                Pmax = posterior_probability;
                naive_topic = tp;
            }
        }
        if(topic_list[td.topic] == naive_topic){
            matched++;
        }
    }
    return (matched*100.)/randtest.size();
}






int main(int argc, char **argv)
{
    memset(wordContainingDocsfreq,0,sizeof(wordContainingDocsfreq));
	global_dictionary.clear();
    trainingDocsNumber = 0;
    testDocsNumber = 0;
    nwords = 1;
    stopwordcount = 0;
    ntopics = 1;
    //double smooth[] = {1.0,.5,.1,.05,.035,.025,.01,.005,.0025,.002,.001,.0005,.0001,.00001};
	//freopen("out.txt","w",stdout);

	freopen("training.data","r",stdin);
    readData(true);
    train_vocabulary = nwords;

	printf("Training docs     : %d\n",trainingDocsNumber);
	printf("Total words       : %d\n",global_dictionary.size());

	freopen("test.data","r",stdin);
	readData(false);

	printf("Test docs         : %d\n",testDocsNumber);
	printf("Total words       : %d\n",global_dictionary.size());
	//printf("Stop words       : %d\n",stopwordcount);
	fflush(stdout);


   // hammingDistance();
   // euclideanDistance();
   //cosineSimilarity();


    priorCalculation();

  /*  randtest.clear();
    while(true){
        int n = rand()%testDocsNumber;
        randtest.push_back(n);
        if(randtest.size() == 50) break;
    }
    for(int i=0;i<15;i++){
        SMOOTHFACTOR = smooth[i];
        double d = naiveBayes();
        cout<<"When Smooth Factor: "<<SMOOTHFACTOR<<" :  Accuracy   "<<d<<"\n";
    }

*/


    double accuracy[2][52], t_statistics[52];

    t_statistics[0] = accuracy[0][0] = accuracy[1][0] = 0.;
    //int run = 5;
    for(int i=1;i<=50;i++){

        randtest.clear();
        while(true){
            int n = rand()%testDocsNumber;
            randtest.push_back(n);
            if(randtest.size() == 50) break;
        }

        accuracy[0][i] = cosineSimilarity();
        accuracy[1][i] = naiveBayes();

        accuracy[0][0] += accuracy[0][i];
        accuracy[1][0] += accuracy[1][i];

        t_statistics[i] = accuracy[1][i] - accuracy[0][i];
        t_statistics[0] += t_statistics[i];
    }
    accuracy[0][0] = accuracy[0][0]/50;
    accuracy[1][0] = accuracy[1][0]/50;
    t_statistics[0] = t_statistics[0]/50;

	printf("\n\nMean:\n\n");
    printf("Cosine Similarity (K=5): %.2lf%%\n", accuracy[0][0]);
    printf("Naive Bayes: %.2lf%%\n\n", accuracy[1][0]);

    double standard_deviation[2]={0};
    double tstat_std_deviation=0.;
    for(int i=1;i<=50;i++){
        standard_deviation[0] += (accuracy[0][i]-accuracy[0][0])*(accuracy[0][i]-accuracy[0][0]);
        standard_deviation[1] += (accuracy[1][i]-accuracy[1][0])*(accuracy[1][i]-accuracy[1][0]);
        tstat_std_deviation += (t_statistics[i] - t_statistics[0])*(t_statistics[i] - t_statistics[0]);
    }
    standard_deviation[0] = sqrt(standard_deviation[0]/50);
    standard_deviation[1] = sqrt(standard_deviation[1]/50);
    tstat_std_deviation = sqrt(tstat_std_deviation/(50*(50-1)));

	printf("standard Deviation:\n\n");
    printf("Cosine Similarity (K=5): %.2lf%%\n", standard_deviation[0]);
    printf("Naive Bayes: %.2lf%%\n\n", standard_deviation[1]);

    printf("t-statistics(at significant level of 0.005): %.2lf%% %.2lf%%\n", t_statistics[0]-2.680*tstat_std_deviation, t_statistics[0]+2.680*tstat_std_deviation);
    printf("t-statistics(at significant level of 0.01): %.2lf%% %.2lf%%\n", t_statistics[0]-2.405*tstat_std_deviation, t_statistics[0]+2.405*tstat_std_deviation);
    printf("t-statistics(at significant level of 0.05): %.2lf%% %.2lf%%\n", t_statistics[0]-1.677*tstat_std_deviation, t_statistics[0]+1.677*tstat_std_deviation);



	return 0;
}








