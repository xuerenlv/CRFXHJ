//
// Created by 薛洪见 on 2018/3/5.
//

#ifndef CRF_XHJ_DATASET_H
#define CRF_XHJ_DATASET_H

#include "feature.h"
#include "node.h"
#include "path.h"
#include "utils.h"


class Sent{
public:
	Sent(){};
	
	void insert(vector<string>& w_t);
	
	void genAnswer(FeatureIndex *featureIndex, int x_size);
	
	void buildFeatures(FeatureIndex *featureIndex, vector<vector<int> >& f);
	
	int size(){return word_tags.size();};
	
	vector<vector<string> > word_tags;    // 存放每一个词的多个元素
//	std::vector <std::vector<Node *> > node_;       // 存放每一个词的每一个可能的label（一个可能的label是一个node）
//	std::vector <std::vector<double> > penalty_;
	vector<unsigned short int> answer;        // 在训练的时候存放一个词的正确label 下标
	vector<unsigned short int>* sen_result;
};


class DataSet{
public:
	DataSet(string data_filename, string temple_filename, bool train=true);
	
	void buildFeatures();
	
	void shrinkFeatures(int freq);
	
	void buildGraph();
	
	void calcCost(const lbfgsfloatval_t* alpha, lbfgsfloatval_t cost_factor);
	void calcCostBysenindex(const lbfgsfloatval_t* alpha, lbfgsfloatval_t cost_factor, size_t sen_index);
	
	void forwardbackward();
	void forwardbackwardBysenindex(size_t sen_index);
	
	void viterbi();
	void viterbiBysenindex(size_t sen_index);
	
	lbfgsfloatval_t calcGradient(lbfgsfloatval_t *gradient);
	lbfgsfloatval_t calcGradientBysenindex(lbfgsfloatval_t *gradient, size_t sen_index);
	
	struct eval_metric evalMetric();
	unsigned int eval_metricBysenindex(size_t sen_index);
	
	lbfgsfloatval_t collins(lbfgsfloatval_t* model_alpha, lbfgsfloatval_t cost_factor, lbfgsfloatval_t* gradient, size_t sen_index) ;
	
	void saveForTest(string filename);
	
	bool initNbest();
	
	int getFeatureSize(){return featureIndex->getfsize();};
	
	int get_x_size(){return x_size;};
	
	FeatureIndex* getFeatureIndex(){return featureIndex;};
	
	vector<Sent*>* getSentences(){return &sentences;};
	
	// load model; return model_alpha
	lbfgsfloatval_t* loadModel(string model_data_filename);
	
private:
//	unordered_map<string, int> voc_dic;
	vector<Sent*>   sentences;
	vector<vector<vector<int> > > feature_cache;
	vector<vector<vector<Node*> > > node_cache;
	vector<lbfgsfloatval_t >* z_cache;
//	vector<lbfgsfloatval_t >* best_cost_cache;
	FeatureIndex*    featureIndex;
	int x_size;
};


#endif //CRF_XHJ_DATASET_H
