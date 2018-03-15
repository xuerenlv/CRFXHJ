//
// Created by 薛洪见 on 2018/3/5.
//

#ifndef CRF_XHJ_FEATURE_H
#define CRF_XHJ_FEATURE_H

#include "utils.h"

class UnRule{
public:
	UnRule(string n, vector<int> r, vector<int> l):name(n),rows(r),cols(l){};
	string toString(){return name+": ["+vec_join<int>(rows, " ")+"] ["+vec_join<int>(cols, " ")+"]";};
	
	string name;
	vector<int> rows;
	vector<int> cols;
};


class FeatureIndex{
public:
	FeatureIndex(string temple_filename){
		max_fid = 0;
		readTemple(temple_filename);
		parseUnRule();
		feature_dict = new unordered_map<string, pair<int, unsigned int>>();
		label_dict = new unordered_map<string, int>();
	};
	
	// feature 相关
	int getfid(string f);
	
	int getfsize(){ return max_fid;}
	
	int setfsize(int max_fid){ return this->max_fid = max_fid;}
	
	void shrinkFeatures(int freq, unordered_map<int, int>& old2new);
	
	unordered_map<string, pair<int, unsigned int>>* getFeatureDict(){ return  feature_dict;};
	void setFeatureDict(unordered_map<string, pair<int, unsigned int>>* feature_dict){this->feature_dict=feature_dict;};
	
	// label 相关
	void init_label_dict(set<string> label_set);
	
	int getlid(string l);
	
	size_t getlsize(){return (*label_dict).size();}
	
	unordered_map<string, int>* getLableDict(){return label_dict;};
	void setLableDict(unordered_map<string, int>* label_dict){this->label_dict=label_dict;};
	
	// 模板相关
	vector<string> un_temples;
	vector<string> bi_temples;
	vector<UnRule*> un_rules;
	
private:
	void readTemple(string filename);
	void parseUnRule();
	
	unordered_map<string, int>* label_dict;
	
	int max_fid;
	unordered_map<string, pair<int, unsigned int>>* feature_dict;
};


#endif //CRF_XHJ_FEATURE_H
