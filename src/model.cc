//
// Created by 薛洪见 on 2018/3/6.
//

#include "model.h"


void CRFModel::init_train(){
	dataSet = new DataSet(pr->value.get<string>("crf.location_train"), pr->value.get<string>("crf.location_temple"));
	dataSet->buildFeatures();
	dataSet->shrinkFeatures(pr->value.get<int>("crf.freq"));
	dataSet->buildGraph();
	
	feature_size = dataSet->getFeatureSize();
	model_alpha = new lbfgsfloatval_t[feature_size];
	gradient = new lbfgsfloatval_t[feature_size];
	std::fill(model_alpha, model_alpha+feature_size, 0.0);
	std::fill(gradient, gradient+feature_size, 0.0);
}


void CRFModel::init_test() {
	dataSet = new DataSet(pr->value.get<string>("crf.location_test"), pr->value.get<string>("crf.location_temple"));
	this->model_alpha = dataSet->loadModel(pr->value.get<string>("crf.location_model"));
}


void CRFModel::save_model() {
	pp("save model !");
	ofstream tof(pr->value.get<string>("crf.location_model"));
	tof.setf(std::ios::fixed, std::ios::floatfield);
	tof.precision(16);
	
	auto fea_dict = *dataSet->getFeatureIndex()->getFeatureDict();
	unordered_map<string, vector<lbfgsfloatval_t>> feas;
	
	// filter 0.0
	unsigned int id_s, id_e, l_size = dataSet->getFeatureIndex()->getlsize();
	string fea; bool all_zero;
	unsigned int fea_size = 0;
	for(const auto& pf : fea_dict){
		all_zero = true;
		fea = pf.first;
		id_s = pf.second.first;
		
		if(fea[0]=='U'){
			id_e = id_s + l_size;
		}else if(fea[0]=='B'){
			id_e = id_s + l_size * l_size;
		}else{
			pp("feature wrong !");
			exit(0);
		}
		
		for(int i = id_s;i<id_e;i++){
			if(std::abs(model_alpha[i]) > 1e-5)
				all_zero = false;
		}
		if(all_zero)
			continue;
		
		vector<lbfgsfloatval_t> v;
		for(int i = id_s;i<id_e;i++){
			v.push_back(model_alpha[i]);
		}
		feas[fea] = v;
		fea_size += v.size();
	}
	
	#define ptof(s) (tof<<s<<endl)
	ptof("tain_sentence size:\t"<<(*(dataSet->getSentences())).size());
	ptof("label         size:\t"<<dataSet->getFeatureIndex()->getlsize());
	ptof("x             size:\t"<<dataSet->get_x_size());
	ptof("feature       size:\t"<<dataSet->getFeatureSize());
	ptof("feature size filter zero:\t"<<fea_size);
	ptof("");
	
	// output label
	auto l_dict = *(dataSet->getFeatureIndex()->getLableDict());
	vector<string> l_v(l_dict.size());
	for(const auto& l : l_dict){
		l_v[l.second] = l.first;
	}
	for(int i=0;i<l_v.size();i++){
		ptof(i<<"\t"<<l_v[i]);
	}
	ptof("");
	// output feature
	id_s = 0;
	for(const auto& pf : feas){
		fea = pf.first;
		ptof(id_s<<"\t"<<fea<<"\t"<<pf.second.size());
		for (int i = 0; i < pf.second.size(); ++i) {
			ptof(id_s+i<<"\t"<<pf.second[i]);
		}
		id_s +=  pf.second.size();
	}
	ptof("");
	#undef ptof
}
