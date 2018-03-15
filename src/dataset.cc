//
// Created by 薛洪见 on 2018/3/5.
//

#include "dataset.h"



void Sent::insert(vector <string> &w_t) {
	word_tags.push_back(w_t);
}


void Sent::genAnswer(FeatureIndex *featureIndex, int x_size) {
	for (auto s : word_tags) {
		answer.push_back(featureIndex->getlid(s[x_size]));
	}
}


void Sent::buildFeatures(FeatureIndex *featureIndex, vector<vector<int>> &f) {
	// unigram feature
	for (int i = 0; i < word_tags.size(); ++i) {
		vector<int> ufeature;
		for(UnRule* uru : featureIndex->un_rules){
			std::stringstream sf;
			sf<<uru->name<<":";
			
			for (int j = 0; j < uru->rows.size(); ++j) {
				int r = uru->rows[j] + i;
				int c = uru->cols[j];
				if(r<0 && -r-1<kMaxContextSize){
					sf<<BOS[-r-1];
				} else if(r>=word_tags.size() && r<word_tags.size()+kMaxContextSize){
					sf<<EOS[r-word_tags.size()];
				} else if(r>=0 && r<word_tags.size() && c>=0 && c<word_tags[r].size()){
					sf<<word_tags[r][c];
				}else{
					cout<<"unirule failed: "<<uru->toString()<<endl;
					exit(0);
				}
				if (j<uru->rows.size()-1)
					sf<<"/";
			}
//			cout<<sf.str()<<" "<<featureIndex->getfid(sf.str())<<endl;
			ufeature.push_back(featureIndex->getfid(sf.str()));
		}
		f.push_back(ufeature);
	}
	// bigram feature
	for (int i = 0; i < word_tags.size(); ++i) {
		vector<int> bfeature;
		// brule 是一个 B，所以这里不作特殊处理
		for (string brule : featureIndex->bi_temples) {
			bfeature.push_back(featureIndex->getfid(brule));
		}
		f.push_back(bfeature);
	}
}


DataSet::DataSet(string data_filename, string temple_filename, bool train) {
	featureIndex = new FeatureIndex(temple_filename);
	ifstream inf(data_filename);

	string line;
	int f_max_size = -1;
	Sent* s = new Sent();
	set<string> label_set; // 如果是训练集，后期就使用，否则不用
	while (getline(inf, line)){
		auto line_spli = string_split(line, "\t");
		if(line_spli.size()>0){
			if(f_max_size == -1){
				f_max_size = line_spli.size();
			}else if(f_max_size != line_spli.size()){
				cout<<data_filename<<" the contents of the file are inconsistent."<<endl;
				exit(0);
			}
			s->insert(line_spli);
			label_set.insert(line_spli[f_max_size-1]);
		} else{
			if(s->size()==0)
				continue;
			s->sen_result = new vector<unsigned short int>(s->size());
			sentences.push_back(s);
			s = new Sent();
		}
	}
	if(s->size() != 0) {
		s->sen_result = new vector<unsigned short int>(s->size());
		sentences.push_back(s);
	}
	
	pp("sentences num:\t"<<sentences.size());
	z_cache = new vector<lbfgsfloatval_t>(sentences.size());
	
	
	x_size = f_max_size-1;
	if(train) {   // 如果是训练集，用训练集的 label 初始化 label_dict
		featureIndex->init_label_dict(label_set);
		for(Sent* sent : sentences) {
			sent->genAnswer(featureIndex, x_size);
		}
	}
}



void DataSet::buildFeatures() {
	for (Sent* s : sentences) {
		vector<vector<int>> f;
		s->buildFeatures(featureIndex, f);
		feature_cache.push_back(f);
	}
}


void DataSet::shrinkFeatures(int freq) {
	cout<<"shink before:\t"<< featureIndex->getfsize()<<endl;
	unordered_map<int, int> old2new;
	featureIndex->shrinkFeatures(freq, old2new);
	vector<vector<vector<int> > > new_feature_cache;
	for(auto i1 : feature_cache){
		vector<vector<int> > v1;
		for(auto i2 : i1){
			vector<int> v2;
			for(auto i3 : i2){
				auto it = old2new.find(i3);
				if(it != old2new.end()){
					v2.push_back(it->second);
				}
			}
			v1.push_back(v2);
		}
		new_feature_cache.push_back(v1);
	}
	feature_cache = new_feature_cache;
	cout<<"shink after:\t"<< featureIndex->getfsize()<<endl;
}


void DataSet::buildGraph() {
	int label_size = featureIndex->getlsize();
	// construct node
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		vector<vector<Node*> > v1;
		for (size_t w = 0; w < feature_cache[sen].size()/2; ++w) {
			vector<Node*> v2;
			for(int i=0; i < label_size; i++) {
				Node *node = new Node();
				node->y = i;
				node->fvector = &feature_cache[sen][w];
				v2.push_back(node);
			}
			v1.push_back(v2);
		}
		node_cache.push_back(v1);
	}
	// construct path
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		for (size_t w = 1; w < feature_cache[sen].size()/2; ++w) {
			for (size_t j = 0; j < label_size; ++j) {
				for (size_t i = 0; i < label_size; ++i) {
					Path *p = new Path();
					p->add(node_cache[sen][w-1][j],
					       node_cache[sen][w][i]);
					p->fvector = &feature_cache[sen][w-1+feature_cache[sen].size()/2];
				}
			}
		}
	}
}


// calculate cost
void DataSet::calcCost(const lbfgsfloatval_t* alpha, lbfgsfloatval_t cost_factor) {
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		calcCostBysenindex(alpha, cost_factor, sen);
	}
}

void DataSet::calcCostBysenindex(const lbfgsfloatval_t *alpha, lbfgsfloatval_t cost_factor, size_t sen_index) {
	int label_size = featureIndex->getlsize();
	for (size_t w = 0; w < feature_cache[sen_index].size()/2; ++w) {
		for(int i=0; i < label_size; i++) {
			node_cache[sen_index][w][i]->calcCost(alpha, cost_factor);
		}
	}
	for (size_t w = 0; w < feature_cache[sen_index].size()/2; ++w) {
		for(int i=0; i < label_size; i++) {
			for(Path* p : node_cache[sen_index][w][i]->lpath){
				p->calcCost(alpha, label_size, cost_factor);
			}
		}
	}
}


void DataSet::forwardbackward() {
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		forwardbackwardBysenindex(sen);
	}
}


void DataSet::forwardbackwardBysenindex(size_t sen_index) {
	size_t label_size = featureIndex->getlsize();
	lbfgsfloatval_t z;
	for (size_t w = 0; w < node_cache[sen_index].size(); ++w) {
		for (int l = 0; l < label_size; ++l) {
			node_cache[sen_index][w][l]->calcAlpha();
		}
	}
	
	for (int w =  node_cache[sen_index].size()-1; w >= 0; --w) {
		for (int l = 0; l < label_size; ++l) {
			node_cache[sen_index][w][l]->calcBeta();
		}
	}
	
	z = 0.0;
	for (size_t j = 0; j < label_size; ++j) {
		z = logsumexp(z, node_cache[sen_index][0][j]->beta, j == 0);
	}
	(*z_cache)[sen_index] = z;
}


void DataSet::viterbi() {
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		viterbiBysenindex(sen);
	}
	
}

void DataSet::viterbiBysenindex(size_t sen_index) {
	size_t label_size = featureIndex->getlsize();
	for (size_t w = 0; w < node_cache[sen_index].size(); ++w) {
		for (size_t j = 0; j < label_size; ++j) {
			lbfgsfloatval_t bestc = -1e10;
			Node* best = 0;
			Node* node = node_cache[sen_index][w][j];
			for (auto lp : node->lpath) {
				lbfgsfloatval_t c = lp->lnode->bestCost + lp->cost + node->cost;
				if(bestc < c){
					bestc = c;
					best = lp->lnode;
				}
			}
			node->prev = best;
			node->bestCost = best?bestc:node->cost;
		}
	}
	
	lbfgsfloatval_t bestc = -1e10;
	Node* best = 0;
	size_t s = node_cache[sen_index].size()-1;
	
	for (size_t j = 0; j < label_size; ++j) {
		Node* node = node_cache[sen_index][s][j];
		if(bestc < node->bestCost){
			best = node;
			bestc = node->bestCost;
		}
	}
	
	Sent* sent = sentences[sen_index];
	for (Node *n = best; n; n = n->prev) {
		(*(sent->sen_result))[s--] = n->y;
	}
}


struct eval_metric DataSet::evalMetric() {
	struct eval_metric mm = {0.0, 0.0};
	int tag_num = 0;
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		tag_num += node_cache[sen].size();
		unsigned int t_error = eval_metricBysenindex(sen);
		mm.terr += t_error;
		if(t_error>0) mm.serr += 1.0;
	}
	mm.serr /= feature_cache.size();
	mm.terr /= tag_num;
	return mm;
}

unsigned int DataSet::eval_metricBysenindex(size_t sen_index) {
	vector<unsigned short int>& ans = sentences[sen_index]->answer;
	unsigned int error = 0;
	for (size_t w = 0; w < feature_cache[sen_index].size()/2; ++w) {
		if(ans[w] != (*sentences[sen_index]->sen_result)[w]){
			error++;
		}
	}
	return error;
}


void DataSet::saveForTest(string filename) {
	ofstream tof(filename);
	// output label
	auto l_dict = *(getFeatureIndex()->getLableDict());
	vector<string> l_v(l_dict.size());
	for(const auto& l : l_dict){
		l_v[l.second] = l.first;
	}
	#define ptof(s) (tof<<s<<endl)
	for(size_t sen = 0; sen < feature_cache.size(); ++sen) {
		for (size_t w = 0; w < feature_cache[sen].size()/2; ++w) {
			ptof(vec_join<string>((sentences[sen]->word_tags)[w],"\t")<<"\t"<<l_v[(*sentences[sen]->sen_result)[w]]);
		}
		ptof("");
	}
	#undef ptof
}

lbfgsfloatval_t DataSet::collins(lbfgsfloatval_t* model_alpha, lbfgsfloatval_t cost_factor, lbfgsfloatval_t* gradient, size_t sen_index) {
	calcCostBysenindex(model_alpha, cost_factor, sen_index);
	lbfgsfloatval_t s = 0.0;
	size_t label_size = featureIndex->getlsize();
	vector<unsigned short int> &ans = sentences[sen_index]->answer;
	vector<unsigned short int> *result = sentences[sen_index]->sen_result;
	{
		bool all_true = true;
		for (size_t i = 0; i < sentences[sen_index]->size(); ++i) {
			if (ans[i] != (*result)[i]){
				all_true = false;
				break;
			}
		}
		if(all_true) return s;
	}
	
	for (size_t i = 0; i < sentences[sen_index]->size(); ++i) {
		{
			s += node_cache[sen_index][i][ans[i]]->cost;
			for (auto f : *(node_cache[sen_index][i][ans[i]]->fvector)) {
				++gradient[f + ans[i]];
			}
			
			for(auto p : node_cache[sen_index][i][ans[i]]->lpath){
				if(p->lnode->y == ans[i-1]){
					for (auto f : *(p->fvector)) {
						++gradient[f + p->lnode->y * label_size + p->rnode->y];
					}
					s += p->cost;
					break;
				}
			}
		}
		
		{
			s += node_cache[sen_index][i][(*result)[i]]->cost;
			for (auto f : *(node_cache[sen_index][i][(*result)[i]]->fvector)) {
				++gradient[f + (*result)[i]];
			}
			
			for(auto p : node_cache[sen_index][i][(*result)[i]]->lpath){
				if(p->lnode->y == (*result)[i-1]){
					for (auto f : *(p->fvector)) {
						--gradient[f + p->lnode->y * label_size + p->rnode->y];
					}
					s -= p->cost;
					break;
				}
			}
		}
	}
	
	return -s;
}


lbfgsfloatval_t DataSet::calcGradient(lbfgsfloatval_t* gradient) {
	lbfgsfloatval_t all_cost = 0.0;
	for(size_t sen = 0; sen < node_cache.size(); ++sen) {
		all_cost += calcGradientBysenindex(gradient, sen);
	}
	return all_cost;
}

lbfgsfloatval_t DataSet::calcGradientBysenindex(lbfgsfloatval_t *gradient, size_t sen_index) {
	size_t label_size = featureIndex->getlsize();
	lbfgsfloatval_t tmp = 0.0;
	for (size_t w = 0; w < node_cache[sen_index].size(); ++w) {
		for (size_t j = 0; j < label_size; ++j) {
			node_cache[sen_index][w][j]->calcExpectation(gradient,(*z_cache)[sen_index],label_size);
		}
	}
	
	vector<unsigned short int>& ans = sentences[sen_index]->answer;
	for (size_t w = 0; w < node_cache[sen_index].size(); ++w) {
		for (auto f : *(node_cache[sen_index][w][ans[w]]->fvector)) {
			--gradient[f+ans[w]];
		}
		tmp += node_cache[sen_index][w][ans[w]]->cost;
		
		for(auto p : node_cache[sen_index][w][ans[w]]->lpath){
			if(p->lnode->y == ans[w-1]){
				for (auto f : *(p->fvector)) {
					--gradient[f + p->lnode->y * label_size + ans[w]];
				}
				tmp += p->cost;
				break;
			}
		}
	}
	return (*z_cache)[sen_index] - tmp;
}


lbfgsfloatval_t* DataSet::loadModel(string model_data_filename) {
	pp("--load model--");
	
	string line;
	ifstream inf(model_data_filename);
	
	size_t fea_size = 0;
	size_t label_size = 0;
	while (getline(inf, line)){
		auto line_spli = string_split(line, "\t");
		if(line_spli.size()==0)
			break;
		if(line_spli[0] == "label         size:")
			label_size = stringToNum<int>(line_spli[1]);
		if(line_spli[0] == "feature size filter zero:")
			fea_size = stringToNum<int>(line_spli[1]);
	}
	
	unordered_map<string, int>* label_dict = new unordered_map<string, int>();
	while (getline(inf, line)){
		auto line_spli = string_split(line, "\t");
		if(line_spli.size()==0)
			break;
		(*label_dict)[line_spli[1]] = stringToNum<int>(line_spli[0]);
	}
	
	lbfgsfloatval_t* model_alpha = new lbfgsfloatval_t[fea_size];
	unordered_map<string, pair<int, unsigned int>>* fea_dict = new unordered_map<string, pair<int, unsigned int>>();
	while (getline(inf, line)){
		auto line_spli = string_split(line, "\t");
		if(line_spli.size()==0)
			break;
		if(line_spli.size()==3){
			(*fea_dict)[line_spli[1]] = std::make_pair(stringToNum<int>(line_spli[0]), static_cast<unsigned int>(1));
		}
		if(line_spli.size()==2){
			model_alpha[stringToNum<int>(line_spli[0])] = stringToNum<float>(line_spli[1]);
		}
	}
	(*featureIndex).setFeatureDict(fea_dict);
	(*featureIndex).setLableDict(label_dict);
	for(Sent* sent : sentences) {
		sent->genAnswer(featureIndex, x_size);
	}
	return model_alpha;
}












