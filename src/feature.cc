//
// Created by 薛洪见 on 2018/3/5.
//

#include "feature.h"


void FeatureIndex::readTemple(string filename) {
	ifstream inf(filename);
	string line;
	while (getline(inf, line)){
		if (!line[0] || line[0] == '#') {
			continue;
		}
		if (line[0] == 'U') {
			un_temples.push_back(line);
		} else if (line[0] == 'B') {
			bi_temples.push_back(line);
		} else {
			cout<<"temple unknown type: "<<line<<" "<<filename<<endl;
			exit(0);
		}
	}
}


void FeatureIndex::parseUnRule() {
	for(string unl : un_temples){
		vector<string> unls = string_split(unl, ":");
		if(unls.size() != 2){
			cout<<"un_temple name is not well-formed: "<<unl<<endl;
			exit(0);
		}
		vector<int> rols, cols;
		for(string ind : string_split(unls[1], "/")){
			vector<string> indexs = string_split(ind.substr(3, ind.find("]")-3), ",");
			if(unls.size() != 2){
				cout<<"un_temple indexs is not well-formed: "<<ind<<endl;
				exit(0);
			}
			rols.push_back(stringToNum<int>(indexs[0]));
			cols.push_back(stringToNum<int>(indexs[1]));
		}
		un_rules.push_back(new UnRule(unls[0], rols, cols));
	}
}


int FeatureIndex::getfid(string f) {
	auto it = (*feature_dict).find(f);
	if(it==(*feature_dict).end()){
		int index = max_fid;
		(*feature_dict)[f] = std::make_pair(index, static_cast<unsigned int>(1));
		if(f[0] == 'U'){
			max_fid += getlsize();
		} else if(f[0] == 'B'){
			max_fid += getlsize()*getlsize();
		} else{
			cout<<"feature is not appropriate!"<<f<<endl;
			exit(0);
		}
		return index;
	}else{
		it->second.second++;
		return it->second.first;
	}
}

void FeatureIndex::shrinkFeatures(int freq, unordered_map<int, int> &old2new) {
	if(freq<=1)
		return;
	int new_max_fid = 0;
	for (auto it = (*feature_dict).begin(); it != (*feature_dict).end();) {
		const string& key = it->first;
		if (it->second.second >= freq) {
			old2new.insert(std::make_pair(it->second.first, new_max_fid));
			it->second.first = new_max_fid;
			new_max_fid += (key[0] == 'U' ? getlsize() : getlsize()*getlsize());
			it++;
		} else {
			(*feature_dict).erase(it++);
		}
	}
	max_fid = new_max_fid;
}

void FeatureIndex::init_label_dict(set <string> label_set) {
	int index = 0;
	for (string l : label_set) {
		(*label_dict)[l] = index++;
	}
}


int FeatureIndex::getlid(string l) {
	auto it = (*label_dict).find(l);
	if (it == (*label_dict).end()) {
		return -1;
	} else {
		return it->second;
		
	}
}


