//
// Created by 薛洪见 on 2018/3/5.
//

#ifndef CRF_XHJ_UTILS_H
#define CRF_XHJ_UTILS_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include "toml.h"
#include <lbfgs.h>

using toml::ParseResult;
using std::string;
using std::vector;
using std::set;
using std::unordered_map;
using std::pair;

using std::ifstream;
using std::ofstream;
using std::istringstream;

using std::cout;
using std::endl;


#define pp(x) do{std::stringstream ss;ss.setf(std::ios::fixed, std::ios::floatfield);ss.precision(6);ss<<x;cout<<ss.str()<<endl;}while(0)
//typedef double lbfgsfloatval_t;

struct eval_metric{
	float serr;
	float terr;
};

// for feature parse
static const size_t kMaxContextSize = 8;
static const char *BOS[kMaxContextSize] = { "_B-1", "_B-2", "_B-3", "_B-4", "_B-5", "_B-6", "_B-7", "_B-8" };
static const char *EOS[kMaxContextSize] = { "_B+1", "_B+2", "_B+3", "_B+4", "_B+5", "_B+6", "_B+7", "_B+8" };


// log(exp(x) + exp(y));
//    this can be used recursivly
// e.g., log(exp(log(exp(x) + exp(y))) + exp(z)) =
// log(exp (x) + exp(y) + exp(z))
#define MINUS_LOG_EPSILON  50
double logsumexp(double x, double y, bool is_x_zero);


// parse conf_file
ParseResult parseArgs(string cof_file_name);


// string op
vector<string> string_split(const string& s, const string &c);


template <class T>
string vec_join(vector<T>& elements, string delimiter){
	std::stringstream ss;
	size_t elems = elements.size(), last = elems - 1;
	for( size_t i = 0; i < elems; ++i ) {
		ss << elements[i];
		if( i != last )
			ss << delimiter;
	}
	return ss.str();
}

template <class T>
T stringToNum(const string& str) {
	istringstream iss(str);
	T num;
	iss >> num;
	return num;
}

#endif //CRF_XHJ_UTILS_H
