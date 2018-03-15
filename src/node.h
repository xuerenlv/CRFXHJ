//
// Created by 薛洪见 on 2018/3/6.
//

#ifndef CRF_XHJ_NODE_H
#define CRF_XHJ_NODE_H

#include "utils.h"
#include "path.h"

class Path;

class Node{
public:
	Node():y(0),alpha(0.0),beta(0.0),cost(0.0),bestCost(0.0),prev(0){}
	
	void calcAlpha();
	void calcBeta();
	void calcCost(const lbfgsfloatval_t* alpha, lbfgsfloatval_t cost_factor);
	void calcExpectation(lbfgsfloatval_t* alpha_gradient, double Z, size_t label_size) const;
	
//	unsigned int         sent_index;
//	unsigned int         word_index;
	unsigned short int   y;       // label
	vector<int>          *fvector;// feature
	double               alpha;   // 前向
	double               beta;    // 后向
	double               cost;
	
	double               bestCost;
	Node                 *prev;
	
	vector<Path *>       lpath;
	vector<Path *>       rpath;
};

#endif //CRF_XHJ_NODE_H
