//
// Created by 薛洪见 on 2018/3/6.
//

#ifndef CRF_XHJ_PATH_H
#define CRF_XHJ_PATH_H

#include "utils.h"
#include "node.h"

class Node;

class Path{
public:
	Path():lnode(0),rnode(0),cost(0.0){}
	void calcCost(const lbfgsfloatval_t* alpha, int label_size, float cost_factor);
	void calcExpectation(lbfgsfloatval_t* alpha_gradient, double Z, size_t label_size) const;
	void add(Node *_lnode, Node *_rnode) ;
	
	Node* lnode;
	Node* rnode;
	double cost;
	
	vector<int> *fvector;
};

#endif //CRF_XHJ_PATH_H
