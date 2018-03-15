//
// Created by 薛洪见 on 2018/3/6.
//


#include "node.h"



void Node::calcAlpha() {
	alpha = 0.0;
	for (auto it = lpath.begin(); it != lpath.end(); ++it) {
		alpha = logsumexp(alpha, (*it)->lnode->alpha + (*it)->cost , (it == lpath.begin()));
	}
	alpha += cost;
}


void Node::calcBeta() {
	beta = 0.0;
	for (auto it = rpath.begin(); it != rpath.end(); ++it) {
		beta = logsumexp(beta, (*it)->rnode->beta+(*it)->cost, (it == rpath.begin()));
	}
	beta += cost;
}


void Node::calcCost(const lbfgsfloatval_t* alpha, lbfgsfloatval_t cost_factor) {
	lbfgsfloatval_t c = 0.0;
	for (auto f : *fvector) {
		c += alpha[f+y];
	}
	cost = cost_factor * c;
	
}


void Node::calcExpectation(lbfgsfloatval_t* alpha_gradient, double Z, size_t label_size) const {
	const double c = std::exp(alpha + beta - cost - Z);
	for (auto f : *fvector) {
		alpha_gradient[f + y] += c;
	}
	for (auto it = lpath.begin(); it != lpath.end(); ++it) {
		(*it)->calcExpectation(alpha_gradient, Z, label_size);
	}
}


