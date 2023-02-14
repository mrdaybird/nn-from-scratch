#define MLPACK_PRINT_INFO
#define STB_IMAGE_IMPLEMENTATION
#define MLPACK_PRINT_WARN

#include <mlpack.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace mlpack;
using namespace arma;

mat linear1(mat const& data, mat const& weights, mat const& bias){
	mat res = weights*data;
	return res + as_scalar(bias);
}

mat sigmoid(mat const& m){
	return 1/(arma::exp(-m) + 1);
}

double accuracy(mat const& preds, mat const& target){
	mat preds_sig = std::move(sigmoid(preds));
//    mat res = mean(conv_to<mat>::from(conv_to<mat>::from(preds_sig > 0.5) == target), 1);
    mat res = mean(conv_to<mat>::from((preds_sig > 0.5) == target), 1);
	return as_scalar(res);
	//return mean(mean((preds_sig > 0.5) == vectorise(target)));
}

double mnist_loss(mat const& preds, mat const& targets){
	mat preds_sig = std::move(sigmoid(preds));
	mat res1 = (targets==1)%(1-preds_sig) + (targets != 1)%preds_sig;
	return as_scalar(mean(res1, 1));
}

std::pair<mat, mat> grad(mat const& x, mat const& preds, mat const& targets){
	mat preds_sig = std::move(sigmoid(preds));
	mat ps_p(preds.n_cols, preds.n_cols);
	ps_p.diag() = preds_sig%(1-preds_sig);
//	ps_p.diag() /= preds.n_cols;
	ps_p.diag() /= preds.n_cols * 0.1;

	rowvec ones_m(targets.n_cols, fill::ones);
	mat l_ps = ((targets==1)%(-ones_m)+ (targets != 1)%ones_m);

//	cout << arma::size(l_ps) << ' ' << arma::size(ps_p) << ' ' << arma::size(x.t()) << endl;
	mat res = l_ps*ps_p;
	return {res*x.t(), res*ones<mat>(x.n_cols, 1)};
}

void write_csv(string filename, mat const& m){
	ofstream file(filename);
	auto it = m.begin();
	auto end = m.end();
	int k = 1;
	for(; it != end; it++, k++){
		file << *it;
		if(k % m.n_rows == 0){
			file << '\n';
		}else{
			file << ',';
		}
	}
	file.close();
}


int main(){
	std::filesystem::path data_path("mnist_sample");
	auto three_fs = data_path/"train"/"3";
	auto seven_fs = data_path/"train"/"7";

	vector<string> threes;
	for(auto const& entry : std::filesystem::directory_iterator(three_fs)){
		std::string p = entry.path();
		threes.emplace_back(p);
	}
	vector<string> sevens;
	for(auto const& entry : std::filesystem::directory_iterator(seven_fs)){
		std::string p = entry.path();
		sevens.emplace_back(p);
	}

	auto v3_fs = data_path/"valid"/"3";
	auto v7_fs = data_path/"valid"/"7";
	
	vector<string> v3s;
	for(auto const& entry : std::filesystem::directory_iterator(v3_fs)){
		std::string p = entry.path();
		v3s.emplace_back(p);
	}
	vector<string> v7s;
	for(auto const& entry : std::filesystem::directory_iterator(v7_fs)){
		std::string p = entry.path();
		v7s.emplace_back(p);
	}	
	mat three_i;
	data::ImageInfo info;
	data::Load(threes, three_i, info, false);
	
	write_csv("three.csv", three_i);

	mat seven_i;
	data::ImageInfo info1;
	data::Load(sevens, seven_i, info1, false);

	// write_csv("seven.csv", seven_i);

	mat valid_3;
	data::ImageInfo info2;
	data::Load(v3s, valid_3, info2, false);

	// write_csv("valid_three.csv", valid_3);

	mat valid_7;
	data::ImageInfo info3;
	data::Load(v7s, valid_7, info3, false);

	// write_csv("valid_seven.csv", valid_7);

	three_i/= 255.99;
	seven_i /= 255.99;
	valid_3 /= 255.99;
	valid_7 /= 255.99;
	mat train_x = std::move(join_rows(three_i, seven_i));
	mat train_y = std::move(join_rows(ones<rowvec>(three_i.n_cols), zeros<rowvec>(seven_i.n_cols)));
	
	mat valid_x = std::move(join_rows(valid_3, valid_7));
	mat valid_y = std::move(join_rows(ones<rowvec>(valid_3.n_cols), zeros<rowvec>(valid_7.n_cols)));

	rowvec weights(28*28, fill::randn);
	vec bias(1, fill::randn);
	
	cout << arma::size(weights) << ' ' << arma::size(train_y) << endl;
	
	cout << weights*train_x.col(0) + bias << endl;
	
	double lr = 1;

	for(int i = 0; i < 30; i++){
		mat preds = linear1(train_x, weights, bias);
		//	cout << preds.head_cols(5) << endl;	
		double loss = mnist_loss(preds, train_y);

		auto [weight_grad, bias_grad] = grad(train_x, preds, train_y);
		weights -= weight_grad * lr;
		bias -= bias_grad * lr;
		
		double acc = accuracy(preds, train_y);
		cout << i << ' ' << acc << ' ' << loss << endl;

		mat valid_preds = linear1(valid_x, weights, bias);
		double valid_acc = accuracy(valid_preds, valid_y);
		cout << i << ' ' << valid_acc << endl;
	}
}


