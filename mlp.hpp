#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <random>

using namespace std;

class mlp
{
private:
    int width, height;
    int data_num;
    int K;
    vector<vector<int>> train_data;
    vector<vector<int>> train_target;
    vector<vector<int>> w; // weight from input(i) to intermediate(k)
    vector<vector<int>> v; // weight from intermediate(k) to ouput(j)
public:
    mlp(int K_);
    void read_data(string filename);
};

class mlp_learn
{
private:
    vector<int> data;
public:
    double sigmoid(double x);
};

mlp::mlp(int K_):K(K_)
{
    random_device rd;
    mt19937 mt(rd());
    normal_distribution<> dist(0.0, 1.0);

    width = 9;
    height = 9;
    for(int i=0; i<width*height; i++){
        vector<int> vv;
        for(int k=0; k<K; k++){
            vv.push_back(dist(mt));
        }
        w.push_back(vv);
    }
    for(int k=0; k<K; k++){
        vector<int> vv;
        for(int j=0; j<10; j++){
            vv.push_back(dist(mt));
        }
        v.push_back(vv);
    }
}

void mlp::read_data(string filename)
{
    string data;
    ifstream ifs(filename);
    if(!ifs){
        cerr << "Failed to open " << filename << endl;
        exit(1);
    }

    try{
        for(int i=0; i<100; i++){
            int tmp;
            vector<int> d;
            vector<int> t; 
            for(int k=0; k<width*height; k++){
                ifs >> tmp;
                d.push_back(tmp);
            }
            for(int k=0; k<10; k++){
                ifs >> tmp;
                t.push_back(tmp);
            }
            train_data.push_back(d);
            train_target.push_back(t);
        }
    }catch(...){
        cerr << "XD" << endl;
        exit(1);
    }
}

double mlp_learn::sigmoid(double x)
{
    return 1/(1+exp(-x));
}
