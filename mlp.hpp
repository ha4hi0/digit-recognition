#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <random>

using namespace std;

double abs_vec(vector<vector<double>> v);

class mlp
{
public:
    int width, height;
    int data_num;
    int K;
    int index;
    double ETA;
    double error;
    vector<vector<int>> train_data;
    vector<vector<int>> train_target;
    vector<vector<double>> w; // weight from input(i) to intermediate(k)
    vector<vector<double>> delta_w;
    vector<vector<double>> v; // weight from intermediate(k) to ouput(j)
    vector<vector<double>> delta_v;
    mlp(int K_, string filename, double eta);
    void read_data(string filename);
    void learn();
    void delta_reset();
    void delta_update(vector<vector<double>> dv, vector<vector<double>> dw);
    void param_update();
};

class mlp_learn
{
private:
    vector<int> input;
    vector<double> u;
    vector<double> output;
    vector<double> ni;
    vector<double> hk;
public:
    mlp_learn(mlp m);
    double sigmoid(double x);
    double d_sigmoid(double x);
    vector<vector<double>> dEdv(mlp m);
    vector<vector<double>> dEdw(mlp m);
    double mse(mlp m);
    bool check(mlp m);
};

mlp::mlp(int K_, string filename, double eta):K(K_), index(0), error(0)
{
    random_device rd;
    mt19937 mt(rd());
    normal_distribution<> dist(0.0, 0.1);

    width = 9;
    height = 9;
    ETA=eta;
    for(int k=0; k<K; k++){
        vector<double> vv;
        for(int i=0; i<width*height+1; i++){
            vv.push_back(dist(mt));
        }
        w.push_back(vv);
    }
    for(int j=0; j<10; j++){
        vector<double> vv;
        for(int k=0; k<K+1; k++){
            vv.push_back(dist(mt));
        }
        v.push_back(vv);
    }
    for(int k=0; k<K; k++){
        vector<double> vv;
        for(int j=0; j<width*height+1; j++){
            vv.push_back(0);
        }
        delta_w.push_back(vv);
    }
    for(int i=0; i<10; i++){
        vector<double> vv;
        for(int k=0; k<K+1; k++){
            vv.push_back(0);
        }
        delta_v.push_back(vv);
    }
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

void mlp::learn()
{
    int accuracy;
    int test = 0;
    int count=0;
    vector<vector<double>> dEdv;
    vector<vector<double>> dEdw;
    do{
        accuracy = 0;
        delta_reset();
        error = 0;
        for(index=0; index<100; index++){
            mlp_learn three_l(*this);
            dEdv = three_l.dEdv(*this);
            dEdw = three_l.dEdw(*this);
            delta_update(dEdv, dEdw);
            error += three_l.mse(*this);
            accuracy += (three_l.check(*this))?(1):(0);
        }
        if(count%10==0){
            cerr << "ERROR:" << error << endl;
            cerr << "accuracy:" << accuracy / 100.0 << endl;
            cerr << abs_vec(v) << endl;
            cerr << abs_vec(dEdv) << endl;
        }
        count++;
        param_update();
    }while(error>10);
}

void mlp::delta_reset()
{
    for(auto &vv : delta_v){
        for(auto &vvv : vv){
            vvv = 0;
        }
    }
    for(auto &vv : delta_w){
        for(auto &vvv : vv){
            vvv = 0;
        }
    }
}

void mlp::delta_update(vector<vector<double>> dv, vector<vector<double>> dw)
{
    for(int i=0; i<10; i++){
        for(int k=0; k<K+1; k++){
            delta_v[i][k] += dv[i][k];
        }
    }
    for(int k=0; k<K; k++){
        for(int j=0; j<height*width+1; j++){
            delta_w[k][j] += dw[k][j];
        }
    }
}

mlp_learn::mlp_learn(mlp m)
{
    double tmp;
    input.push_back(-1);
    for(auto &vv : m.train_data[m.index]){
        input.push_back(vv);
    }
    for(int k=0; k<m.K; k++){
        tmp = 0;
        for(int j=0; j<m.width*m.height+1; j++){
            tmp += m.w[k][j]*input[j];
        }
        hk.push_back(tmp);
    }
    u.push_back(-1);
    for(int i=0; i<m.K; i++){
        u.push_back(sigmoid(hk[i]));
    }
    for(int i=0; i<10; i++){
        tmp = 0;
        for(int k=0; k<m.K+1; k++){
            tmp += m.v[i][k]*u[k];
        }
        ni.push_back(tmp);
        output.push_back(sigmoid(tmp));
    }
}

vector<vector<double>> mlp_learn::dEdv(mlp m)
{
    vector<vector<double>> ret;
    for(int i=0; i<10; i++){
        vector<double> v;
        for(int k=0; k<m.K+1; k++){
            v.push_back((m.train_target[m.index][i]-output[i])*d_sigmoid(ni[i])*u[k]);
        }
        ret.push_back(v);
    }
    return ret;
}

vector<vector<double>> mlp_learn::dEdw(mlp m)
{
    vector<vector<double>> ret;
    for(int k=0; k<m.K; k++){
        vector<double> v;
        for(int j=0; j<m.height*m.width+1; j++){
            double tmp=0;
            for(int i=0; i<10; i++){
                tmp += (m.train_target[m.index][i]-output[i])*d_sigmoid(ni[i])*m.v[i][k]*d_sigmoid(hk[k])*input[j];
            }
            v.push_back(tmp);
        }
        ret.push_back(v);
    }
    return ret;
}

double mlp_learn::sigmoid(double x)
{
    return 1/(1+expl(-x));
}

double mlp_learn::d_sigmoid(double x)
{
    return expl(-x)/((1+expl(-x))*(1+expl(-x)));
}

double mlp_learn::mse(mlp m)
{
    double ret = 0;
    for(int i=0; i<10; i++){
        ret += (output[i] - m.train_target[m.index][i])*(output[i] - m.train_target[m.index][i]);
    }
    return ret/2;
}

bool mlp_learn::check(mlp m)
{
    int max_ind = 0;
    for(int i=1; i<10; i++){
        if(output[max_ind] < output[i]){
            max_ind = i;
        }
    }
    if(m.train_target[m.index][max_ind] == 1){
        return true;
    }else{
        return false;
    }
}

void mlp::param_update()
{
    for(int i=0; i<10; i++){
        for(int k=0; k<K+1; k++){
            v[i][k] += ETA*delta_v[i][k];
        }
    }
    for(int k=0; k<K; k++){
        for(int j=0; j<height*width+1; j++){
            w[k][j] += ETA*delta_w[k][j];
        }
    }
}

double abs_vec(vector<vector<double>> v)
{
    double ret = 0;
    for(auto &vv : v){
        for(auto &vvv : vv){
            ret += vvv*vvv;
        }
    }
    return sqrt(ret);
}
