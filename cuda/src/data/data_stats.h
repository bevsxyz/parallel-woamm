#ifndef data_stats_h
#define data_stats_h

#include <vector>
#include <string>

using namespace std;

class DataStats
{
public:
    float mean;
    float median;
    float stand;
    float range[2];
    float time_avg;
    string func_name;
    vector<float> time;
    vector<float> data;

    void run();

private:
    void get_mean();
    void get_median();
    void get_stand();
    void get_range();
    void get_timeavg();
};

#endif //data_stats_h