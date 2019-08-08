#include "mlp.hpp"

int main(void)
{
    mlp AI(200);
    AI.read_data("mlp_train.data");
    return 0;
}
