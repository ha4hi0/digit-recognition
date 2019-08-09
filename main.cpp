#include "mlp.hpp"

int main(void)
{
    mlp AI(300, "mlp_train.data", 0.05);
    AI.learn();
    return 0;
}
