struct int2 
{
    int x;
    int y;
    int2(int x, int y){
        this->x = x;
        this->y = y;
    };
    int2(){
        this->x = 0;
        this->y = 0;
    };
};

struct int4 
{
    int x;
    int y;
    int z;
    int w;
    int4(int x, int y, int z, int w){
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    };
    int4(){
        this->x = 0;
        this->y = 0;
        this->z = 0;
        this->w = 0;
    };
};