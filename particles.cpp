
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"

#define pows(x) (pow(float(x), 2))
#include "immintrin.h"

struct double2
{
    double x;
    double y;

    double2(int x, int y)
    {
        this->x = x;
        this->y = y;
    };

    double2()
    {
        this->x = 0;
        this->y = 0;
    };

    double2 operator+(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator-(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator/(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator*(double2 &other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator*(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator/(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_div_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator+(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator-(double other)
    {
        double2 out;
        _mm_storeu_pd((double *)&out, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return out;
    };

    double2 operator+=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator-=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator/=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator*=(double2 &other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_loadu_pd((double *)&other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator*=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_mul_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator/=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_div_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator+=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_add_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };

    double2 operator-=(double other)
    {
        _mm_storeu_pd((double *)this, _mm_sub_pd(_mm_set1_pd(other), _mm_loadu_pd((double *)this)));
        return *this;
    };


    double2 copy()
    {
        double2 out;
        out.x = x;
        out.y = y;
        return out;
    };

    static double2 random()
    {
        double2 out;
        out.x = (rand() % 10000) / 10000.0 - 0.5;
        out.y = (rand() % 10000) / 10000.0 - 0.5;
        return out;
    };

    double2 operator=(double2 &other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    double2 operator=(double other)
    {
        x = other;
        y = other;
        return *this;
    };

    double2 operator=(double2 other)
    {
        x = other.x;
        y = other.y;
        return *this;
    };

    Tensor toTensor()
    {
        Tensor out({2}, kFLOAT_64);
        out[0] = x;
        out[1] = y;
        return out;
    };

    // operator Tensor()
    // {
    //     Tensor out({2}, &x, kFLOAT_64);
    //     out[0] = x;
    //     out[1] = y;
    //     return out;
    // };
    
};

#define sample(i, j, feild) feild[i][j].as<double2>()

#define bleed(i, j, b, feild) (sample(i, j, feild))->b = ((sample(i, j, feild))->b / (9) + ((sample(i + 1, j, feild))->b + (sample(i - 1, j, feild))->b + (sample(i, j + 1, feild))->b + (sample(i, j - 1, feild))->b) / 9 + ((sample(i + 1, j + 1, feild))->b + (sample(i - 1, j + 1, feild))->b + (sample(i + 1, j - 1, feild))->b + (sample(i - 1, j - 1, feild))->b) / (9));

int main()
{
    const int width = 512;
    const int height = 512;
    auto repulsion = 0.05;
    auto friction = 0.00001;
    auto gravity = 0.01;
    auto warp = 0.0;

    // create the window
    sf::RenderWindow window(sf::VideoMode(width, height), "Some Funky Title");

    // create a texture
    sf::Texture texture;
    texture.create(width, height);

    // Create a pixel buffer to fill with RGBA data
    Tensor screenbuffer = Tensor({width, height, 4}, kUINT_8);
    Tensor Particles = Tensor({4096*10, 2}, kFLOAT_64);
    Particles = 0;
    Tensor TensorField = Tensor({width, height, 2}, kFLOAT_64);

    Tensor dt = Tensor{{4, 2}, kFLOAT_64};
    
   
    screenbuffer = 0;
    auto count = 0;
    TensorField = 0;

    // The colour we will fill the window with
    unsigned char red = 0;
    unsigned char blue = 255;
    auto currenttime = std::chrono::system_clock::now();

    // run the program as long as the window is open
    int frame = 0;
    while (window.isOpen())
    {
        // check all the window's events that were triggered since the last iteration of the loop
        sf::Event event;
        while (window.pollEvent(event))
        {
            // "close requested" event: we close the window
            if (event.type == sf::Event::Closed)
                window.close();
        }
        

        // clear the window with black color
        window.clear(sf::Color::Black);

        // Create RGBA value to fill screen with.
        // Increment red and decrement blue on each cycle. Leave green=0, and make opaque
        // uint32_t RGBA;
        // Stuff data into buffer
        // get mouse position in the window
        sf::Vector2i localPosition = sf::Mouse::getPosition(window);
        // screenbuffer[localPosition.y][localPosition.x][0] = 255;
        // screenbuffer[55][56][0] = 255;
        // screenbuffer[56][55][0] = 255;
        // screenbuffer[56][56][0] = 255;
        // each cell has a: pressure, velocity
        // pressure equalizes with neighbours
        // as it equalizes, it pulls velocity towards it
        // velocity is then used to move the pressure

        // TensorFieldLast = TensorFieldLast*0.5;
        // TensorField[] = a;
        screenbuffer = 0;
        bool pressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool rpressed = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        if (pressed)
        {

            *Particles[count++].as<double2>() = double2{localPosition.x, localPosition.y};
            *TensorField[localPosition.y][localPosition.x].as<double2>() = double2::random() * 0.01;
            
        }
        if (rpressed)
        {
            
        }

    
#pragma omp parallel for collapse(2) num_threads(16)
        for (int i = 0; i < Particles.shape[0]; i++)
        {

            double2 &particle = *Particles[i].as<double2>();
            if (particle.x == 0 || particle.y == 0)
            {
                continue;
            }


            auto mx = 0.0;
            auto my = 0.0;
          
            mx += TensorField[particle.y - 1][particle.x - 1].as<double2>()->x;
            mx += TensorField[particle.y + 1][particle.x - 1].as<double2>()->x;
            mx += TensorField[particle.y - 1][particle.x + 1].as<double2>()->x;
            mx += TensorField[particle.y + 1][particle.x + 1].as<double2>()->x;
            mx += TensorField[particle.y    ][particle.x - 1].as<double2>()->x;
            mx += TensorField[particle.y    ][particle.x + 1].as<double2>()->x;
            mx += TensorField[particle.y][particle.x].as<double2>()->x;

            my += TensorField[particle.y - 1][particle.x - 1].as<double2>()->y;
            my += TensorField[particle.y - 1][particle.x + 1].as<double2>()->y;
            my += TensorField[particle.y + 1][particle.x - 1].as<double2>()->y;
            my += TensorField[particle.y + 1][particle.x + 1].as<double2>()->y;
            my += TensorField[particle.y - 1][particle.x    ].as<double2>()->y;
            my += TensorField[particle.y + 1][particle.x    ].as<double2>()->y;
            my += TensorField[particle.y][particle.x].as<double2>()->y;

            TensorField[particle.y - 1][particle.x - 1].as<double2>()->x = 0;
            TensorField[particle.y + 1][particle.x - 1].as<double2>()->x = 0;
            TensorField[particle.y - 1][particle.x + 1].as<double2>()->x = 0;
            TensorField[particle.y + 1][particle.x + 1].as<double2>()->x = 0;
            TensorField[particle.y    ][particle.x - 1].as<double2>()->x = 0;
            TensorField[particle.y    ][particle.x + 1].as<double2>()->x = 0;

            TensorField[particle.y - 1][particle.x - 1].as<double2>()->y = 0;
            TensorField[particle.y + 1][particle.x - 1].as<double2>()->y = 0;
            TensorField[particle.y - 1][particle.x + 1].as<double2>()->y = 0;
            TensorField[particle.y + 1][particle.x + 1].as<double2>()->y = 0;
            TensorField[particle.y - 1][particle.x    ].as<double2>()->y = 0;
            TensorField[particle.y + 1][particle.x    ].as<double2>()->y = 0;

            TensorField[particle.y][particle.x].as<double2>()->x = 0;
            TensorField[particle.y][particle.x].as<double2>()->y = gravity;



            // mx *= 0.99;
            // my *= 0.99;

            // mx += (float(rand() % 10000) / 10000.0f - 0.5) * 0.001;
            // my += (float(rand() % 10000) / 10000.0f - 0.5) * 0.001
            // my += 0.01;

            // particle.x += mx;
            // particle.y += my;

            particle.x += mx;
            particle.y += my;

            if(particle.x < 1 || particle.x > width-2){
                mx = -mx;
                particle.x += mx;
            }
            if(particle.y < 1 || particle.y > height-2){
                my = -my;
                particle.y += my;
            }
        
            particle.x = std::max(1.0, std::min(double(width-2), particle.x));
            particle.y = std::max(1.0, std::min(double(height-2), particle.y));
            // if(TensorField[particle.y][particle.x][0].as<double2>()->x>0.1){

            //     particle.x += (float(rand()%10000)/10000.0f - 0.5)*0.1 ;
            //     particle.y += (float(rand()%10000)/10000.0f - 0.5)*0.1 ;

            //     particle.x = std::max(0.0f, std::min(float(width-1), particle.x));
            //     particle.y = std::max(0.0f, std::min(float(height-1), particle.y));
            

            auto ax = (mx / 7) * (1.0-friction);
            auto ay = (my / 7) * (1.0-friction);
            // }
            TensorField[particle.y - 1][particle.x - 1].as<double2>()->x += repulsion * -1 + ax;
            TensorField[particle.y    ][particle.x - 1].as<double2>()->x += repulsion * -1 + ax;
            TensorField[particle.y + 1][particle.x - 1].as<double2>()->x += repulsion * -1 + ax;

            TensorField[particle.y - 1][particle.x + 1].as<double2>()->x += repulsion * 1 + ax;
            TensorField[particle.y    ][particle.x + 1].as<double2>()->x += repulsion * 1 + ax;
            TensorField[particle.y + 1][particle.x + 1].as<double2>()->x += repulsion * 1 + ax;

            TensorField[particle.y - 1][particle.x - 1].as<double2>()->y += repulsion * -1 + ay;
            TensorField[particle.y - 1][particle.x    ].as<double2>()->y += repulsion * -1 + ay;
            TensorField[particle.y - 1][particle.x + 1].as<double2>()->y += repulsion * -1 + ay;

            TensorField[particle.y + 1][particle.x - 1].as<double2>()->y += repulsion * 1 + ay;
            TensorField[particle.y + 1][particle.x    ].as<double2>()->y += repulsion * 1 + ay;
            TensorField[particle.y + 1][particle.x + 1].as<double2>()->y += repulsion * 1 + ay;

            TensorField[particle.y][particle.x].as<double2>()->x += ax;
            TensorField[particle.y][particle.x].as<double2>()->y += ay;

          

       

            // density = TensorField[particle.y][particle.x].as<double2>()->x - 2;

            // mx = -mx*density;
            // my = -my*density;

            // my +=  activitylevel * (density < 2) ;

            // auto density = TensorField[particle.y][particle.x].as<double2>()->x * 0.0;
            // auto activitylevel = TensorField[particle.y][particle.x].as<double2>()->y * 0.1;

            // *screenbuffer[particle.y + 1][particle.x][0].as<uint8_t>() += density;
            // screenbuffer[particle.y + 1][particle.x][1] = 255 - activitylevel * 255;
            // screenbuffer[particle.y + 1][particle.x][2] = 0.0;
            // screenbuffer[particle.y + 1][particle.x][3] = 255;
            // *screenbuffer[particle.y][particle.x + 1][0].as<uint8_t>() += density;
            // screenbuffer[particle.y][particle.x + 1][1] = 255 - activitylevel * 255;
            // screenbuffer[particle.y][particle.x + 1][2] = 0.0;
            // screenbuffer[particle.y][particle.x + 1][3] = 255;
            // *screenbuffer[particle.y - 1][particle.x][0].as<uint8_t>() += density;
            // screenbuffer[particle.y - 1][particle.x][1] = 255 - activitylevel * 255;
            // screenbuffer[particle.y - 1][particle.x][2] = 0.0;
            // screenbuffer[particle.y - 1][particle.x][3] = 255;
            // *screenbuffer[particle.y][particle.x - 1][0].as<uint8_t>() += density;
            // screenbuffer[particle.y][particle.x - 1][1] = 255 - activitylevel * 255;
            // screenbuffer[particle.y][particle.x - 1][2] = 0.0;
            // screenbuffer[particle.y][particle.x - 1][3] = 255;
            *screenbuffer[particle.y][particle.x][0].as<uint8_t>() += 2.0;
            screenbuffer[particle.y][particle.x][1] = 255;
            screenbuffer[particle.y][particle.x][2] = 255;
            screenbuffer[particle.y][particle.x][3] = 255;
        }
        // Update screen
        texture.update((uint8_t *)screenbuffer.data);
        sf::Sprite sprite(texture);
        window.draw(sprite);

        // end the current frame
        window.display();
        auto currtime = std::chrono::system_clock::now();
        std::cout << "frames per second: " << 1000.0f / (std::chrono::duration_cast<std::chrono::milliseconds>(currtime - currenttime).count()) << "      \r";
        std::cout << std::flush;
        currenttime = currtime;
        frame++;
        // if(frame==1000)break;
    }

    return 0;
}