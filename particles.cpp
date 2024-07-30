
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"

#define pows(x) (pow(float(x), 2))
#include "immintrin.h"

struct float4
{
    float x;
    float y;
    float mx;
    float my;

    float4 operator+(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_add_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 operator*(float4 &other)
    {
        float4 out;
        _mm_storeu_ps((float *)&out, _mm_mul_ps(_mm_loadu_ps((float *)&other), _mm_loadu_ps((float *)this)));
        return out;
    };

    float4 copy()
    {
        float4 out;
        out.mx = mx;
        out.my = my;
        out.x = x;
        out.y = y;
        return out;
    };
};

#define sample(i, j, feild) feild[i][j].as<float4>()

#define bleed(i, j, b, feild) (sample(i, j, feild))->b = ((sample(i, j, feild))->b / (9) + ((sample(i + 1, j, feild))->b + (sample(i - 1, j, feild))->b + (sample(i, j + 1, feild))->b + (sample(i, j - 1, feild))->b) / 9 + ((sample(i + 1, j + 1, feild))->b + (sample(i - 1, j + 1, feild))->b + (sample(i + 1, j - 1, feild))->b + (sample(i - 1, j - 1, feild))->b) / (9));

int main()
{
    const unsigned width = 512;
    const unsigned height = 512;

    // create the window
    sf::RenderWindow window(sf::VideoMode(width, height), "Some Funky Title");

    // create a texture
    sf::Texture texture;
    texture.create(width, height);

    // Create a pixel buffer to fill with RGBA data
    Tensor screenbuffer = Tensor({width, height, 4}, kUINT_8);
    Tensor Particles = Tensor({204800, 4});
    Particles = 0;
    Tensor TensorField = Tensor({width, height, 4}, kFLOAT_32);
   
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
            float4 *particle = Particles[count++].as<float4>();
            TensorField[particle->y][particle->x].as<float4>()->x -= 1;
            particle->x = localPosition.x;
            particle->y = localPosition.y;
            TensorField[particle->y][particle->x].as<float4>()->x += 1;
            particle->mx = rand() % 10000 / 10000.0f - 0.5;
            particle->my = rand() % 10000 / 10000.0f - 0.5;
        }
        if (rpressed)
        {
            
        }

#pragma omp parallel for collapse(2) num_threads(16)
        for (int i = 0; i < Particles.shape[0]; i++)
        {

            float4 *particle = Particles[i].as<float4>();
            if (particle->x == 0 || particle->y == 0)
            {
                continue;
            }

            TensorField[particle->y][particle->x].as<float4>()->x -= 1;
          
            particle->mx += TensorField[particle->y - 1][particle->x - 1].as<float4>()->mx;
            TensorField[particle->y - 1][particle->x - 1].as<float4>()->mx = 0;
            particle->mx += TensorField[particle->y][particle->x - 1].as<float4>()->mx;
            TensorField[particle->y][particle->x - 1].as<float4>()->mx = 0;
            particle->mx += TensorField[particle->y + 1][particle->x - 1].as<float4>()->mx;
            TensorField[particle->y + 1][particle->x - 1].as<float4>()->mx = 0;

            particle->mx += TensorField[particle->y - 1][particle->x + 1].as<float4>()->mx;
            TensorField[particle->y - 1][particle->x + 1].as<float4>()->mx = 0;
            particle->mx += TensorField[particle->y][particle->x + 1].as<float4>()->mx;
            TensorField[particle->y][particle->x + 1].as<float4>()->mx = 0;
            particle->mx += TensorField[particle->y + 1][particle->x + 1].as<float4>()->mx;
            TensorField[particle->y + 1][particle->x + 1].as<float4>()->mx = 0;

            particle->my += TensorField[particle->y - 1][particle->x - 1].as<float4>()->my;
            TensorField[particle->y - 1][particle->x - 1].as<float4>()->my = 0.001;
            particle->my += TensorField[particle->y - 1][particle->x].as<float4>()->my;
            TensorField[particle->y - 1][particle->x].as<float4>()->my = 0.001;
            particle->my += TensorField[particle->y - 1][particle->x + 1].as<float4>()->my;
            TensorField[particle->y - 1][particle->x + 1].as<float4>()->my = 0.001;

            particle->my += TensorField[particle->y + 1][particle->x - 1].as<float4>()->my;
            TensorField[particle->y + 1][particle->x - 1].as<float4>()->my = 0.001;
            particle->my += TensorField[particle->y + 1][particle->x].as<float4>()->my;
            TensorField[particle->y + 1][particle->x].as<float4>()->my = 0.001;
            particle->my += TensorField[particle->y + 1][particle->x + 1].as<float4>()->my;
            TensorField[particle->y + 1][particle->x + 1].as<float4>()->my = 0.001;

            particle->mx += TensorField[particle->y][particle->x].as<float4>()->mx;
            TensorField[particle->y][particle->x].as<float4>()->mx = 0;
            particle->my += TensorField[particle->y][particle->x].as<float4>()->my;
            TensorField[particle->y][particle->x].as<float4>()->my = 0.001;

            // particle->mx *= 0.99;
            // particle->my *= 0.99;

            // particle->mx += (float(rand() % 10000) / 10000.0f - 0.5) * 0.001;
            // particle->my += (float(rand() % 10000) / 10000.0f - 0.5) * 0.001
            // particle->my += 0.01;

            // particle->x += particle->mx;
            // particle->y += particle->my;

            particle->x += particle->mx;
            particle->y += particle->my;
            if(particle->x<1 || particle->x >= width-1){
                particle->mx = -particle->mx;
                particle->x += particle->mx;
            }

            if(particle->y<1 || particle->y >= height-1){
                particle->my = -particle->my;
                particle->y += particle->my;
            }
            // if(TensorField[particle->y][particle->x][0].as<float4>()->x>0.1){

            //     particle->x += (float(rand()%10000)/10000.0f - 0.5)*0.1 ;
            //     particle->y += (float(rand()%10000)/10000.0f - 0.5)*0.1 ;

            //     particle->x = std::max(0.0f, std::min(float(width-1), particle->x));
            //     particle->y = std::max(0.0f, std::min(float(height-1), particle->y));
            auto ax = particle->mx / 6 * 1.0000;
            auto ay = particle->my / 6 * 1.0000;
            // }
            TensorField[particle->y - 1][particle->x - 1].as<float4>()->mx += 0.1 * -1 + ax;
            TensorField[particle->y][particle->x - 1].as<float4>()->mx += 0.1 * -1 + ax;
            TensorField[particle->y + 1][particle->x - 1].as<float4>()->mx += 0.1 * -1 + ax;

            TensorField[particle->y - 1][particle->x + 1].as<float4>()->mx += 0.1 * 1 + ax;
            TensorField[particle->y][particle->x + 1].as<float4>()->mx += 0.1 * 1 + ax;
            TensorField[particle->y + 1][particle->x + 1].as<float4>()->mx += 0.1 * 1 + ax;

            TensorField[particle->y - 1][particle->x - 1].as<float4>()->my += 0.1 * -1 + ay;
            TensorField[particle->y - 1][particle->x].as<float4>()->my += 0.1 * -1 + ay;
            TensorField[particle->y - 1][particle->x + 1].as<float4>()->my += 0.1 * -1 + ay;

            TensorField[particle->y + 1][particle->x - 1].as<float4>()->my += 0.1 * 1 + ay;
            TensorField[particle->y + 1][particle->x].as<float4>()->my += 0.1 * 1 + ay;
            TensorField[particle->y + 1][particle->x + 1].as<float4>()->my += 0.1 * 1 + ay;

            // TensorField[particle->y][particle->x].as<float4>()->mx += ax;
            // TensorField[particle->y][particle->x].as<float4>()->my += ay;

            particle->mx = 0;
            particle->my = 0;

            // density = TensorField[particle->y][particle->x].as<float4>()->x - 2;

            // particle->mx = -particle->mx*density;
            // particle->my = -particle->my*density;

            // particle->my +=  activitylevel * (density < 2) ;

            auto density = TensorField[particle->y][particle->x].as<float4>()->x * 0.0;
            auto activitylevel = TensorField[particle->y][particle->x].as<float4>()->y * 0.1;

            // *screenbuffer[particle->y + 1][particle->x][0].as<uint8_t>() += density;
            // screenbuffer[particle->y + 1][particle->x][1] = 255 - activitylevel * 255;
            // screenbuffer[particle->y + 1][particle->x][2] = 0.0;
            // screenbuffer[particle->y + 1][particle->x][3] = 255;
            // *screenbuffer[particle->y][particle->x + 1][0].as<uint8_t>() += density;
            // screenbuffer[particle->y][particle->x + 1][1] = 255 - activitylevel * 255;
            // screenbuffer[particle->y][particle->x + 1][2] = 0.0;
            // screenbuffer[particle->y][particle->x + 1][3] = 255;
            // *screenbuffer[particle->y - 1][particle->x][0].as<uint8_t>() += density;
            // screenbuffer[particle->y - 1][particle->x][1] = 255 - activitylevel * 255;
            // screenbuffer[particle->y - 1][particle->x][2] = 0.0;
            // screenbuffer[particle->y - 1][particle->x][3] = 255;
            // *screenbuffer[particle->y][particle->x - 1][0].as<uint8_t>() += density;
            // screenbuffer[particle->y][particle->x - 1][1] = 255 - activitylevel * 255;
            // screenbuffer[particle->y][particle->x - 1][2] = 0.0;
            // screenbuffer[particle->y][particle->x - 1][3] = 255;
            *screenbuffer[particle->y][particle->x][0].as<uint8_t>() += density;
            screenbuffer[particle->y][particle->x][1] = 255 - activitylevel * 255;
            screenbuffer[particle->y][particle->x][2] = 255;
            screenbuffer[particle->y][particle->x][3] = 255;
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