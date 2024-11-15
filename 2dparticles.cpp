
#include <SFML/Graphics.hpp>
#include <iostream>
#include <cstdint>
#include "tensor/tensor.hpp"
#include <chrono>
#include <cmath>
#include "ops/ops.h"
#include "vector/vectors.hpp"

#define repulsion 0.02
#define gravity 0.001
#define warp 0.0
#define scale 1
int size = 1;
auto numthreads = 12;
int particlecount = 1e5;
#define width 1024 * scale
#define height 1024 * scale

void draw_circle(Tensor &screenbuffer, int x, int y, int radius, int r, int g, int b, int a)
{
    for (int i = -radius; i < radius; i++)
    {
        for (int j = -radius; j < radius; j++)
        {
            if (i * i + j * j < radius * radius && x + j > 0 && x + j < screenbuffer.shape[1] && y + i > 0 && y + i < screenbuffer.shape[0])
            {

                uint64_t *pixel = screenbuffer.get<uint64_t>(y + i, x + j);
                uint8_t *pixel8 = (uint8_t *)pixel;
                pixel8[0] += r;
                pixel8[1] += g;
                pixel8[2] += b;
                pixel8[3] = 255;
            }
        }
    }
};

#define FADEOFF(io,jo) sqrt((io*io+jo*jo + 1.0))
// #define FADEOFF(io, jo) 1.0
// #define FADEOFF(io,jo) sqrt(io*io+jo*jo + 1.0)
#include <thread>

void ProcessParticles(Tensor *Particles, Tensor *TensorField, int numParticles)
    {
        while (1)
        {

            auto friction = 0.0;
            for (int i = -size; i < size + 1; i++)
            {
                for (int j = -size; j < size + 1; j++)
                {
                    friction += 1 / FADEOFF(i, j);
                }
            }
            friction = 1. / friction;
            for (int i = 0; i < numParticles; i++)
            {

                float4 &particle = *(*Particles)[i].as<float4>();
                if (particle.x == 0 || particle.y == 0)
                {
                    continue;
                }
                auto mm = particle.w;

                if (mm > 0)
                {

                    float4 momentum = {0, 0, 0, 0};

                    // #pragma omp parallel for
                    for (int io = -size; io < size + 1; io++)
                    {
                        for (int jo = -size; jo < size + 1; jo++)
                        {
                            int2 offset = {io, jo};
                            momentum += *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x);
                            *TensorField->get<float4>(particle.y + offset.y, particle.x + offset.x) *= 0.0;
                        }
                    }

                    momentum.y += gravity;

                    momentum *= mm;

                   

                    particle += momentum;

                     if (particle.x + momentum.x < 1 || particle.x + momentum.x > width - 2 || particle.y + momentum.y < 1 || particle.y + momentum.y > height - 2)
                    {

                        if (particle.y + momentum.y < 1 || particle.y + momentum.y > height - 2)
                        {
                            momentum.y *= -1.0;
                            particle.y = std::max(1.0f + size, std::min(float(height - (1.0 + size)), particle.y + momentum.y));
                        }
                        if (particle.x + momentum.x < 1 || particle.x + momentum.x > width - 2)
                        {
                            momentum.x *= -1.0;
                            particle.x = std::max(1.0f + size, std::min(float(width - (1.0 + size)), particle.x + momentum.x));
                        }
                        
                       
                    }

                    // particle.x = std::max(1.0f + size+1, std::min(float(width - (1.0 + size)), particle.x));
                    // particle.y = std::max(1.0f + size+1, std::min(float(height - (1.0 + size)), particle.y));

                    auto aa = momentum * friction;
                    // }
                    // #pragma omp parallel for
                    for (int io = -size; io < size + 1; io++)
                    {
                        for (int jo = -size; jo < size + 1; jo++)
                        {
                            int2 offset = {io, jo};
                            auto fade = FADEOFF(io, jo);
                            float4 offsetto = float4({io, jo, 0, 0}) / fade;
                            int2 poffset = {particle.x + offset.x, particle.y + offset.y};
                            float4 moffset = (offsetto * repulsion + aa) / fade;
                            *TensorField->get<float4>(poffset.y, poffset.x) += moffset;
                        }
                    }
                }
                else
                {
                    // #pragma omp parallel for
                    for (int io = -size - 2; io < size + 3; io++)
                    {
                        for (int jo = -size - 2; jo < size + 3; jo++)
                        {
                            int2 offset = {io, jo};
                            int2 poffset = {particle.x + offset.x, particle.y + offset.y};
                            *TensorField->get<float4>(poffset.y, poffset.x) *= -1.0;
                        }
                    }
                }

                //
            }
            // std::this_thread::sleep_for(std::chrono::milliseconds(1));

        }

    }

struct thread
{
    Tensor *Particles;
    Tensor *TensorField;
    int numParticles;
    std::thread t;

    thread(Tensor &TensorField, int numParticles)
    {
        this->Particles = new Tensor({numParticles, 4}, kFLOAT_32);
        this->TensorField = &TensorField;

        *Particles = 0;
        this->numParticles = numParticles;
    }

    void start()
    {
        t = std::thread(
            [Particles = this->Particles, TensorField = this->TensorField, numParticles = this->numParticles]()
            {
                
                ProcessParticles(Particles, TensorField, numParticles);
            });
        

    }

    
};

int main()
{

    // auto friction = ((1.0-0.0)/ pow(size*2 + 1,2));

    // create the window
    sf::RenderWindow window(sf::VideoMode(width / scale, height / scale), "Some Funky Title");

    // create a texture
    sf::Texture texture;
    texture.create(width, height);

    // Create a pixel buffer to fill with RGBA data
    Tensor screenbuffer = Tensor({width, height, 4}, kUINT_8);
    Tensor TensorField = Tensor({width, height, 4}, kFLOAT_32);

    std::vector<thread> threads;
   
    for (int i = 0; i < numthreads; i++)
    {
        threads.push_back(thread(TensorField, particlecount / numthreads));
        threads[i].start();
    }


    screenbuffer = 0;
    auto count = 0;
    TensorField = 0;

    // The colour we will fill the window with
    unsigned char red = 0;
    unsigned char blue = 255;
    auto currenttime = std::chrono::system_clock::now();

    // run the program as long as the window is open
    auto lastFrame = std::chrono::system_clock::now();
    auto fps = 60;
    bool ispressed = false;
    double2 startdragpos = {0, 0};
    int frame = 0;
    bool but = 0;
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
        auto show = (std::chrono::system_clock::now() - lastFrame) > std::chrono::milliseconds(1000 / fps);
        if (show)
        {
            lastFrame = std::chrono::system_clock::now();

            window.clear(sf::Color::Black);

            screenbuffer = 0;
        }

        // Create RGBA value to fill screen with.
        // Increment red and decrement blue on each cycle. Leave green=0, and make opaque
        // uint32_t RGBA;
        // Stuff data into buffer
        // get mouse position in the window
        sf::Vector2i localPosition = sf::Mouse::getPosition(window);

        // TensorFieldLast = TensorFieldLast*0.5;
        // TensorField[] = a;
        bool lpressed = sf::Mouse::isButtonPressed(sf::Mouse::Left);
        bool rpressed = sf::Mouse::isButtonPressed(sf::Mouse::Right);
        bool pressed = lpressed || rpressed;
        if (pressed && !ispressed)
        {
            startdragpos = {localPosition.x + rand() % scale, localPosition.y};
            but = lpressed;
        }

        if (!pressed && ispressed)
        {
            auto enddragpos = int2{localPosition.x, localPosition.y};
            for (int i = startdragpos.x; i < enddragpos.x; i += (size * 2) * but + 1)
            {
                for (int j = startdragpos.y; j < enddragpos.y; j += (size * 2) * but + 1)
                {
                    auto part = (*threads[count%numthreads].Particles)[(count++)/numthreads].as<float4>();
                    // *part = double2{i * scale, j*scale };
                    part->x = i * scale;
                    part->y = j * scale;
                    part->w = but ? 1 : 0;
                }
            }
        }

        ispressed = pressed;


        // Update screen
        if (show)
        {
            for (int tt = 0; tt < numthreads; tt++)
            {
                
                for (int i = 0; threads[tt].numParticles > i; i++)
                {
                    auto part = (*threads[tt].Particles)[i].as<float4>();
                    if(part->x == 0 || part->y == 0)continue;
                    draw_circle(screenbuffer, part->x/scale, part->y/scale, (size+1)/scale, 0, tt*255/numthreads,255,255);
                }
            }

            texture.update((uint8_t *)screenbuffer.data);
            sf::Sprite sprite(texture);
            window.draw(sprite);

            // end the current frame
            window.display();
            auto currtime = std::chrono::system_clock::now();
            std::cout << "frames per second(0dp): " << int(1.0 / std::chrono::duration<double>(currtime - currenttime).count()) << " : Active Particles: " << std::min(count, particlecount) << "\r";
            std::cout << std::flush;
            currenttime = currtime;
            frame++;
        }
        // if(frame==1000)break;
    }

    return 0;
}