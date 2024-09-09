
#include <iostream>
#ifndef SHAPE
#define SHAPE
struct Shape
{
    int *shape;
    int ndim = 0;

    template < typename... Args>
    Shape(Args... args)
    {
        ndim = sizeof...(args);
        shape = new int[ndim]{args...};        
    }

    Shape()
    {
        shape = nullptr;
        ndim = 0;
    }

    Shape(const Shape& a)
    {
        ndim = a.ndim;
        shape = a.shape;
    }

    inline Shape clone()
    {
        Shape a;
        a.ndim = ndim;
        a.shape = new int[ndim];
        for (int i = 0; i < ndim; i++)
        {
            a.shape[i] = shape[i];
        }
        return a;
    }

    template <typename T>
    Shape(std::vector<T> a)
    {
        ndim = a.size();
        shape = new int[ndim];
        for (int i = 0; i < ndim; i++)
        {
            shape[i] = a[i];
        }
    }

    int& operator[](int i)
    {
        // if (i >= ndim && i < -int(ndim))
        // {
        //     std::cerr << "Index out of range" << std::endl;
        //     std::cerr << "Index: " << i << " Ndim: " << ndim << std::endl;
        //     throw std::runtime_error("Index out of range");
        // }
        
        return shape[i%ndim];
    }

    operator std::string()
    {
        std::string s = "[";
        for (int i = 0; i < ndim; i++)
        {
            s += std::to_string(shape[i]);
            if (i != ndim - 1)
            {
                s += ", ";
            }
        }
        s += "]";
        return s;
    }

    size_t total_size() const
    {
        size_t size = 1;
        for (size_t i = 0; i < ndim; i++)
        {
            size *= shape[i];
        }
        return size;
    }
    
};

std::ostream &operator<<(std::ostream &os, Shape a)
{
    os << a.operator std::string();
    return os;
}
#endif

// Shape a(1,2,3);