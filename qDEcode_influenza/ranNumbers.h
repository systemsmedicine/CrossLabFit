/*=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=- RANDOM NUMBERS STRUCTURES =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-*/
// These generators were taken from "Numerical Recipes: The Art of Scientific Computing, 3rd Ed"
//========================================= UNIFORM ============================================//
// This is a recommend uniform random generator, its period is ~ 3.138 x 10^57.
struct Ran
{
        unsigned long long u,v,w;
        // Call with any integer seed (exept value of v below)
        Ran(unsigned long long j) : v(4101842887655102017LL), w(1)
        {
                u = j^v; int64();
                v = u; int64();
                w = v; int64();
        }
        // Return 64-bit random integer
        inline unsigned long long int64()
        {
                u = u * 2862933555777941757LL + 7046029254386353087LL;
                v ^= v >> 17; v ^= v << 31; v ^= v >> 8;
                w = 4294957665U*(w & 0xffffffff) + (w >> 32);
                unsigned long long x = u ^ (u << 21); x ^= x >> 35; x ^= x << 4;
                return (x + v) ^ w;
        }
        // Return random double-precision floating value in the range from 0 to 1
        inline double doub() { return 5.42101086242752217E-20 * int64(); } // multiply by 1/INT_MAX
        // Return 32-bit random integer
        inline unsigned int int32() { return (unsigned int)int64(); }
};
//======================================== NORMAL ===========================================//
struct Normaldev : Ran {
        double mu,sig;
        Normaldev(double mmu, double ssig, unsigned long long i) : Ran(i), mu(mmu), sig(ssig){}
        double dev() {
                double u,v,x,y,q;
                do {
                        u = doub();
                        v = 1.7156*(doub()-0.5);
                        x = u - 0.449871;
                        y = abs(v) + 0.386595;
                        q = x*x + y*(0.19600*y-0.25472*x);
                } while (q > 0.27597 && (q > 0.27846 || v*v > -4.*log(u)*u*u));
                return mu + sig*v/u;
        }
};
//======================================== GAMMA ===========================================//
struct Gammadev : Normaldev {
        double alph, oalph, bet;
        double a1, a2;
        Gammadev (double aalph, double bbet, unsigned long long i)
        : Normaldev(0.0,1.0,i), alph(aalph), oalph(aalph), bet(bbet) {
                if (alph <= 0.0) throw("Bad alpha in Gammadev");
                if (alph < 1.0) alph += 1.0;
                a1 = alph - 1.0/3.0;
                a2 = 1.0/sqrt(9.0*a1);
        }
        double dev() { 
                double u, v, x;
                do {
                        do {
                                x = Normaldev::dev();
                                v = 1.0 + a2*x;
                        } while (v <= 0.0);
                        v = v*v*v;
                        u = doub();
                } while (u > 1.0 - 0.331*x*x*x*x && log(u) > 0.5*x*x + a1*(1.0 - v + log(v)));
                if (alph == oalph) return a1*v/bet;
                else {
                        do u = doub(); while (u == 0.0);
                        return pow(u,1.0/oalph)*a1*v/bet;
                }
        }
};
