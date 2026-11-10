//! Market microstructrure simulation for testing and benchmarking the POV algo engine
//! 
//! Generates one synthetic trading day as a sequence of discretized time buckets
//! with: (1) a U-shaped intraday volume profile (the well-documented empirical
//! pattern of heavy trading at the open/close, thin at midday), (2) a Geometric
//! Brownian Motion (GBM) model for the price process/path, and (3) a square-root-law
//! price impact model for our own child orders.


use rand::Rng;
use rand_distr::{Distribution, LogNormal, Normal};

#[derive(Debug, Clone, Copy)]
pub struct MarketBucket {
    pub t: f64,          // time in [0, 1) through the session (0 = open, 1 = close)-- t is the bucket's midpoint time
    pub volume: f64,     // market volume in this bucket (EXCLUDING our own prints)
    pub vwap: f64,       // volume-weighted average price of the bucket (EXCLUDING our own prints i.e. pre-impact)
    pub mid_price: f64,  // reference mid-price at bucket close
}

/*
    volume: is "realized trade volume"  in the bucket. volume is NOT known until the bucket closes, and hence is not availble
    to the algo engine until the bucket closes. That is why we use an EWMA forecast instead of this value when deciding the
    current bucket's order. 
    vwap: volume weighted average price of the bucket excluding our own trades. This is defined after the bucket closes. We can 
    simply prove this mathematically: Let the price be p_0 = a and p_T = b where the bucket starts at t = 0 and ends at t= T.
    Let the price now be governed by p_t = a + \sigma * W_t where W_t in [0 , T] is a standard Brownian motion. Since (W_t, W_T)
    is bivariate normal with Cov(W_t , W_T) = t and Var(W_t) = t, Var(W_T) = T, we have E[W_t | W_T] = (t/T) * W_T. Hence, 
    E[p_t | p_T = b] = a + sigma * (t/T) * W_T = a + (t/T) * (b - a). With uniform volume, i.e. v_t = v for all t in [0 , T], we
    have vwap = E[(1/T) * \int_0^T p_t dt | p_0 = a , p_b = T] = (1/T) * \int_0^T (a + (t/T) * (b - a)) dt = 
    = (1/T) * [aT + (b - a) * T/2] = (a + b)/2.
*/

#[derive(Debug, Clone)]
pub struct MarketSimulator {
    pub adv: f64,           // average daily volume (ADV) in shares
    pub n_buckets: usize,   // number of buckets in the session
    pub sigma_daily: f64,   // daily volatility in price returns (e.g. 0.02 = 2% daily volatility)
    pub s0: f64,            // initial price at the open
    pub permanent_impact_coeff: f64,
    pub temporary_impact_coeff: f64,
    pub volume_noise_cv: f64,            // This needs to  be calibrated from previous market data
}
// 0.25 -> -0.25 -> 0.6 + 3.2 + 0.0625 = 0.8  int  0.6 + 3.2/3
impl MarketSimulator {
    /// U-shaped intraday volume weight at normalized time t in [0 , 1]
    /// Based on classif parabola-plus-floor approximation of the empirical
    /// U-shaped volume curve (heavy at open/close, light in the middle)
    fn intraday_weight(t: f64) -> f64 {
        let centered = t - 0.5;
        0.6 + 3.2 * centered * centered

    }

    /*
        There is a different method to find intraday_weight at any given time t:
        1. Fix the bucket grid, e.g. 5-minute bars, same curoffs every day (9:30am-9:35am, 9:35am-9:40am, ... 03:55pm-04:00pm)
        2. Lookback window: pick the past N trading days (e.g. 20-60 days) for that specific stock
        3. Per-day Normalization: For each day d and bucket i, compute f(d , i) = Volume(d , i) / TotalDailyVolume(d) i.e. each day's total volume is normalized to 1
           This removes the day-to-day average daily volume (ADV)-level differences 
        4. Average of volume across days: weight(i) = \mean_d f(d , i) (sometimes median or recency EWMA across days to adapt faster to regime changes)
        5. Re-normalize so that we have \sum_i wight(i) = 1
        6. Use it: expected_volume_today(i) = forecast_ADV_today * weight(i)
    */

    pub fn generate_day(&self, rng: &mut impl Rng) -> Vec<MarketBucket> {  // Recall that rng: &mut impl Rng means "rng: T where T is any type implementing the Trait Rng"
                                                                           // Notice that this is static, not dynamic. It is fixed at compile time meaning the compiler 
                                                                           // makes a separate copy of the function for each concrete type used (monomorphization). That
                                                                           // makes it fast, with no runtime lookup, unline &mut dyn RngCore or 
        let n = self.n_buckets;
        let dt = 1.0 / n as f64;

        // Normalize the U-shaped weights so they sum to n (i.e. average weight 1).
        let raw_weights: Vec<f64> = (0..n)
            .map(|i| Self::intraday_weight((i as f64 + 0.5) / n as f64))
            .collect();
        let sum_w: f64 = raw_weights.iter().sum();
        // Note that *as* binds tighter than * and / so: w * n as f64 / sum_w  ==  (w * (n as f64)) / sum_w
        let norm_weights: Vec<f64> = raw_weights.iter().map(|w| w * n as f64 / sum_w).collect();

        let avg_bucket_vol = self.adv / n as f64;           // self.adv is the average daily volume - in practice you use trimmed mean or EWMA
        let sigma_bucket = self.sigma_daily * dt.sqrt();    // daily return R_day = ln(P_clsoe / P_open) ~ N(0 , \sigma_day^2) (GBM) and R_day = R_1 + ... + R_n with dt = 1/sqrt(n). 

        let price_shock = Normal::new(0.0 , sigma_bucket).unwrap();     // Normal::new returns a Result<Normal<f64>, Error>, not the distribution directly.
                                                                        // It fails with Err if the std dev is negative or non-finite (NaN/∞). On success,
                                                                        // Ok holds a Normal object, which is just the parameters plus a .sample(rng) method.

        let mut price = self.s0;
        let mut buckets = Vec::with_capacity(n);
        // pub fn Vec::with_capacity(capacity: usize) -> Vec<T> : Constructs a new, empty Vec<T> with at least the specified capacity.

        // Run a for loop on buckets of the day
        for i in 0..n {
            // Volume: lognormal noise around the U-shaped deterministic profile                  Where does this come from???
            let mean_vol = avg_bucket_vol * norm_weights[i];

            // cv is the coefficient of variation and equals sigma/mu : it is unitless and A lower coefficient of variation indicates less dispersion relative to the mean
            let cv = self.volume_noise_cv;
            let variance_ln = (1.0 + cv * cv).ln();                     // variance of ln(volume) which is Normal
            let mu_ln = mean_vol.max(1.0).ln() - 0.5 * variance_ln;     // Mean of ln(volume) which is Normal
            let lognorm = LogNormal::new(mu_ln, variance_ln.sqrt()).unwrap();  // LogNorma::new() in Rust accepts (mean , std) of the "underlying Normal Distribution" 
            let volume = lognorm.sample(rng).max(1.0); // It draws one random bucket volume from the lognormal, using rng, which advances the generator’s state. Then it floors the result at 1 share.

            // Price: GBM style random walk (no drift assumption; pure execution cost study)
            let shock = price_shock.sample(rng);
            let bucket_open = price;
            price *= (shock - 0.5 * sigma_bucket * sigma_bucket).exp();
            let vwap = 0.5 * (bucket_open + price);   // simple approximation of intra-bucket VWAP

            buckets.push(MarketBucket {
                t: (i as f64 + 0.5) / n as f64,
                volume,
                vwap,
                mid_price: price,
            });
        }

        buckets

    }

    /// Price at which our child order of size qty fills, given the bucket's
    /// pre-impact VWAP, using the empirical square root market impact law
    /// Remember that square root is applied to 'trading rate in the bucket'
    /// 
    ///     impact = epsilon * sigma_bucket * sign * sqrt( qty / bucket_volume)
    /// 
    /// This also nudges the *path* of the simulator permanently (not modeled 
    /// bucket-to-bucket here for simplicity -- see Almgren-Chriss for permanent
    /// impact extension).
    pub fn fill_price_with_impact(&self, bucket: &MarketBucket, qty: f64, is_buy: bool) -> f64 {
        let sign = if is_buy { 1.0 } else { -1.0 };
        let participation = qty / (qty + bucket.volume).max(1.0);          // should this not be (qty/()).max(1.0)????
        let temp_impact = self.temporary_impact_coeff * self.sigma_daily * participation.sqrt();
        bucket.vwap * (1.0 + sign * temp_impact)
    }

    /// Permanent price shift (fraction) after cum_qty shares executed
    pub fn permanent_shift(&self, cum_qty: f64, is_buy: bool) -> f64 {
        let sign = if is_buy { 1.0 } else { -1.0 };
        sign * self.permanent_impact_coeff * self.sigma_daily * (cum_qty / self.adv).sqrt()
    }
}
