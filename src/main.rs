//! --------------------------------------------------------------------------------------
//! Percentage-of_Volume (POV) Equity Execution Algorithm
//! with Monte-Carlo market simulation
//! --------------------------------------------------------------------------------------
//! 
//! A POV (a.k.a "participate", "volume-inline") algorithm slices a parent order into 
//! child orders so that the trader's own volume stays close to a target fraction 'rho'
//! of *total* market volume (own prints included), over a trading horizon, subject to
//! min/max participation bounds and a total quantity constraint
//! 
//! Key Mathematical Subtlety
//! ----------------------------------------
//! Participation rate is almost always defined as 
//!     rho = v / (v + V_m)
//! 
//! where 'v' is the volume the trader executes in the interval and 'V_m' is the volume
//! printed by everyone else in the same interval. If you want 'rho' to hold and you'
//! observe (or forecast) V_m, you must NOT set 'v = rho * V_m' (a common rookie mistake)
//! that undershoots the target because it ignores that your own prints also enter the
//! denominator. Solving this fixed-point equation
//! 
//!     v = rho * (v + V_m)
//! 
//! yields
//! 
//!     v = [rho / (1 - rho)] * V_m
//! 
//! This repo implements the correct formula everywhere.
//! 
//! See the README.md for full references (Almgren-Chriss, Perold Implementation Shortfall,
//! square-root market impact law, etc.)


mod market;
mod pov;
mod stats;

use market::MarketSimulator;
use pov::{PovConfig, PovEngine};
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Result of executing one simulated trading day
#[derive(Debug, Clone)]
pub struct DayResult {
    pub filled_qty: f64,
    pub parent_qty: f64,
    pub achieved_participation: f64,
    pub market_vwap: f64,
    pub our_avg_price: f64,
    pub arrival_price: f64,
    pub vwap_slippage_bps: f64,
    pub implementation_shortfall_bps: f64,
    pub n_child_orders: usize,
}

fn simulate_one_day(
    cfg: &PovConfig,
    market_sim: &MarketSimulator,
    rng: &mut StdRng,
) -> DayResult {
    // let buckets = market_sim.generate_day(rng);
    // let mut engine = PovEngine::new(cfg.clone());

    // // EWMA forecast of next-bucket volume, seeded at the average bucket size
    // let mut ewma_vol = market_sim.adv / market_sim.n_buckets as f64;
    // let alpha = 0.30_f64;  // EWMA smoothing factor        \hat{V}_n = alpha * v_n + (1 - alpha) * \hat{V}_{n - 1}

    // let arrival_price = buckets[0].mid_price;         // mid price of the first bucket of the day
    // let mut n_child_orders = 0usize;                  // 0usize is the integer literal 0 explicitly typed as usize.

    // for (i , bucket) in buckets.iter().enumerate(){
    //     let buckets_remaining = buckets.len() - i;

    //     let qty = engine.size_child_order(ewma_vol, buckets_remaining); // ewma_vol is the forecast of the bucket volume

    //     if qty > 0.0 {
    //         n_child_orders += 1;
    //         let fill_price = market_sim.fill_price_with_impact(bucket , qty , cfg.is_buy);
    //         engine.record_fill(bucket , qty , fill_price);
    //     }
    //     else {
    //         // still need to account for market volume passing by even if we do not trade
    //         engine.observe_market_only(bucket);
    //     }
    //     ewma_vol = alpha * bucket.volume + (1.0 - alpha) * ewma_vol;       // Core EWMA formula
    // }

    // engine.finalize_result(&buckets , arrival_price)


    let mut buckets = market_sim.generate_day(rng);
    let mut engine = PovEngine::new(cfg.clone());

    let mut ewma_vol = market_sim.adv / market_sim.n_buckets as f64;
    let alpha = 0.30_f64;

    let arrival_price = buckets[0].mid_price;
    let mut cum_qty = 0.0;                                       // NEW

    for i in 0..buckets.len() {                                  // CHANGED: index loop
        let shift = market_sim.permanent_shift(cum_qty, cfg.is_buy);  // NEW
        buckets[i].vwap *= 1.0 + shift;                          // NEW
        buckets[i].mid_price *= 1.0 + shift;                     // NEW
        let bucket = buckets[i];                                 // NEW: a copy, not a reference
        let buckets_remaining = buckets.len() - i;

        let qty = engine.size_child_order(ewma_vol, buckets_remaining);

        if qty > 0.0 {
            let fill_price = market_sim.fill_price_with_impact(&bucket, qty, cfg.is_buy);  // &bucket now
            engine.record_fill(&bucket, qty, fill_price);
            cum_qty += qty;                                      // NEW
        } else {
            engine.observe_market_only(&bucket);
        }
        ewma_vol = alpha * bucket.volume + (1.0 - alpha) * ewma_vol;
    }

    engine.finalize_result(&buckets, arrival_price)
}


fn run_monte_carlo(n_sims: usize, cfg: PovConfig, market_sim: MarketSimulator, seed: u64) -> Vec<DayResult> {
    let mut rng = StdRng::seed_from_u64(seed);

    (0..n_sims)
        .map(|_| simulate_one_day(&cfg, &market_sim, &mut rng))
        .collect()
}

fn main() {
    // -------------------------- Parent order & algo configuration ------------------------------
    let cfg = PovConfig {
        parent_quantity: 500_000.0,     // shares to buy
        target_rate: 0.10,              // target 10% of total volume
        min_rate: 0.1,                 // never go below 3% if trading 
        max_rate: 0.25,                 // hard risk cap at 25%                                                own / (own + market) <= max_rate
        min_clip: 100.0,                // smallest child order (odd lot size)
        max_clip_frac_of_bucket: 0.5,   // Liquidity cap: Never try to be more than 50% of a single bucket     own / market <= max_clip
        is_buy: true,
        urgency_catchup: 0.6,           // 0 = no catch-up, 1 = aggressive catch-up
    };

    // ---------------- Market microstructure assumptions ---------------------------------------
    let market_sim = MarketSimulator {
        adv: 5_000_000.0,
        n_buckets: 78,                 // We have 78 5-minute buckets in a trading day
        sigma_daily: 0.02,             // 2% daily volatility
        s0: 100.0,                     // starting price
        permanent_impact_coeff: 0.5,  // eta: permanent impact coefficient
        temporary_impact_coeff: 0.10,  // epsilon: temporary impact coefficient
        volume_noise_cv: 0.35,         // coefficient of variation of bucket volume
    };



    let n_sims = 2_000;
    let results = run_monte_carlo(n_sims , cfg.clone(), market_sim, 42);

    stats::report(&results , &cfg);
}
