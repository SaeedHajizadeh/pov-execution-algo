// Core "POV" Scheduling engine
//
// This engine is separated from market simulator: in production, this struct would sit behind a 
// live market-data feed and an order management system instead of a Monte Carlo generator.

// use crate::market::{Market, Order, OrderType, Side};
use crate::market::MarketBucket;
use crate::DayResult;

#[derive(Debug, Clone)]
pub struct PovConfig {
    pub parent_quantity: f64,             // total shares to execute
    pub target_rate: f64,                 // target participation rate rho \in (0.0 - 1.0)
    pub min_rate: f64,                    // floor on instantaneous participation rate once active
    pub max_rate: f64,                    // hard ceiling (risk control / exchange norms)
    pub min_clip: f64,                    // minimum child order size (avoid odd lots) -- odd lot is a lot size < 100 shares
    pub max_clip_frac_of_bucket: f64,     // cap: child order <= this fraction of bucket volume
    pub is_buy: bool,                     // flag indicating if the order is a buy order
    pub urgency_catchup: f64,             // 0..1, how aggressively to catch up if behind schedule (0 = no catchup, 1 = full catchup)
    // urgency_catchup is like damping factor in a control system: like label smoothing in deep learning or ddamping factor in an optimization algorrithm
    // it prevents overreaction to short-term fluctuations in market volume and helps maintain a more stable execution strategy.
}


#[derive(Debug, Clone, Copy)]
pub struct Fill {
    pub t: f64,          // time of fill
    pub qty: f64,        // quantity filled
    pub price: f64,      // price of fill
}

pub struct PovEngine {
    cfg: PovConfig,
    cum_own: f64,         // total shares we have filled so far-- summed via record_fill after each bucket
    cum_market: f64,     // cumulative market volume printed by others
    remaining: f64,
    fills: Vec<Fill>,    // A vector of fills that have occurred during the day; each fill is a tuple of (time, quantity, price)
}



impl PovEngine {
    pub fn new(cfg: PovConfig) -> Self {
        let remaining = cfg.parent_quantity;
        Self {
            cfg,
            cum_own: 0.0,
            cum_market: 0.0,
            remaining,
            fills: Vec::new(),
        }
    }

    /// Converts a target participation rate into the equivalent "of others' volume"
    /// ratio, solving fthe fixed point equation: rho = v / (v + V_m) => v = rho / (1 - rho) * V_m
    #[inline]
    fn rate_to_of_market_ratio(rate: f64) -> f64 {
        let r = rate.clamp(0.0, 0.999);       // avoid divide by zero
        r / (1.0 - r)
    }   

    /// Decide the child order quantity for the upcoming bucket, given the current state of the engine and the market bucket.
    /// 
    /// 'forecast_bucket_volume' is our best estimate (e.g. EWMA) of the market volume (exlcuding our own)
    /// that we expect to see in the *next* bucket, since the actual value is only known after the bucket
    /// closes.
    pub fn size_child_order(&mut self, forecast_bucket_volume: f64, buckets_remaining: usize) -> f64 {
        if self.remaining <= 0.0 {
            return 0.0;
        }

        let rho = self.cfg.target_rate;
        let ratio = Self::rate_to_of_market_ratio(rho);
        let raw_target = ratio * forecast_bucket_volume;   // note that forecast_bucket_volume is the *other* volume, so this is the target child order size for the next bucket

        // -------- Catchup / slow-down controller ----------
        // Compare realized cumulative participation to the target and scale
        // this bucket's order up (if behind) or down (if ahead) to catch up to the target.
        let total_so_far = self.cum_own + self.cum_market;   // total market volume so far (ours + everyone else's)
        let actual_rate = if total_so_far > 0.0 { 
            self.cum_own / total_so_far
        } else {
            rho
        };
        let deficit = rho - actual_rate; // positive = behind schedule; negative = ahead of schedule
        let correction = 1.0 + self.cfg.urgency_catchup * deficit / rho.max(1e-9);  // scale factor for this bucket's order size


        let mut qty = raw_target * correction.max(0.0);  // ensure non-negative

        // -------- Hard participation-rate risk bounds ----------
        let max_qty = Self::rate_to_of_market_ratio(self.cfg.max_rate) * forecast_bucket_volume;
        let min_qty = Self::rate_to_of_market_ratio(self.cfg.min_rate) * forecast_bucket_volume;
        qty = qty.clamp(min_qty.min(max_qty), max_qty);

        // ----- Liquidity cap: Never try to be more than X% of a single bucket, X == max_clip_frac_of_bucket here
        qty = qty.min(self.cfg.max_clip_frac_of_bucket * forecast_bucket_volume);

        // ------ Do not exceed what is left of the parent order
        qty = qty.min(self.remaining);

        // ------ On the very last bucket, dump whatever remains (must complete the parent order)
        if buckets_remaining == 1 {
            qty = self.remaining;
        }

        // ------ Minimum clip filter: avoid sending economincally-noisy and inefficient child orders that are too small (odd lots)
        if qty > 0.0 && qty < self.cfg.min_clip && self.remaining > self.cfg.min_clip {
            // wait for a bucket where the target is big enough
            qty = 0.0;   
        }

        qty.max(0.0)  // ensure non-negative
    }
    // To be completed
    pub fn record_fill(&mut self, bucket: &MarketBucket, own_qty: f64, fill_price: f64) {  // self is PovEngine
        self.cum_own += own_qty;
        self.cum_market += bucket.volume;
        self.remaining -= own_qty;

        self.fills.push(
            Fill {
                t: bucket.t,
                qty: own_qty,
                price: fill_price,
            }
        );
    }

    pub fn observe_market_only(&mut self, bucket: &MarketBucket) {
        self.cum_market += bucket.volume;
    }

    pub fn finalize_result(&self, buckets: &[MarketBucket], arrival_price: f64 ) -> DayResult {   // DayResult is a struct defined in src/main.rs
        let filled_qty: f64 = self.fills.iter().map(|f| f.qty).sum();
        let notional: f64 = self.fills.iter().map(|f| f.qty * f.price).sum();
        let our_avg_price: f64 = if filled_qty > 0.0 { notional / filled_qty } else { arrival_price };   // VWAP

        let total_mkt_vol: f64 = buckets.iter().map(|b| b.volume).sum();
        let total_mkt_notional: f64 = buckets.iter().map(|b| b.volume * b.vwap).sum();
        let market_vwap = total_mkt_notional / total_mkt_vol.max(1.0);

        let achieved_participation = if (filled_qty + self.cum_market) > 0.0 {
            filled_qty / (filled_qty + self.cum_market)
        } else {
            0.0
        };

        let sign = if self.cfg.is_buy { 1.0 } else { -1.0 };
        // VWAP Slippage: positive = we paid more than the market VWAP (bad for a buyer)
        let vwap_slippage_bps = sign * (our_avg_price - market_vwap) / market_vwap * 1e4;   // 1 bps == 1e-4
        // Implementation shortfall vs. the price at the moment we started trading (perold, 1988).
        let implementation_shortfall_bps = 
            sign * (our_avg_price - arrival_price) / arrival_price * 1e4;

        DayResult {
            filled_qty,
            parent_qty: self.cfg.parent_quantity,
            achieved_participation,
            market_vwap,
            our_avg_price,
            arrival_price,
            vwap_slippage_bps,
            implementation_shortfall_bps,
            n_child_orders: self.fills.len(),
        }
    }
     
    
    
}