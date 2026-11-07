//! Aggreagtion and reporting of Monte Carlo results

use crate::pov::PovConfig;
use crate::DayResult;

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

fn std_dev(xs: &[f64]) -> f64 {
    let m = mean(xs);
    let var = xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len().max(2) - 1) as f64;
    var.sqrt()
}

fn percentile(xs: &[f64], p: f64) -> f64 {
    let mut sorted = xs.to_vec();
    sorted.sort_by(|a , b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}


pub fn report(results: &[DayResult], cfg: &PovConfig) {
    let n = results.len();
    let completion: Vec<f64> = results.iter().map(|r| r.filled_qty / r.parent_qty).collect();
    let participation: Vec<f64> = results.iter().map(|r| r.achieved_participation).collect();
    let vwap_slip: Vec<f64> = results.iter().map(|r| r.vwap_slippage_bps).collect();
    let is_bps: Vec<f64> = results.iter().map(|r| r.implementation_shortfall_bps).collect();
    let n_children: Vec<f64> = results.iter().map(|r| r.n_child_orders as f64).collect();

    println!("====================================================================");
    println!(" POV ALGORITHM — MONTE CARLO SIMULATION REPORT  (n = {} days)", n);
    println!("====================================================================");
    println!(" Target participation rate : {:>6.2}%", cfg.target_rate * 100.0);
    println!(" Parent order size         : {:>12.0} shares", cfg.parent_quantity);
    println!(" Rate bounds               : [{:.1}%, {:.1}%]", cfg.min_rate * 100.0, cfg.max_rate * 100.0);
    println!("--------------------------------------------------------------------");
    println!(" {:<32} {:>10} {:>10}", "Metric", "Mean", "StdDev");
    println!(" {:<32} {:>9.2}% {:>9.2}%", "Order completion rate", mean(&completion) * 100.0, std_dev(&completion) * 100.0);
    println!(" {:<32} {:>9.2}% {:>9.2}%", "Achieved participation rate", mean(&participation) * 100.0, std_dev(&participation) * 100.0);
    println!(" {:<32} {:>9.2}b {:>9.2}b", "VWAP slippage (bps)", mean(&vwap_slip), std_dev(&vwap_slip));
    println!(" {:<32} {:>9.2}b {:>9.2}b", "Implementation shortfall (bps)", mean(&is_bps), std_dev(&is_bps));
    println!(" {:<32} {:>10.1} {:>10.1}", "# child orders / day", mean(&n_children), std_dev(&n_children));
    println!("--------------------------------------------------------------------");
    println!(" Distribution tails:");
    println!("   VWAP slippage   p5 / p50 / p95 (bps): {:>7.2} / {:>7.2} / {:>7.2}",
        percentile(&vwap_slip, 5.0), percentile(&vwap_slip, 50.0), percentile(&vwap_slip, 95.0));
    println!("   Impl. shortfall p5 / p50 / p95 (bps): {:>7.2} / {:>7.2} / {:>7.2}",
        percentile(&is_bps, 5.0), percentile(&is_bps, 50.0), percentile(&is_bps, 95.0));
    println!("   Participation   p5 / p50 / p95 (%)  : {:>7.2} / {:>7.2} / {:>7.2}",
        percentile(&participation, 5.0) * 100.0, percentile(&participation, 50.0) * 100.0, percentile(&participation, 95.0) * 100.0);
    println!("====================================================================");
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_percentile_basic() {
        let xs = vec![1.0 , 2.0 , 3.0 , 4.0 , 5.0];
        assert!((mean(&xs) - 3.0).abs() < 1e-9);
        assert!(std_dev(&xs) > 0.0);
        assert!((precentile(&xs, 50.0) - 3.0).abs() < 1e-9);
    }
}