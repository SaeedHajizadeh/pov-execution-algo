use plotters::prelude::*;


/// pts = (target rate %, mean IS bps, standard error bps)
pub fn plot_is_vs_rate(path: &str, pts: &[(f64, f64, f64)]) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (1000, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let lo = pts.iter().map(|p| p.1 - 1.96 * p.2).fold(f64::INFINITY, f64::min);
    let hi = pts.iter().map(|p| p.1 + 1.96 * p.2).fold(f64::NEG_INFINITY, f64::max);
    let pad = 0.05 * (hi - lo);

    let mut chart = ChartBuilder::on(&root)
        .caption("Implementation shortfall vs target participation", ("sans-serif", 26))
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0..51.0, (lo - pad)..(hi + pad))?;

    chart.configure_mesh()
        .x_desc("Target participation rate (%)")
        .y_desc("Mean IS (bps)")
        .draw()?;

    // 95% confidence interval: mean ± 1.96·SE
    chart.draw_series(pts.iter().map(|&(x, m, se)| {
        ErrorBar::new_vertical(x, m - 1.96 * se, m, m + 1.96 * se, BLUE.filled(), 6)
    }))?;
    chart.draw_series(LineSeries::new(pts.iter().map(|&(x, m, _)| (x, m)), &BLUE))?;
    chart.draw_series(pts.iter().map(|&(x, m, _)| Circle::new((x, m), 3, BLUE.filled())))?;

    root.present()?;
    Ok(())
}